# AUTOFIX — multicast all-gather completion race (LFM2.5-24B-A2B prefill nondeterminism, 1x4 Blackhole ring)

## Failing contract
`ttnn.all_gather` must not complete on a chip before every remote tile of its output has landed.  Observed through the
LFM2.5-24B-A2B prefill (`MeshReduce.prefill` -> `ttnn.all_reduce` -> reduce_scatter + `ttnn.all_gather`, 80 per
prefill): identical 300-token prompts produced different last-position logits in 9/149 (with interleaved decode),
10/149 (with `synchronize_device` before each prefill), 1/149 (no decode) and 1/149 (eager prefill) runs; large diffs
(max |dlogit| 1-4, ~125k of 128k entries, top-1 flips), recurring bit-exact wrong states.

## Diagnosis (AUTODEBUG.md H1, verified against source)
`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp` (S <= ~512 rows for this shape on
Blackhole) enables route alternation on even rings (`axis_load_balance`): data packets alternate between the primary and
the alternate fabric route (`kernels/multicast_common.hpp`), but the completion credit (`done_sem`) is sent only on the
primary routes (`sem_route_id`, `kernels/multicast_reader.cpp` / `multicast_writer.cpp`).  Fabric ordering covers only
the same route, so on a 1x4 ring a receiver can count its `num_devices - 1` credits while the odd packets from the chip
two hops away (sent over the alternate route) are still in flight; the next op reads stale tiles.

## Fix (minimum change)
`all_gather_multicast_factory.cpp`: `axis_load_balance = false` — every data packet takes the same primary routes as its
completion credit, so credit-after-data ordering holds per receiver.  (A complete alternative that keeps load balancing
would send completion credits on both routes and raise `barrier_wait_value` accordingly; not needed for correctness.)

## Validation (1x4 Blackhole p150b, FABRIC_1D_RING)
| check | unfixed | fixed |
|---|---|---|
| op regression `models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_ccl_ring_all_gather.py` (traced all_gather -> add, fresh random input per replay, 1000 replays x 4 chips, rows 128 / 512) | **37/4000 stale outputs** at 512 rows (max err 5.6), 128 rows passed | **2 passed** |
| model stress, 300-token prompt (bucket 512, multicast AG), decode interleaved | 9/149, 10/149 (sync) | **0/299** |
| model stress, 100-token prompt (bucket 128, multicast AG) | n/a | **0/199** |
| control: 900-token prompt (bucket 1024, unicast AG), unfixed code | 0/199 | — |
| perf (readiness_driver perf, C1): TTFT 128/1024/2048/4100, token-out b1, batch step b4/b8/b32 | 65.7/97.5/135.0/295.9 ms, 107.97 t/s/u, 13.63/19.09/26.10 ms | 65.6/97.4/135.1/296.1 ms, 107.96 t/s/u, 13.63/19.08/26.10 ms |

Logs: `models/autoports/liquidai_lfm2_5_24b_a2b/doc/vllm_integration/logs/{test_ccl_ring_all_gather_unfixed,test_ccl_ring_all_gather_fixed,diag_stress_*,perf_after_allgather_fix}.log`.
Build: `cmake --build build_Release --target install` (incremental, 40 s).

## Unresolved / risk
- Other users of the multicast all-gather on even rings / tori lose the route load balancing (perf only; this model:
  no measurable change).  The 2D (`ns_load_balance`) case shares the same flag and is also disabled.
- Why interleaved decode raises the visible rate (9/149 vs 1/149) is not proven (AUTODEBUG.md §4: likely visibility of
  stale bytes); the fixed runs include decode interleaving and show 0.
- Not pushed upstream; a tt-metal issue/PR is the user's call.
