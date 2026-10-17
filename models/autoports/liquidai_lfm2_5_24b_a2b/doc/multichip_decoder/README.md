# LFM2.5-24B-A2B multichip decoder (stage 4): 1x4 Blackhole p150b ring, TP = 4

Code: `tt/multichip_decoder.py` (`MultichipDecoder`, `MC_POLICY`, `MeshPlan`, `make_multichip`,
`mesh_device_params`); tests: `tests/test_multichip_decoder.py`; history and every rejected alternative:
`work_log.md` (M-0 ... ); evidence paths below are relative to this directory.

## Baseline and target
- Single-chip baseline: stage-3 `OptimizedDecoder` + `DEFAULT_POLICY` (commit `06705dad`), the measured single-chip
  optimum.  The multichip modules ARE the stage-3 modules, built on a chip-local slice of the layer (so every
  stage-3 program config / fused path applies to the local shapes) with the per-chip weights placed by
  `ShardTensorToMesh`.
- Target: QuietBox, 4x Blackhole p150b (cluster type `P150_X4`), one 1x4 mesh, `FABRIC_1D_RING`
  (router payload 8192 B), CCL topology Ring, 2 links (`ttnn.get_num_links` = 4; 2..4 measure the same,
  M-9).  The code supports exactly this mesh (asserted).

## Parallelization (per chip, d = 0..3)
| part | strategy | per-chip shape | why / rejected (M-1, M-2, M-8) |
|---|---|---|---|
| attention q/k/v | column-parallel by heads: q heads 8d..8d+7 with their GQA kv heads 2d, 2d+1 | wqkv [2048, 768] = q 512 + k 128 + v 128 | KV cache must be sharded: replicated 32 x 131072 KV = 45.6 GB/chip |
| paged KV cache | local kv heads | K, V [blocks, 2, 64, 64] BFP8 each | page tables / positions replicated (global block ids) |
| SDPA | local heads | 8 q heads / 2 kv heads | - |
| attention out_proj | row-parallel + all-reduce | wo [512, 2048] | - |
| ShortConv | channel-parallel: B / C / x groups, taps and state sliced | in_proj [2048, 1536], taps / state [.., 512], out_proj [512, 2048] | replicated conv: no speedup, same reduce count per layer |
| dense MLP (layers 0, 1) | w1 / w3 column-, w2 row-parallel | w1 / w3 [2048, 2944] (decode copy N-padded 3072), w2 [2944, 2048] (decode K-padded 3072) | - |
| router / top-k / routing glue | replicated (identical inputs -> identical selections on all chips) | [2048, 64] (+ DS [2048, 256] decode) | sharded router + logits gather: extra CCL per MoE layer |
| experts, decode + short prefill | tensor-parallel inside every expert (I 1536 -> 384) | stacked [1, 64, 2048, 768] (up \| gate), [1, 64, 384, 2048] | EP: the static compact bucket path (b*K indexed slots) would still read b*K full experts per chip; 64-wide path with E[max chips] 2.1 of 4 experts at b1 (M-2) |
| experts, prefill token dispatch | expert-parallel: experts 16d..16d+15 at full width | 16 x (gate / up [2048, 1536], down [1536, 2048]) | TP shapes run the dispatch FFN op at 20 vs 45 TFLOP/s: 1907 vs 860 us per chip (M-8) |
| norms, RoPE tables, page tables, cur_pos, slot masks | replicated | - | - |

Collectives: exactly one all-reduce per sublayer (mixer, FFN) = 2 per layer; nothing else crosses chips.
- decode: `all_reduce_async` on width-sharded L1 [32, 2048] (8 cores), scratch [32, 8192] allocated per call
  (allocated inside the trace like any intermediate; limitation 4), 8.9 us on device per all-reduce;
- prefill: `ttnn.all_reduce` (composite reduce-scatter + all-gather) in DRAM, ~129 us per 2048-row all-reduce;
- residual contract: replicated [1, 1, rows, 2048] bf16, bitwise identical on every chip (checked); decode keeps it
  width-sharded in L1 on the 8-core grid that is also the operator-norm, ffn-norm, out-projection output and
  all-reduce layout (no reshard between producer, all-reduce, residual add and the next norm).

Topology table (M-1, traced, bf16, 2 links): decode [32, 2048] all_reduce_async 16.7 us / composite all_reduce 20.2 /
RS + AG 47.4 / AG + local sum 36.7; prefill [2048, 2048] composite all_reduce 157.2 / RS 120.2 + AG 99.2.
A fractured residual still needs RS + AG around every norm (both norms consume whole rows): 44 vs 17 us decode,
219 vs 157 us prefill per sublayer -> the replicated residual with one all-reduce is the cheapest contract MEASURED here.
NOT yet measured (rejected by arithmetic only, so open): the fused-CCL rows -- `matmul_reduce_scatter_async` on the
row-parallel wo / out_proj / w2 and `all_gather_matmul_async` on the column-parallel qkv / in_proj / w1-w3 -- and the
residual-sharded family that would make them worth it (RS -> fractured residual add -> `rms_norm_pre_all_gather` /
stats AG / `rms_norm_post_all_gather` -> AG fused into the next column-parallel matmul).  Batch-1 decode is op-count
bound, so these fusions are the first measurement of the optimized-multichip stage (they remove ops, not bytes).

## Tensor / program-config table (per chip; BFP8 weights unless noted)
| tensor | single chip | per chip | mapper / padding | config change (M-4 sweeps) |
|---|---|---|---|---|
| wqkv (DS) | [2048, 3072] (8,8,2) | [2048, 768] | host regroup [q_d \| k_d \| v_d], Shard(-1) | (8,8,3): 10.5 us |
| wo (DS) | [2048, 2048] (8,8,2) | [512, 2048] | Shard(-2) | (8,2,2): 9.1 us, output on the residual grid |
| conv in_proj (DS) | [2048, 6144] (32,2,3) | [2048, 1536] | regroup [B_d \| C_d \| x_d], Shard(-1) | (8,8,2): 11.5 us |
| conv out_proj (DS) | [2048, 2048] (8,8,2) | [512, 2048] | Shard(-2) | (8,2,2): 9.1 us |
| conv taps / state | [1,1,1,2048] / [1,1,32,2048] | [..., 512] | Shard(-1) / local zeros | slot mask width 512 |
| dense w1 / w3 decode (DS) | [2048, 12288 pad] (32,2,3) | [2048, 3072 pad of 2944] | per-chip zero pad, Shard(-1) | (8,8,2): 21.3 us each |
| dense w2 decode (DS) | [12288 pad, 2048] (48,8,2) | [3072 pad, 2048] | per-chip K pad, Shard(-2) | (8,6,2): 19.9 us |
| dense w1 / w3 / w2 prefill | [2048, 11776] / [11776, 2048] | [2048, 2944] / [2944, 2048] (DS (46,2,2)) | Shard(-1) / Shard(-2) | prefill_2d_config recomputed |
| experts stacked | [1, 64, 2048, 3072] / [1, 64, 1536, 2048] | [1, 64, 2048, 768] / [1, 64, 384, 2048] | per-chip [up_d \| gate_d], Shard(-1) / Shard(-2) | scan (8,3,32,1) / (8,8,12,1); compact (8,3,32,1) / (8,8,12,1) |
| experts prefill dispatch | 64 x [2048, 1536] / [1536, 2048] | 16 x full | Shard(0) of the 4 chips' experts per local index | local table of 32 entries (dummy + 16 + unused) |
| KV cache | [blocks, 8, 64, 64] | [blocks, 2, 64, 64] | allocated per chip | - |
| router, bias, norms, RoPE | - | same | Replicate | - |

## Results (final code: md5 `logs/final_code_md5.txt`; real weights)

### Correctness
- vs HF, `tests/test_multichip_decoder.py` (TT_METAL_TRACE_ALLOC_TRACKING=1, program-cache allocations included):
  **95/95** (`logs/m2_full_suite.log` 93 + the rewritten near-tie control `logs/m2_near_tie_control.log` + the stacked
  test `logs/stacked_layers.log`):
  single user 1/31/32/64/65/96/97/300@128/300/2049 (all three MoE prefill paths), batch 32, slot reuse, increasing
  lengths through ONE decode trace with prefill between replays, full 131072 context with and without bucket (131069-token
  prompt, KV for 32 x 131072), bucketed decode b1/b2/b4 (idle slot inside)/b8/b16, multi-bucket traces (buckets
  1/4/32 coexisting, holes, slot reuse), deep layers on real hidden states (1, 6, 7, 15, 23, 31, 38, 39), router
  near-tie control, perf / determinism / no-host-fallback (b32, b1, b1 bucket 1, b4 bucket 4), memory capacity.
  Every case within ~1e-4 of the stage-3 PCC except router near-ties (M-10): the multichip attention layer's decode
  router scores carry ~12 % more noise (max |score - HF| 6.8e-3 vs 4.9e-3 single chip), so ties up to ~3e-3 can
  flip; accepted only when one swap at that position explains the update: 4 acceptances in m2_full_suite (slot 9 pos 33
  gap 3.4e-3 -> 0.99981; slot 13 pos 769 gap 2.3e-4 -> 0.99979, the stage-3 case; slot 2 pos 122 gap 2.5e-4 -> 0.99985;
  slot 6 pos 357 gap 3.6e-4 -> 0.99984; the other logged candidates were tried and rejected).
- vs the single-chip TTNN baseline on identical inputs (`test_multichip_vs_single_chip`, all kinds, bucket none / 1):
  output PCC >= 0.99996, update >= 0.9986; per-chip KV heads 2d, 2d+1 and ShortConv channels 512d.. match the
  single-chip cache / state slices (PCC >= 0.999996); decode output bitwise identical on the 4 chips.
- Paged KV on the mesh: page tables / cur_pos replicated, local caches [65537, 2, 64, 64] per K / V, slot reuse and
  multi-trace schedules above; non-aligned lengths 1, 31, 33, 65, 77, 97, 300, 2047, 2049, 4100, 131069.
- Watcher (`TT_METAL_WATCHER=10`, nothing disabled): 33 tests (prefill paths, perf b32/b1/bucket 1/4, multi-bucket
  traces, multichip-vs-single-chip), 0 error lines, 10212 records per device (`logs/watcher/`).
- Determinism: measured prefill == compile run bitwise; two decode replays from the same state equal; finding M-5.
- Stacked-decoder contract (`test_multichip_stacked_layers`, `logs/stacked_layers.log`): real consecutive layers
  1 (conv_dense) -> 2 (attn_moe) -> 3 (conv_moe) in ONE decode trace with the residual converted to `resid_mem` once
  at entry: every layer output is exactly `resid_mem` (asserted), active rows bitwise equal to the DRAM-entry path,
  vs the chained HF layers prefill PCC >= 0.99947 / decode >= 0.99995, stack update >= 0.99649 / 0.99978.

### Performance (Tracy device time, max over the 4 chips, per decoder layer; `tracy_final/`, `tracy_final_runs.txt`)
| kind | mode | 1 chip (stage 3) | 4 chips | speedup | efficiency |
|---|---|---|---|---|---|
| attn_moe | decode b32 | 1991.9 | 718.3 | 2.77x | 69 % |
| attn_moe | decode b4 bucket | 742.9 | 384.4 | 1.93x | 48 % |
| attn_moe | decode b1 bucket | 348.3 | 242.2 | 1.44x | 36 % |
| attn_moe | decode b1 no bucket | 1052.1 | 436.0 | 2.41x | 60 % |
| conv_moe | decode b32 | 1567.1 | 673.8 | 2.33x | 58 % |
| conv_moe | decode b4 bucket | 664.0 | 332.1 | 2.00x | 50 % |
| conv_moe | decode b1 bucket | 299.7 | 208.2 | 1.44x | 36 % |
| conv_moe | decode b1 no bucket | 497.4 | 368.1 | 1.35x | 34 % |
| conv_dense | decode (any batch) | 310.5 | 150.9 | 2.06x | 51 % |
| attn_moe | prefill 2048 | 6631.4 | 3502.9 | 1.89x | 47 % |
| conv_moe | prefill 2048 | 6192.3 | 3191.2 | 1.94x | 49 % |
| conv_dense | prefill 2048 | 3688.3 | 1415.0 | 2.61x | 65 % |

40-layer decoder stack per decode step (2 conv_dense + 28 conv_moe + 10 attn_moe): batch 1 with bucket 1
8.55 ms (single chip 12.5 ms, 1.46x); batch 32 26.4 ms (64.4 ms, 2.44x); prefill 2048 ~127 ms device (247 ms).
Wall clock (50 traced replays, `logs/final_perf_wall50.log`): decode attn b1 bucket 266 us, conv_moe 248, conv_dense 185,
b32 750 / 704 us; host - device ~20-30 us per single-layer replay (paid once per token for a whole-model trace).
Eager prefill wall 2.8-9.4 ms per layer vs 1.4-3.5 ms device: host dispatch, a full-model item (traced prefill).

### tt-perf-report findings (final reports, device 0)
| case | total | CCL | experts | matmul | SDPA | eltwise | data movement | norms | topk |
|---|---|---|---|---|---|---|---|---|---|
| attn decode b1 bucket | 241.6 | 18.3 (8 %) | 40.8 | 39.7 | 20.0 | 38.0 | 44.5 | 20.2 | 15.8 |
| conv_moe decode b1 bucket | 207.8 | 18.5 (9 %) | 41.9 | 40.3 | - | 54.8 | 19.4 | 12.7 | 15.8 |
| attn decode b32 | 717.3 | 19.6 (3 %) | 403.9 | 37.9 | 63.3 | 85.5 | 56.6 | 20.9 | 15.8 |
| conv_moe decode b32 | 673.4 | 20.3 (3 %) | 443.4 | 38.9 | - | 102.0 | 26.6 | 12.8 | 15.8 |
| conv_dense decode | 150.3 | 18.2 (12 %) | - | 81.1 | - | 23.7 | 14.6 | 12.7 | - |
| conv_moe prefill 2048 | 3165.8 | 265.7 (8 %) | 930.2 | 221.9 | - | 399.2 | 1059.8 | 110.3 | 17.9 |
| attn prefill 2048 | 3488.4 | 263.8 (8 %) | 927.9 | 182.7 | 468.4 | 290.0 | 1041.7 | 133.6 | 17.9 |
| conv_dense prefill 2048 | 1395.9 | 258.5 (19 %) | - | 589.2 | - | 317.6 | 121.0 | 109.6 | - |

- Communication: 2 all-reduces per layer, 9 us each in decode (8-12 % at batch 1, 3 % at b32), ~130 us each in prefill;
  links are not the limit (2/3/4 links identical, M-9).
- DRAM: b32 decode is the expert weights (gate_up 24 cores x 1 N tile at ~256 GB/s per chip, sweeps/local_moe_*);
  the dense / attention / conv projections run 9-21 us each at their DS-sharded optimum (M-4).
- Compute: prefill experts 930 us at 45 TFLOP/s per chip (EP dispatch, M-8).
- Data movement (per-device report, device 0, individual ops): prefill MoE dispatch glue (gathers 358 us, FFN-output
  untilize 153 us, combine 137 us) is replicated
  token-dispatch work; decode batch-1 is op-count bound (~95 ops/layer at 1-3 us).
- SLOW matmul rows in the decode reports: row-parallel wo / conv out_proj 32 x 512 x 2048 (latency floor of a
  16-K-tile DS matmul, 9 us, 22 % of DRAM BW), the DS router 32 x 2048 x 256 and the routing 0/1 matmuls (as in stage 3, README of
  optimized_decoder); prefill: the 2048 x 2048 x 1536 in_proj / 2048 x 512 x 2048 out_proj at ~50 % of HiFi2 peak.

## Limitations
1. Batch-1 decode is op-count bound: TP divides bytes and FLOPs but not the ~95 ops per layer, so b1 speedup is 1.44x.
   Fusing the replicated routing glue / attention plumbing is the next lever (optimized-multichip stage).
2. Prefill MoE: the replicated token-dispatch glue is the main prefill item -- (prefill_perf_report.csv, merged rows,
   by op code) gathers 364 + elementwise 338 +
   untilize 213 + post-combine 138 + reshape 106 us = ~1.1 ms of 3.2 ms per conv_moe chunk, identical on every chip,
   vs the EP expert FFN 940 us and CCL 8 %; it caps prefill speedup at ~1.9x.  Next-stage candidates: bound the
   gathers / untilize to the local rows, remove the device reshape, LoFi unified (time-neutral in stage 3).  The hybrid
   routed-expert op cannot run with any L1 allocation (arena = all of L1 above the base, M-3) -> unified op.
3. Router near-ties: the multichip attention layer adds router-score noise (M-10); accepted with a 1e-2 window +
   single-swap evidence.  The root cause ("8-head local SDPA / head-split core mapping") is a HYPOTHESIS -- fp32
   partials were ruled out, no component comparison yet; full-model top-1 agreement decides whether anything must change.
4. Decode all-reduce scratch allocated per call.  Persistent L1 buffers (scratch, the attention concat zero buffer,
   `moe_route_v3_l1` constants) were blocked only by the hybrid prefill op, which is not used here: an open
   optimized-multichip candidate.  `l1_small_size` 24576 is required (CCL semaphores, M-10).
5. A 1x1 submesh of the open mesh breaks lockstep allocation (hang in the next all-reduce, M-6): never mix.
6. Supports exactly the 1x4 P150_X4 ring (asserted); bucket 8 decode uses the 64-wide expert path (compact paths
   are buckets 1/2/4) -- relevant for the expected ~6 concurrent users.
7. Op issue candidates (repros in `repro/`, drafts in `op_issues.md`, NOT filed upstream -- filing is left to the
   user): hybrid MoE arena sizing; sparse_matmul indexed in0_block_w 64 nondeterministic inside the decoder trace
   (M-5); all_reduce_async rejects DRAM / interleaved inputs on Blackhole.

## Commands
```bash
pytest -v --timeout=4000 models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_multichip_decoder.py      # (TT_METAL_TRACE_ALLOC_TRACKING=1)
LFM2_MC_DECODE_ITERS=50 pytest -q models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_multichip_decoder.py -k perf
doc/multichip_decoder/repro/tracy_mc.sh <outdir> <kind> <active> <bucket|None> <tag> [policy-override-json]
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=<dir> pytest ... -k "<selection>"
LFM2_POLICY_OVERRIDE='{...}' / LFM2_PLAN_OVERRIDE='{...}'   # A/B overrides, logged by the tests
```

## Evidence index
`work_log.md` (M-0 ... M-10), `sweeps/` (CCL probe, per-chip DS / MoE sweeps, EP FFN, memory_capacity.json),
`repro/` (every probe / diagnostic), `logs/` (test runs, watcher, triage of the submesh hang in `triage/`),
`tracy_final/` (+ exploratory `tracy_v1/`), `../context_contract.json` ("multichip_decoder").
