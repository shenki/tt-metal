# Stage 4 work log: multichip decoder (1x4 Blackhole p150b ring, TP = 4)

Baseline: stage-3 `OptimizedDecoder` + `DEFAULT_POLICY` (commit `06705dad`), the measured single-chip
optimum (README of `doc/optimized_decoder`).  Paths below are relative to `doc/multichip_decoder/`.

## M-0 Hardware / setup (2026-10-17)
- Cluster type `P150_X4` (4x p150b), compute grid 11x10, 8 DRAM banks per chip, `ttnn.get_num_links(mesh, 1)` = 4.
- Fabric: `FABRIC_1D_RING` with router payload 8192 B before the mesh opens (pytest `device_params` =
  `multichip_decoder.mesh_device_params()`), CCL topology `Ring` (`models/common/modules/tt_ccl.py`
  `default_topology` picks Ring for P150_X4).
- Infrastructure: after the reboot a Qwen vLLM container held the chips (stopped with the user's approval);
  the first 1x4 open then timed out on ethernet core 31-25; one `tt-smi -r` fixed it (smoke 1x1, 1x4 OK).

## M-1 Collective topology probe (`repro/ccl_topology_probe.py` -> `sweeps/ccl_topology_probe.log`)
Restoring a replicated residual from four per-chip partials, traced replay wall / 50, bf16:

| variant | decode [32,2048], 1 link | 2 links | prefill [2048,2048], 1 link | 2 links |
|---|---|---|---|---|
| `ttnn.all_reduce` (composite RS + AG), DRAM | 21.5 | 20.2 | 191.3 | **157.2** |
| `all_reduce_async` + persistent buffer, L1 width-sharded 32 cores | 20.4 | **16.7** | n/a (L1 width-sharded only) | n/a |
| RS(dim 3) only (fractured residual) | 21.6 | 28.6 | 132.8 | 120.2 |
| RS + AG(dim 3) | 35.2 | 47.4 | 250.0 | 201.8 |
| AG(dim 2) of partials + local sum | 32.4 | 36.7 | - | - |
| AG(dim 3) of an H/4 fractured tensor | 22.4 | 30.2 | 128.5 | 99.2 |

- `all_reduce_async` rejects DRAM and interleaved L1 on Blackhole (`all_reduce_async_device_operation.cpp:24-56`:
  width-sharded input / buffer / output only); its output may use a subset of the buffer's cores.
- A fractured residual needs RS now and AG before the next norm / column-parallel matmul: decode 21.6 + 22.4
  = 44 us vs 16.7 us for one all-reduce; prefill 120 + 99 = 219 us vs 157 us.  Fused alternatives
  (`matmul_reduce_scatter_async`, `all_gather_matmul_async`) still need that pair per sublayer (the norm needs
  the whole row); so the replicated residual + one all-reduce per sublayer is the cheapest contract here.

## M-2 Expert placement: tensor-parallel inside each expert (not expert-parallel)
- Stage-3 bucket decode (b <= 4) runs the experts compactly: `sparse_matmul` in indexed mode over a STATIC
  b*K slot list (no sparsity skip).  Expert parallelism (16 experts / chip) would leave every chip reading
  b*K full experts (non-local slots cannot be skipped in indexed mode) = no decode gain at the latency target;
  the 64-wide sparsity path would read only local experts but with E[max over chips] ~2.1 of 4 experts at b1
  (worst case 4) and a 16-wide routing remap.
- Tensor parallel over the expert intermediate (I = 1536 -> 384 per chip): every chip runs the same selected
  experts on a quarter of each (balanced for any routing, the compact path reads b*K quarter-experts per chip),
  router / top-k replicated (identical inputs -> identical selections on every chip), one all-reduce of the
  down partials.  Same weight bytes per chip as EP; the routing glue is replicated in both designs.

## M-3 First implementation (`tt/multichip_decoder.py`)
- Every module is the stage-3 module built on chip 0's slice of the layer (`local_state_dict`, `local_args`):
  all program configs / fused paths apply to the local shapes; the per-chip weights then replace the
  chip-0-shaped tensors via `ShardTensorToMesh` (`_shard_like`, same dtype / layout / memory config).
- Row-parallel DRAM-sharded projections (wo, conv out_proj, dense w2) return their width-sharded L1 output
  (`_PartialOut`); MoE partials are resharded to the residual grid; `MeshReduce.decode` =
  `all_reduce_async` (scratch buffer allocated per call), `MeshReduce.prefill` = `ttnn.all_reduce`.
- Residual: decode width-sharded L1 [32, 2048] on the 8-core ffn-norm grid (input and output of the layer),
  `h = x + AR(mixer partial)` and `out = h + AR(ffn partial)` are sharded adds; prefill DRAM interleaved.
- Harness: `DecodeInputs(slot_mask_width=...)` (default H): the multichip ShortConv state is [32, 512] per chip.
- Blocker found: `hybrid_routed_expert_moe` (stage-3 prefill expert op) sizes its L1 arena as all of L1 above
  the allocator base (`hybrid_program_factory.cpp:2882-2900 arena_bytes_for`).  With `l1_small_size` 24576 the
  bank is 24576 B short; with 0 it still fails next to the CCL global semaphores (2688 B allocated per bank):
  "Not enough space to allocate 128578560 B L1 buffer across 88 banks ... allocated: 2688 B".  -> MC_POLICY
  `dispatch_ffn="unified"` (stage 3: hybrid 3101 vs unified 3494 us per 2048-row MoE prefill on one chip;
  ~1/4 of that gap per chip under TP) and `l1_small_size=0`.
- First results (real weights): single-user matrix 30/30 vs HF (`logs/dev_single_user2.log`, trace
  allocation tracking on); per case within ~1e-4 of the stage-3 single-chip PCC (same worst case, conv_moe
  prompt 32 update 0.993366 on both).
- First perf (50 traced replays, wall): attn_moe b1 bucket 279 us (1 chip 371), conv_moe b1 bucket 267 (328),
  conv_dense 209 (326), attn b32 909 (2021), conv_moe b32 876 (1584) (`logs/dev_perf_wall50.log`).  Tracy attn b1
  bucket: 260 us device on every chip, AllReduceAsync 8.9 us each (`tracy_dev/attn_moe/a1_b1/v0`).

## M-4 Per-chip program-config sweeps (`repro/local_ds_sweep.py`, `repro/local_moe_sweep.py`; 1 chip each)
DRAM-sharded decode projections (traced, BFP8, HiFi2; `sweeps/local_ds_c*.log`):

| role (per chip) | first guess | best | adopted |
|---|---|---|---|
| qkv [2048, 768] | (16,4,1) 15.0 us | (8,8,3) 10.5 | (8,8,3) |
| wo [512, 2048] | (8,2,2) 9.1 | (8,2,2) 9.1 | (8,2,2) |
| conv_in [2048, 1536] | (32,2,3) 30.0 | (8,8,2) 11.5 | (8,8,2) |
| conv_out [512, 2048] | (8,2,2) 9.1 | (8,2,2) 9.1 | (8,2,2) |
| w1 / w3 [2048, 2944 -> 3072] | (32,2,3) 30.7 | (8,8,2) 21.3 | (8,8,2) |
| w2 decode [3072, 2048] | (8,6,2) 19.9 | (8,6,2) 19.9 | (8,6,2) |

Expert `sparse_matmul` (indexed, 4 slots = bucket 1; `sweeps/local_moe_*_idx_4.jsonl`): gate_up [2048, 768]
(8,3,64,1) 30.0 us vs (8,3,8,1) 42.9 (N = 24 tiles caps it at 24 cores, ~220 GB/s); down [384, 2048]
(8,8,12,1) 17.1 us (best).  16 slots (bucket 4): gate_up (8,3,32,1) 104.6 us.
Scan mode at b32 (56 active experts; `sweeps/local_moe_*_scan_56.jsonl`): gate_up (8,3,32,1) 364.6 us vs
(8,3,8,1) 592.8; down (8,8,12,1) 178.9 us (best).  Adopted: `moe_gate_up_cfg` (8,3,32,1), compact (8,3,32,1).
- qkv on 8 cores gives 96-column shards (heads straddle cores); `nlp_create_qkv_heads_decode` with the sharded
  input still produces the right heads (attention tests pass, `logs/dev_cfg2.log`).
- All DRAM-sharded inputs / outputs now live on the 8-core residual grid: the operator norm runs on the residual
  layout directly (no reshard), wo / conv_out / w2 partials need no reshard before the all-reduce.

## M-5 Determinism finding: compact gate_up in0_block_w 64
`test_multichip_perf_determinism_no_fallback[4-4-conv_moe]` failed "traced decode replay is not deterministic"
with compact gate_up (8,3,64,1); bisect (`logs/det_bisect.log`, 2 runs each): (8,3,64,1) fail/fail, (8,3,32,1)
pass/pass, (8,3,8,1) pass/pass, reverting conv_in to (32,2,3) still fails.  The op alone (eager, nothing else in
L1) is bitwise stable at 64 (`repro/sparse_mm_idx_bw64_determinism.py` -> `sweeps/sparse_mm_idx_bw64_determinism.log`),
so the failure needs the decoder's L1 state (a 64-tile double-buffered in0 CB = 256 KB per core).  Adopted 32
(30.5 vs 30.0 us at 4 slots).  Upstream candidate: in0 CB sizing / placement of sparse_matmul indexed mode.

## M-6 Hang: a 1x1 submesh of the open mesh
First version of `test_multichip_vs_single_chip` built the single-chip baseline on `mesh_device.create_submesh(1x1)`.
The run hung in the multichip decoder's dense-w2 decode all-reduce (op 472): tt-triage (`triage/tt-triage.txt`,
`triage/triage-summary.txt`) shows `AllReduceAsyncDeviceOperation` RUNNING on devices 0, 1, 2 with
`reduction_receiver` in `noc_semaphore_wait`, device 3 idle.  The submesh allocates on device 0 only, so mesh
(lockstep) allocations no longer match across chips.  Recovery: kill the exact PID, `tt-smi -r`, smoke 1x1 / 1x4 OK.
Fix: the baseline is the stage-3 `OptimizedDecoder` replicated on the same 1x4 mesh (every chip computes the full
single-chip layer, no collectives); compare chip 0 (and check the per-chip state slices).  No hang since.
Second finding from this test (a stage-3 latent bug): `optimized_decoder._ZERO_ROWS` (bucketed attention concat
zero buffer) was cached per device only; a second decoder with other head counts on the same mesh reused the wrong
width (`Number of shards along width 32 must not exceed number of cores 8`).  Key now (device, rows, NH*HD).

## M-7 Multichip vs single-chip (`test_multichip_vs_single_chip`, `logs/dev_vs_sc.log`)
4 users (prompts 1, 65, 300, 2049; decode 3-4 steps) per kind + bucket 1 (prompt 300): output PCC >= 0.99996,
update PCC >= 0.99857 (prefill) / 0.99989 (decode); every chip's KV-cache heads (chip d = heads 2d, 2d+1 of the
single-chip cache) and ShortConv state (channels 512d..512d+511) PCC >= 0.999996; decode output bitwise identical
on all 4 chips (replicated residual contract).

## M-8 Prefill experts: expert-parallel dispatch
Tracy prefill 2048 (`tracy_v1/*/a1_b1/cfg3`): conv_moe 4380 us vs 6192 single-chip (1.41x only):
UnifiedRoutedExpertFfn 1968 us (single chip 3494 at full I).  Isolated op (`repro/prefill_expert_ffn_tp_vs_ep.py`
-> `sweeps/prefill_expert_ffn_tp_vs_ep.log`, 128 tokens per expert, HiFi2):

| per chip | experts x I | pairs | us | TFLOP/s |
|---|---|---|---|---|
| single chip | 64 x 1536 | 8192 | 3413 | 45.3 |
| TP 4 (MC first version) | 64 x 384 | 8192 | 1907 | 20.3 |
| EP 4 | 16 x 1536 | 2048 | 860 | 45.0 |

-> `MCMoE` prefill token dispatch is expert-parallel: chip d holds experts 16d..16d+15 at full width (same bytes as
64 quarter experts; the decode stacked copy stays TP), a 32-entry local table (entry 0 = dummy one-tile region with
zero input rows, output exactly 0; 1..16 local; rest count 0), non-local pairs get a dummy destination (gather back
exact zeros) and scatter source 0.  First version pointed every non-local reference at ONE zero row: the two
gathers went 480 -> 1138 us (hot DRAM page, `tracy_v1/conv_moe/a1_b1/ep_prefill`).  Spread over 32 zero rows
(destination p mod 32, unused buffer rows r mod 32 via a max with a spread pattern; written local sources are >= 32):
gathers 358 us.  Prefill: conv_moe 4380 -> 3170 us, attn_moe 4690 -> 3488 us (`.../ep_prefill_spread`); FFN 918-935
us on the four chips for real routing.  PCC unchanged (`logs/dev_ep_prefill2.log`).

## M-9 Links
`ttnn.get_num_links` = 4 on the ring axis; decode all_reduce_async 16.6 / 16.1 / 16.2 us and prefill all_reduce
157.3 / 161.9 / 154.1 us at 2 / 3 / 4 links (`sweeps/ccl_topology_probe_links234.log`): latency-bound, keep 2.

## M-10 Full-suite round 1 findings (`logs/m1_full_suite.log`: 84 passed, 9 failed)
- 8 MoE failures "Statically allocated circular buffers ... clash with L1 buffers ... L1 buffer allocated at
  1045760": with `l1_small_size=0` the prefill composite all-reduce allocates its semaphores in plain L1
  ("Allocating semaphores in L1 ... Configure an L1_SMALL region"), where later static CBs need the space.
  The zero L1_SMALL was only needed by the hybrid MoE op, which is no longer used -> `l1_small_size=24576`
  restored; the 8 cases pass (`logs/m1_rerun_l1small.log`, no L1-semaphore warning).
- Router near-ties.  attn_moe increasing-lengths slot 9 prompt 33: decode update 0.981865 vs HF (deterministic,
  3/3 runs, `logs/m1_incr_attn_repeat.log`); HF's 4th/5th biased-score gap at decode position 33 is 3.44e-3;
  vs HF with experts 4/5 swapped at position 33 the update PCC is 0.999813 (`repro/near_tie_slot9_p33.py` ->
  `logs/near_tie_slot9_p33.log`): exactly one top-k flip.  Router-score noise measured on identical inputs
  (`repro/router_score_noise.py`, 50 decode rows, eager; single-chip = stage-3 decoder replicated on the mesh with
  the same expert-FFN op):

  | layer | decoder | max \|score - HF\| | p99 | median | top-4 set != HF |
  |---|---|---|---|---|---|
  | 2 (attn_moe) | single chip | 4.92e-3 | 4.53e-3 | 1.52e-3 | 2 (gaps 1.44e-3, 5.6e-4) |
  | 2 (attn_moe) | multichip | 6.77e-3 | 6.31e-3 | 1.71e-3 | 4 (gaps 3.44e-3, 2.8e-4, 1.44e-3, 5.6e-4) |
  | 3 (conv_moe) | single chip | 1.46e-3 | 1.42e-3 | 1.04e-3 | 1 (gap 3.3e-4) |
  | 3 (conv_moe) | multichip | 1.60e-3 | 1.58e-3 | 1.05e-3 | 1 (gap 3.3e-4) |

  Both decoders' score errors exceed the stage-3 near-tie window (1e-3): the single-chip decoder also flips a
  1.44e-3 tie.  The multichip attention layer is ~12 % noisier (median) -- not the bf16 rounding of the row-parallel
  partials: producing them in fp32 and summing in fp32 left it unchanged (max 6.88e-3, median 1.63e-3,
  `logs/router_score_noise_L2_fp32mixer.log`; option removed); the remaining difference is HYPOTHESIZED to be inside the 8-head local
  attention (different SDPA / head-split core mapping; no component comparison yet).  -> multichip tests use a 1e-2 near-tie window (~2x the
  measured max score error); acceptance unchanged: one swap at that position must explain the decode update at
  PCC >= 0.99, logged per case.  Full-model top-1/top-5 accuracy (datatype sweep stage) is the real arbiter.
- The comparison test's single-chip baseline now uses the multichip expert-FFN op (unified, HiFi2): the stage-3
  hybrid op cannot allocate its arena next to the CCL L1_SMALL region.

## M-11 Final runs (final code, `logs/final_code_md5.txt`)
- Suite: 93/94 in `logs/m2_full_suite.log`; the failure was the stage-3-style near-tie control, which asserts the
  SINGLE-CHIP side of the 7.3e-4 tie at position 428; the multichip decoder lands on HF's own choice (update vs plain
  HF 0.999837, vs the forced swap 0.989230).  Rewritten to accept plain HF or that one swap, logged; passes
  (`logs/m2_near_tie_control.log`).  Near-tie acceptances in the suite: slot 9 pos 33 (gap 3.44e-3, alt 0.999813),
  slot 13 pos 769 (2.28e-4, 0.999793; same case as stage 3), slot 2 pos 122 (2.47e-4, 0.999849), slot 6 pos 357
  (3.59e-4, 0.999840); rejected candidates are logged next to them.
- Memory (`test_multichip_memory_capacity` -> `sweeps/memory_capacity.json`): per chip conv_dense layer 0.041 GiB,
  conv_moe 0.311, attn_moe 0.371 (incl. RoPE tables), KV 32 x 131072 per attention layer 1.063 GiB; 40-layer stack
  23.12 GiB of 31.74 GiB (72.8 %) -> context 131072 at batch 32 kept (`../context_contract.json`).
- Watcher: 33 tests, 0 error lines, all 4 devices (`logs/watcher/`).
- Tracy / wall: `tracy_final/`, `tracy_final_runs.txt`, `logs/final_perf_wall50.log`; tables in README.

## M-12 Review round 1 (Fable): more-work-needed -> fixes
1. (P2) stacked contract untested -> `test_multichip_stacked_layers` (layers 1 -> 2 -> 3, one decode trace on the
   sharded residual): each layer output == `resid_mem` (asserted); sharded entry vs DRAM entry: active rows bitwise
   equal (idle rows are undefined by contract: max |diff| 75 there), sharded vs sharded bitwise stable; vs chained HF
   prefill 0.99947 / decode 0.99995, stack update 0.99649 / 0.99978 (`logs/stacked_layers.log`).
2. (P3) "no persistent L1" justification stale (hybrid op unused) -> code comment + README fixed; persistent L1
   scratch / constants recorded as an optimized-multichip candidate.
3. (P3) fused CCL+matmul and residual-sharded rows were rejected by arithmetic only -> README now lists them as open,
   first measurement of the next stage.
4. (P3) prefill dispatch glue (~1.1 ms of 3.2 ms, replicated) -> README limitation 2 with candidates.
5. (P3) router-noise cause marked as a hypothesis (work_log M-10, README limitation 3).
6. Op issues -> drafts in `op_issues.md` with repro paths; not filed (outward-facing: the user decides).
7. Round 2 (Fable): clean-pass, non-blocking nits fixed: the stacked test logs TT_METAL_TRACE_ALLOC_TRACKING and was
   rerun with it (`logs/stacked_layers.log`); the only decoder change after the m2 runs is the comment at
   `tt/multichip_decoder.py` MeshReduce.decode ("scratch allocated per call ... optimized-multichip-stage candidate",
   replacing "a persistent L1 buffer would collide with the prefill hybrid routed-expert op"); README labels which
   report each prefill-glue aggregation comes from; context_contract.json newline.

## Stage 4 commit
`70eb2c13` (2026-10-17). Pre-commit hooks reformatted the code; post-commit device smoke on the reformatted code (real
weights, TT_METAL_TRACE_ALLOC_TRACKING=1, `-k "stacked or perf or (single_user and 65-2048) or (vs_single_chip and None)"`):
19 passed (`logs/post_commit_smoke.log`).
