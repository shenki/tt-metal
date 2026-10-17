# Stage 5 work log: optimized multichip decoder (1x4 Blackhole p150b ring, TP = 4)

Start: stage-4 commit `70eb2c13` (`MultichipDecoder`, `MC_POLICY`, `DEFAULT_MESH_PLAN`).  Plan:
`session_state/stage5_plan.md` (written by a Fable planning subagent).  User context: usually a few concurrent users
(weight b1..b8 decode), keep the 32-slot capability, context 131072, BFP4/LoFi weight precision deferred to the datatype
sweep, prefix caching later (keep ShortConv state / page tables easy to snapshot).  Paths relative to this directory;
stage-4 evidence = `../multichip_decoder/`.

## O-0 Baseline ("before") rows
Stage-4 final Tracy (`../multichip_decoder/tracy_final/`) for b32 / b1 / b1 bucket 1 / b4 bucket 4 / prefill 2048, plus
the missing bucket-8 rows (`tracy_before/*/a8_b8/`, `tests/test_multichip_decoder.py` perf parametrization + (8, 8)):

| kind | b8 bucket 8, 64-wide experts (stage-4 default) | b8 with `moe_compact_max_users=8` (stage-4 compact configs) |
|---|---|---|
| attn_moe | 585.4 us | 556.2 us |
| conv_moe | 461.7 us | 497.4 us |

40-layer decode step (2 conv_dense + 28 conv_moe + 10 attn_moe): b1 bucket 8.54 ms, b4 13.44, b8 19.07, b32 26.3; prefill
2048 126 ms (device).

## O-1 Persistent buffers / L1 placement (topology audit rows C1, R5, R6)
Tracy per layer (device, max over chips), candidates via `LFM2_POLICY_OVERRIDE` / `LFM2_PLAN_OVERRIDE` (`tracy_cand/`):
- `moe_route_v3_l1=True` (routing constants in L1; forbidden in stage 3 only by the hybrid prefill op, unused here):
  conv_moe b1 bucket 207.8 -> 206.4, attn_moe 242.2 -> 240.9.  Adopted.
- `MeshPlan.persistent_scratch=True`: one persistent [32, 8192] L1 all-reduce scratch per mesh (shared by all layers and
  bucket traces; dead between all-reduces) instead of an in-trace allocation per all-reduce: device time unchanged (allocation
  is host-side), 80 fewer in-trace buffers per 40-layer step.  Adopted.
- `Policy.attn_concat_persistent=True` (new field, default off in stage 3): persistent zeroed L1 concat output per (bucket
  rows, layout) instead of a per-call DRAM->L1 zero copy; rows >= b of a bucket's buffer are never written, so they stay 0
  even with several bucket traces: attn b1 bucket 240.9 -> 239.2, b32 unchanged (720.5 vs 718.3: noise).  Adopted.
- Regression with the three: perf/determinism (b32, b1, bucket 1, 4, 8), multi-bucket traces, stacked layers, vs single chip,
  single user 65 / 2049: 28/28 (`logs/o1_scratch.log`).

## O-2 Bucket 8 (user: usually a few users)
`repro/../../multichip_decoder/repro/local_moe_sweep.py` at bucket-8 shapes (`sweeps/b8_*.jsonl`): compact 32 indexed slots
gate_up 204.6 / down 100.9 us; scan at 26 active experts gate_up 197.9 / down 124.3 (current cfg) -> 110.3 with (8,6,12,2).
Whole-layer b8 (`tracy_before/*/a8_b8/`): compact is not uniformly better -- conv_moe 461.7 (scan) vs 497.4 (compact),
attn_moe 585.4 vs 556.2: the scan path's gate_up cost follows the number of distinct experts in the batch (137 vs 185 us on
the two test inputs) while compact always reads 32 slots; compact saves ~25 us of 64-wide glue (weight multiply / reduce over
64 experts).  Kept the 64-wide path for b8; new `Policy.moe_down_small_bucket_cfg=(16, (8,6,12,2))` (MCMoE._mm_cfg picks it
for static buckets <= 16).  b8 after O-1 + O-2: conv_moe 461.7 -> 440.5, attn_moe 585.4 -> 569.1 (`tracy_cand/*/a8_b8/phase1`).

## O-3 Residual layout / collective placement / fused matmul-CCL family (`repro/fractured_family_probe.py`)
Attention sublayer boundary at decode, each chain under its OWN residual contract (no restore to replicated), eager-checked
against torch on all 4 chips (PCC qkv / out >= 0.99996 every chain), traced wall / 50 (`sweeps/fractured_family_probe.log`,
`sweeps/ff_<grid>.log`, `sweeps/fractured_family_grid.log`):

| chain | residual | ops | us |
|---|---|---|---|
| A ShardedNorm -> DS qkv; DS wo -> all_reduce_async -> add (current) | replicated, 8-core L1 | 5 | **40.5** |
| F2 dit_fused_distributed_rmsnorm -> all_gather_async -> DS qkv; DS wo -> reduce_scatter_minimal_async -> add | fractured [32,512] | 6 | 67.5 |
| F dit_fused_distributed_rmsnorm -> all_gather_minimal_matmul_async(qkv); minimal_matmul_strided_reduce_scatter_async(wo) -> add, mm grid 10x8 | fractured | 4 | 97.8 |
| F, mm grid 8x4 | fractured | 4 | 93.5 |
| F, mm grid 4x4 | fractured | 4 | 87.6 |
| F, mm grid 8x2 | fractured | 4 | HANG (killed by timeout; `tt-smi -r`, smoke 1x4 OK) |

The fractured family needs three fabric round trips per sublayer (norm stats, activation gather, partial scatter) where the
replicated contract needs one all-reduce, and the fused minimal-matmul CCL ops (built for large M) cannot use the DRAM-sharded
decode matmuls (`all_gather_matmul_async` 1D/2D mcast only, `matmul_reduce_scatter_async` 2D only; stage-3 interleaved vs DS
qkv 48 vs 21 us).  Rejected for decode with the table above (best fractured 67.5 vs 40.5 us per attention boundary; the MoE /
conv boundaries have the same CCL structure).  Same conclusion as the qwen3.8 QB2 TP4 default (replicated + direct AR).

## O-4 Batch-1 op-count items (round 1)
- Compact experts (`MCMoE._compact_forward`, new Policy fields `moe_compact_weight_before_down`, `moe_compact_out_sharded`):
  routing weight applied to the [n,32,384] activation before `down` (linear; rounding order only) and the expert sum written
  by `fast_reduce_nc` straight into the all-reduce layout (no I2S before the AR): conv_moe b1 bucket 206.4 -> 204.6 us
  (`tracy_cand/conv_moe/a1_b1/{wbd,wbd_outsh}`).  Adopted; vs single chip unchanged to 1e-6, bucketed b1/b2/b4, perf /
  determinism, multi-bucket, stacked: 24/24 (`logs/o4_compact.log`).
- SDPA decode k_chunk (per-chip shape 8 q / 2 kv heads, ctx ~2K): 512 (current) b1 19.92 / b32 63.23 us; 256: 19.88 / 60.36;
  128: 24.27 / 79.52 (`tracy_cand/attn_moe/*/kc*`).  256 saves 2.9 us only at b32 and stage 3 measured 512 faster from ~32K
  context (R2-1) -> kept 512.
- Per-step attention inputs (RoPE cos/sin: slice, 2x embedding, 2x tilize, 2x transpose, 2x I2S = 13.4 us; b-row
  page_table / cur_pos slices 1.9 us) run inside each single-layer measurement but are computed once per decode step for all
  10 attention layers in a full model (the stacked test computes cos/sin once per step): accounted separately in the README.
- q/k per-head RMSNorm round trips (2 x S2I + LN + I2S = 8.8 us) + 2 RoPE (3.0 us): a fused per-head norm + RoPE
  (`dit_fused_distributed_rmsnorm(per_head_norm, rope_*)`) needs the q and k column blocks sliced out of the qkv row and
  concatenated back before `nlp_create_qkv_heads_decode` (v must not be normalized): >= 5 ops vs 8 -> <= ~4 us / attention
  layer (40 us / step); not pursued (see O-6 for the remaining-budget ranking).
- Explained, not a change: vs-single-chip prefill update numbers (e.g. conv_moe prompt 300 0.99933 -> 0.99984) moved when the
  stage-4 baseline switched to the same expert-FFN op (unified HiFi2, stage-4 M-10); they are identical in stage-4 m2 and here.

## O-5 Prefill all-reduce, CCL dtype, packed vs separate, remaining audit rows
- Prefill all-reduce [2048, 2048] per chip (`repro/prefill_ar_probe.py` -> `sweeps/prefill_ar_probe.log`, traced wall / 20):
  composite `ttnn.all_reduce` bf16 (current) **159.5 us**; composite on a bfp8 payload incl. the 2 typecasts 200.8 (max err
  0.19 vs 0.05); RS-minimal + AG-async 203.5; `all_reduce_async` on L1 width-sharded 64 cores: CB/L1 clash
  (program.cpp:2491), 32 cores: OOM (4x scratch = 1 MB/core) -> kept the composite.
- Decode CCL payload dtype (`repro/decode_ccl_dtype_probe.py` -> `sweeps/decode_ccl_dtype_probe.log`): DS wo partial written
  as bfp8 + all_reduce_async to bf16: 23.1 vs 25.2 us (-2.1 us per all-reduce) but PCC 0.999936 vs 0.999963 (max err
  0.102 vs 0.081).  A precision reduction: per the user's decision (precision -> datatype-sweep stage) recorded there as a
  measured candidate (-4 us / layer), not adopted.
- Packed vs separate projections: dense w1 | w3 (per chip [2048, 2 x 3072 padded]) as one DS matmul (32, 2, 3) 33.7 vs
  2 x 21.3 us separately (`repro/local_ds_sweep.py w13_packed` -> `sweeps/ds_w13_packed.log`), and one input reshard instead of two: `MeshPlan.dense_w13_packed`
  -> conv_dense decode 150.9 -> 144.9 us (`tracy_cand/conv_dense/a1_b1/w13_{packed,separate}`).  Adopted.
  Conv in_proj split into B / C / x DS matmuls (to keep the ShortConv chain on one shard grid without slices): 9.63 us
  each = 28.9 vs 16.9 for packed + S2I + 3 slices (`repro/local_ds_sweep.py conv_in_third` -> `sweeps/ds_conv_in_third.log`) -> rejected (the shard-grid chain would
  also need B / C / x on the same cores: a sub-shard slice that ttnn does not provide).
- A4 MoE experts read the width-sharded ffn-norm output (no S2I): the indexed sparse_matmul accepts it, identical output
  (`repro/sparse_mm_sharded_in0_probe.py`); `MeshPlan.moe_sharded_input`: conv_moe 204.2 -> 203.3 (b1), 440.7 -> 438.5 (b8),
  672.5 -> 669.8 (b32) (`tracy_cand/conv_moe/*/{sharded_in,o4_ref}`).  Adopted.
- Prefill MoE dispatch ReshapeView copies (8 per chunk, 105 us): from ops.csv shapes they are the [P,1] <-> [1,P] pair-index
  conversions (dest / src / e_col), the pair-cumsum block totals and the RM scatter-row reshapes; the exact row forms need
  fp32 values up to ~9k that bf16 cannot carry (no exact matmul transpose trick) -> not changed (3 % of prefill).
- Precision data for the datatype sweep (perf only, NOT adopted; `sweeps/precision_r33_perf_only.txt`), stage-3 R3-3
  combination (expert gate_up BFP4 decode copy + expert LoFi + attention LoFi): conv_moe b1 203.3 -> 197.8, b8 438.5 -> 429.0,
  b32 669.8 -> 630.9; attn_moe b1 ~238 -> 230.1, b8 -> 540.0, b32 -> 679.3 us.

## O-6 Final runs (final code `logs/final_code_md5.txt`)
- Full suite 98/98 (`logs/o5_full_suite.log`); watcher 37 tests, 0 error lines (`logs/watcher/`); final Tracy all kinds x
  {b32, b1 none, bucket 1, 4, 8} + prefill (`tracy_final/`, `tracy_final_runs.txt`); wall 50 replays 15/15
  (`logs/final_perf_wall50.log`); memory: stack 24.82 GB = 23.11 GiB / chip at 32 x 131072 (72.8 %), context contract
  block "optimized_multichip_decoder" (131072, no reduction).
- Remaining-budget ranking (why the pass stops here): every audit row has an action or a measured rejection; what remains for
  b1..b8 is structural (TopK, replicated routing glue, 2 all-reduces, expert matmul bandwidth at N = 24 tiles) and the
  precision candidates owned by the datatype-sweep stage.
- The memory test now writes `doc/optimized_multichip_decoder/sweeps/memory_capacity.json` (env `LFM2_MEMORY_JSON`); the
  o5 suite run had overwritten the committed stage-4 file, restored from git (values differed by < 0.1 %).
  `local_ds_sweep.py` (with the new roles) lives in `repro/` here; the stage-4 copy is unchanged.

## O-7 Review round 1 (Fable): more-work-needed -> fixes
1. (P1) ShortConv chain on the shard grid -- my O-5 "sub-shard slice not provided" was an unverified assertion and wrong:
   `ttnn.slice` takes the 8-core [32, 192] width-sharded in_proj output and writes B / C / x straight into the out_proj
   input layout (8 cores x [32, 64]).  Probe `repro/conv_shard_chain_probe.py` (`sweeps/conv_shard_chain_probe.log`, one
   chip, DS in_proj + chain): current 56.58 us vs sharded 51.08 us, identical PCC (y 0.99993, state 0.999999 / 0.99999),
   output on the out_proj layout.  Implemented: `MeshPlan.conv_sharded_chain` (`MCShortConv.decode_forward`): state, taps
   (pre-expanded to [32, 512]) and the slot mask on the chain shards (harness `DecodeInputs(slot_mask_mem=...)`,
   `MultichipDecoder.slot_mask_mem`), prefill converts the state to interleaved around the stage-2 prefill and back.
   Tracy (`tracy_cand/*/conv_chain`): conv_moe b1 203.4 -> 194.8, b32 670.3 -> 663.2; conv_dense 144.6 -> 136.5 us;
   prefill +~10 us (state conversions).  Correctness: 26/26 (`logs/r1_conv_chain.log`: vs single chip incl. per-chip state
   slices, single user, slot reuse, stacked, perf/determinism).  Adopted.
2. (P2) Prefill dispatch, executed this time:
   - pair-index vectors straight to the [R, P/R] scatter form (one tile reshape instead of [P,1]->[1,P]->RM reshape):
     ReshapeView 104 -> 248 us per chunk (the [8192,1]->[32,256] tile reshape is a slow repack): conv_moe prefill
     3175 -> 3318 us (`tracy_cand/conv_moe/a1_b1/dispatch_rows`, `logs/r1_dispatch_rows.log`) -> reverted.
   - concat-free gather table: the dummy expert (entry 0) gets a ZERO down projection (output exactly 0 whatever its
     input rows), so the token gather reads x itself (no 32-zero-row header + concat of the whole chunk); unused / dummy
     buffer rows point at x rows r mod 32 (spread kept).  conv_moe prefill 3190.5 -> 3138.6, attn_moe 3504.8 -> 3451.8 us
     (`tracy_cand/*/a1_b1/concat_free`), PCC vs single chip identical (`logs/r1_concat_free.log`).  Adopted (+3.3 MB BFP8
     zero down weight per MoE layer per chip).
   - bounded local rows (plan 5.1.3): per-chip share of a chunk's top-4 pairs on real hidden states (HF router, layers
     3 / 15 / 31 / 38, 4 prompts of 400 tokens; `repro/ep_local_pair_histogram.py` -> `sweeps/ep_local_pair_histogram.log`):
     max 0.395, mean of per-prompt max 0.30-0.36 (ideal 0.25).  A static cap (e.g. 1/2) holds on this data but not in
     general (a repeated token routes all pairs to its 4 experts, possibly one chip): an exact capped path needs a per-chunk
     host decision (breaks the traced-prefill plan of the full model) -> not adopted, data recorded.
   - typecasts / untilize / tilize: 11 typecasts total 30 us, each at a dtype boundary the ops require (uint32 indices,
     fp32 scans, bf16 gather); untilize of the BFP8 FFN output (153 us) is required by the row-major combine gather.
3. (P2) Fused per-head q/k norm (+RoPE) measured (`repro/qk_norm_rope_fused_probe.py` -> `sweeps/qk_norm_rope_fused_probe.log`):
   `dit_fused_distributed_rmsnorm(per_head_norm=True, num_heads_per_device=8)` on the q block: 16.2 us (17.4 with weight)
   vs 11.1 us for the current S2I + rms_norm + I2S, output head-major [1, 8, 32, 64] (SDPA decode wants [1, 32, heads, 64]:
   one more transpose); its RoPE is the trans-mat (interleaved) convention: HF rotate-half would need head-dim-permuted
   q / k weights, norm weights and KV cache across prefill and decode.  Rejected (slower before even adding the transpose).
4. (P3) Persistent all-reduce scratch: two buffers used alternately (see `MeshReduce.__init__`): a chip can be at most one
   all-reduce ahead (it cannot finish all-reduce k+1 before every chip sent its k+1 data, sent only after finishing k), so
   a leading chip never writes into a scratch another chip still reads; 2 x 512 KB L1 per chip.
5. (P3) `moe_down_small_bucket_cfg` limited to bucket 8 (the measured point) with the config that was fastest there:
   (8,6,12,1) 110.29 us ((8,6,12,2) 111.45; current (8,8,12,1) 124.3; `sweeps/b8_down_scan_26.jsonl`).
6. (P3) `MCDenseMLP.forward` packed decode path asserts residual is None.
7. (P3) docs: README gains the prefill fractured-family arithmetic (composite AR 159.5 vs RS-min 120 + AG 99 / RS+AG 203.5:
   the fractured prefill residual cannot win), the AR-core-count row by arithmetic, NOT measured: the all-reduce runs
   on the 8-core residual grid with no reshard (8.9-11 us device in the decoder); a 16 / 32-core all-reduce would need a
   reshard of the partial in and of the result out (1.2-1.9 us each, from the decode profiles) and would have to save more
   than that inside a latency-bound CCL (links 2 / 3 / 4 identical, M-9); and the lazy persistent-buffer warm-up contract.
- Final runs after O-7 (final code `logs/final_code_md5.txt`): suite 98/98 (`logs/r2_full_suite.log`), watcher 39 tests 0 errors,
  final Tracy (`tracy_final/`), wall 15/15; step b1 8.09 / b4 12.87 / b8 18.00 / b32 26.00 ms, prefill 125.8 ms.

## O-8 Review round 2 (Fable): more-work-needed -> fixes
1. (P2) Persistent L1 of the sharded ShortConv chain: L1 is allocated at one address across all banks, so the round-1
   placement (state x2 + expanded taps x3 on the 8-core shards, 4 KB each) would have pinned ~600 KB of every core's L1 for
   30 conv layers.  Placement probe (`repro/conv_shard_chain_probe.py` -> `sweeps/conv_shard_chain_probe_placement.log`):
   state + taps on L1 shards 51.20 us, taps in DRAM 51.28, state + taps in DRAM 52.47, stage-4 chain 56.67 -> state and
   taps in DRAM (prefill state conversions removed: the state is the stage-4 DRAM tensor again).  The per-layer
   `moe_route_v3_l1` bias ([32, 64] fp32, 4 KB per bank per MoE layer) moved back to DRAM; the shared index constants stay
   in L1.  `test_multichip_memory_capacity` now records L1 per bank (two conv_moe layers to separate shared vs per-layer):
   40-layer stack 160 000 B per core at construction (+ <= 24 576 B of lazily allocated per-bucket attention concat buffers
   once every bucket has run: <= 184 576 B), 0 per additional layer (`sweeps/memory_capacity.json`, `logs/r4_full_suite.log`).
   context_contract.json updated (two scratches, L1 budget, log pointer).
2. (P3) Concat-free gather exactness invariant documented at `d_spread` (a pair's rank inside its expert region <= its token
   index because a token's top-K experts are distinct).
3. (P3) README prefill before/after now pairs the stage-4 a1 b1 runs (3498.4 / 3185.5 / 1406.9); step totals recomputed.
Final runs (final code `logs/final_code_md5.txt`): suite 98/98 (`logs/r4_full_suite.log`), watcher 39 tests 0 errors,
`tracy_final/` + `tracy_final_runs.txt`, wall 15/15.  Steps b1 8.29 / b4 13.04 / b8 18.16 / b32 26.15 ms, prefill 125.6 ms.

## O-9 Review round 3 (Fable): doc-only fixes
1. L1 figure stated as construction-time 160 000 B + <= 24 576 B lazy per-bucket concat buffers = <= 184 576 B per core
   (README, work_log O-8, context_contract).  2. Audit row R7 now 56.7 -> 52.5 us (O-8 placement).  3. Superseded
   intermediate `tracy_final_runs_conv_r3.txt` deleted (tracy_final_runs.txt is the final set).
