# Optimized decoder (stage 3) — work log

Branch `autoport/lfm2_5_24b_a2b`, base = stage 2 commit `fee73388`. One Blackhole p150b (1x1 mesh,
compute grid 11x10, 8 DRAM banks). Optimization target: batch-1 single-user decode latency and
prefill/TTFT; batch 32 kept correct and measured.

Measurement tools:
- `tests/bench_policy.py`: one `Policy` → real-weight PCC (32 users or one long prompt) + warmed
  wall-clock perf at batch 1/32 (screening only). Results in `sweeps/bench_p*.jsonl`.
- Tracy + tt-perf-report (with advice) via the same test in `LFM2_BENCH_MODE=perf_only`
  (`tracy_explore/<kind>/b<batch>/<tag>/`, final `tracy/`). All reported numbers are device time.
- Isolated op sweeps in `repro/` with traced timing (trace of N calls, outputs freed inside the
  capture, capture always closed).

## Harness mistakes found and fixed during the stage
- The first DRAM-sharded sweep kept 20 traced outputs alive in L1 and did not close the trace capture
  when a config raised; one L1-clash cascaded into "Cannot load new binaries during trace capture"
  for every later config. Fixed (`repro/dram_sharded_sweep.py` `timeit`), all roles re-run
  (`sweeps/ds2_*.log`, 0 cascades; 2 single-config L1 rejections remain as FAIL lines).
- Idle decode slots (cur_pos = -1) have undefined attention output rows by contract; the
  determinism check compared full outputs and flagged them. It now compares active rows (every
  downstream op is row-local).

## Changes, in order (decode at batch 1 / batch 32, prefill per 2048-token chunk)

1. **Idle-slot routing mask** (`mask_idle_routing`): routing rows of idle slots are multiplied by
   `slot_active`, so the union sparsity only covers active users' experts. conv_moe b1 decode
   6065 -> 1226 μs (wall), attn_moe 6364 -> 1871 μs; b32 unchanged; PCC identical (`bench_p1`).
2. **KV cache BFP8 + SDPA decode k_chunk 512** (`bench_p2`): PCC same/better; attn_moe b1 −90 μs.
   SDPA decode sweep (`repro/sdpa_sweep.py`, wall): k_chunk 0/64/128/256/512/1024, HiFi4/HiFi2/LoFi,
   max_k_splits 2/4, grids — best exact config 414 μs b1 / 527 μs b32 (bfp8, kc 512). `exp_approx_mode`
   151/298 μs but decode update PCC 0.9799 (< 0.99) -> rejected (precision; datatype-sweep candidate).
   Root cause of the cost: B=32 on 110 cores gives 3 cores/user and 3 KV heads per core, each over the
   full context, and GQA groups of 4 q-heads fill 32-row tiles (half tiles only when <= 16 q heads per
   user) — a single-chip limitation; KV-head sharding over 4 chips (stage 4) gives 8 q heads/chip
   (half tiles) and 1 head/core.
3. **Expert weight dtype** (`bench_p3`): BFP8/BFP4 and LoFi made *no* difference (6060 μs b32) —
   the sparse_matmul config was the bottleneck, not DRAM.
4. **sparse_matmul program configs** (`sweeps/sparse_decode_sweep_wallclock.log`, decode shapes,
   64 or 4 active experts): stage-2 helper `in0_block_w=1, subblock 1x1`: gate_up 4180 μs, down
   2800 μs (64 active). Best: gate_up grid 8x8 in0_bw 8 → bf16 2064 / bfp8 1264 / bfp4 987 μs;
   down grid 8x6 in0_bw 16 → 1110 / 704 / 553 μs. 4 active: ~230 μs floor regardless of dtype.
   In-decoder (`bench_p4`): conv_moe b32 6067 -> 2813 (bf16) -> 1928 (bfp8) -> 1724 μs (bfp4 gate_up);
   b1 1223 -> 819 (bfp8) -> 807 (bfp4). BFP4 gate_up: update PCC 0.982 (conv_moe) for −13 μs at b1 /
   −200 μs at b32 -> kept BFP8 (batch-1 target; precision), BFP4 gate_up recorded for the datatype sweep.
   LoFi vs HiFi2 for BFP8 experts: same speed, slightly lower PCC -> HiFi2.
5. **Decode MoE intermediates in L1** (`bench_p5`): conv_moe b1 821 -> 644, b32 1929 -> 1722 μs.
6. **Dense projection dtypes** (`bench_p6`): BFP8+HiFi2 everywhere: conv_dense decode 686 -> 564 μs,
   PCC unchanged. BFP8+LoFi: conv_dense output PCC 0.9944 (< 0.995), attn update 0.981, no speed gain
   -> rejected. BFP4 (attn/conv/dense w13): output PCC 0.976–0.993 -> rejected (OPT-007 trial done on
   real weights).
   (Fixed on the way: `dit_minimal_matmul_addcmul_fused` needs the weight tile size to match the
   residual's -> residual fold applies only to bf16 weights; `_project` checks it.)
7. **DRAM-sharded decode matmuls** (`sweeps/ds2_*.log`, traced, bfp8 HiFi2, 1/2/3 readers per bank,
   all core counts dividing K, all in0_block_w):
   | role | K x N | interleaved | best DRAM-sharded | config |
   |---|---|---|---|---|
   | wqkv | 2048x3072 | 48.0 | 21.3 | 8 cores, in0_bw 8, 2 readers |
   | wo / conv out | 2048x2048 | 38.3 | 14.7 | 8 cores, in0_bw 8, 2 readers |
   | conv in | 2048x6144 | 48.4 | 35.0 | 32 cores, in0_bw 2, 3 readers |
   | dense w1/w3 | 2048x11776 | 68.7 | 72.5 | slower -> kept interleaved (68% DRAM BW) |
   | dense w2 | 11776x2048 | 209.4 | 63.7 | 46 cores, in0_bw 8, 2 readers |
   Prefill with the DRAM-sharded weight needs an explicit 2D mcast config (default config rejects a
   width-sharded B; `repro/ds_prefill_probe.py`: 2D config 241 μs vs interleaved 265 μs) -> one weight
   copy. In-decoder (`bench_p7`, Tracy `v2_ds`): conv_dense 564 -> 386, conv_moe 628 -> 597,
   attn_moe 1211 -> 1160 μs (b1). Bug found on the way: an edit misplaced `ckc_sdpa_decode`, fixed.
8. **Sharded decode norms** (`bench_p8`): operator_norm width-sharded on the consumer DRAM-sharded
   matmul's grid (8 cores attn / 32 cores conv; output fed without reshard), ffn_norm on 8 cores then
   interleaved. conv_dense 389 -> 343, conv_moe 595 -> 570, attn_moe 1170 -> 1137 μs (b1); 16/32
   ffn-norm cores within noise.
9. **Prefill sparse_matmul configs** (`sweeps/ps_*.log`, per 512 tokens, ~55/64 experts active per
   tile): gate_up 57.6 -> 17.2 ms (8x8, in0_bw 8), down 38.6 -> 9.2 ms (8x6, in0_bw 8, sub 2).
   In-decoder: MoE prefill 2048: conv_moe 359 -> 103, attn_moe 338 -> 97 ms.
10. **Token-dispatch MoE prefill** (`repro/dispatch_proto.py`): per-pair expert one-hot, exact
    in-expert ranks and tile-aligned region offsets, expert-sorted buffer by row gather (`embedding`),
    `deepseek_prefill.unified_routed_expert_moe`, gather back + `post_combine_reduce`. Fabric-free
    (`dispatch`/`combine`/`offset_cumsum` need fabric or >1 device). PCC vs torch fp32 0.99934.
    Crossover vs tuned sparse (`sweeps/dispatch_crossover.log`, traced): T=64 sparse 3.3 / dispatch
    3.8 ms; T=96 4.9 / 4.2; T=2048 97.8 / 15.5 ms -> dispatch for chunks >= 96 rows. Decode (T=32):
    dispatch 0.93 ms (b1) / 3.2 ms (b32) vs sparse ~0.4 / ~1.3 ms -> decode stays sparse, so the expert
    weights are held twice (stacked for sparse_matmul, per-expert for the dispatch op): ~12 GB/chip
    BFP8 on 4 chips with EP — fits with the 32-user full-context BFP8 KV (~10.6 GB/chip) in 32 GB
    (context_contract.json).
11. **Exact rank scans as matmuls** (`sweeps/cumsum_alternatives.log`, [8192 pairs, 64]): ttnn.cumsum
    dim 0 9.997 ms (internal permutes + 2-tile-wide accumulation); two-level cumsum 1.545 ms; tril(32)
    block matmul + strict-lower block-prefix matmul 0.379 ms, exact (0/1 bf16 inputs, fp32 accumulation,
    HiFi4). Region offsets as a [1,E]x[E,E] matmul in tile units (values <= 65, exact). MoE prefill
    2048: 18.1 -> 8.45 ms (attn_moe), 7.5 ms (conv_moe).
12. **Prefill SDPA chunk** (Tracy): 256 -> 1760 μs, 128 -> 1360 μs, 64 -> 1379 μs, 512/1024 ->
    L1 overflow (5.7 MB CBs) -> 128.
13. **Dense MLP w1/w3 prefill 2D configs**: first attempts overflowed L1 (34 output tiles/core);
    `prefill_2d_config` now picks in0_block_w and the output block (divisors) under an L1 budget.
    w1/w3 1321/1350 -> 923/920 μs; conv_dense prefill 5187 -> 4528 μs.
14. **bf16 router operands** (`router_bf16`): checkpoint router weight is bf16 and x is bf16, so each
    product is exact in fp32; fp32 accumulation/output. Selection identical to fp32
    (`logs/router_selection_probe_stage3.log`: 3/512 and 1/512 vs torch fp32, same as fp32 path).
    Decode −5..−8 μs, prefill −125 μs (Tracy `tracy_explore/conv_moe/*/router_*`).
15. **K-split decode router** (`router_ksplit`): the 32x64 output has 2 tiles -> 2 cores run K=2048.
    First version reduced the fp32 partials with `fast_reduce_nc`: decode update PCC 0.935 at 16/32
    slices. `repro/ksplit_debug.py`: `fast_reduce_nc` on fp32 input loses precision (max |err| 0.020 /
    0.031 / 0.046 at 8/16/32 slices) while `ttnn.sum` is exact (PCC 1.0) -> `ttnn.sum`; then all split
    counts give identical PCC; 8 slices: conv_moe b1 564 -> 553, b32 1637 -> 1625 μs.
    (fast_reduce_nc elsewhere in the decoder reduces bf16 tensors, as in stages 1-2.)
16. **Dispatch FFN kernel** (Tracy): unified LoFi = HiFi2 time (3494 μs); `hybrid_routed_expert_moe`
    requires both halves at the same fidelity (TT_FATAL "the halves disagree on math fidelity") ->
    LoFi; threshold 320: 3224 μs, 128: 3101 μs; PCC unchanged incl. the long-prompt check that actually
    exercises hybrid (`bench_p15`, 3049-token prompt in 2048 chunks).
17. **Windowed ShortConv prefill tail** (`conv_prefill_windowed_tail`): the two tail-selection
    matmuls (`32 x 2080 x 2048`, 146 μs) select from a <= 64-row tile-aligned window instead of all
    32 + S rows. conv_dense prefill 4532 -> 4389 μs; PCC identical (`bench_p16`).

## Not adopted / limits (evidence above)
- SDPA decode: exact configs swept; `exp_approx_mode` rejected on precision; structural single-chip
  limit (see 2). Owner: stage 4 KV-head sharding.
- Decode MoE glue (~150 μs at b1: zero-fill of the [64, 32, N] sparse outputs, slices, swiglu mul,
  routing-weight mul, reduce) scales with all 64 experts; `sparse_matmul(indices=...)` compact output
  needs a static active count (decode union varies) — a per-active-user-bucket trace is a serving-stage
  design, not a layer change.
- Dispatch decode slower than sparse at T = 32 (see 10).
- `moe_split_gate_up` (stage 2), BFP4 projections, LoFi projections: see above.
- tt-perf-report advice: "No program_config specified" on the 7 μs K-split router batched matmuls
  (attn/conv MoE decode) — 0.6 % of the layer; DRAM-sharded advice on dense w1/w3 — measured slower
  (72.5 vs 68.7 μs).

## Watcher
- Default path, `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_NOINLINE=1`: all
  conv_dense tests and every test without hybrid dispatch pass, 0 error lines (`logs/watcher/`).
- `hybrid_routed_expert_moe` cannot be watcher-instrumented on this build: "Program size (75872-79056)
  too large for kernel config buffer (70656) on TENSIX" with full watcher; 71232-74992 with NOINLINE +
  waypoint/ring-buffer/stack-usage/pause disabled (asserts and NOC sanitize on): 2/8 passed
  (`logs/watcher_hybrid_program_size_limit/`). The buffer size is a firmware memory-map constant
  (`MEM_*_KERNEL_CONFIG_SIZE`). Replacement evidence: the same 8 MoE tests with only the expert FFN
  swapped to `unified_routed_expert_moe` (everything else identical) pass under the full watcher
  (`logs/watcher_unified_ffn/`, 0 error lines); hybrid correctness/determinism/trace-allocation
  safety from the non-watcher runs.
- Stage-1/2 regression after the round-1 stage-3 edits: functional + perf_determinism 39 passed, fused 39 passed
  (`logs/functional_regression_final.log`, `logs/fused_regression_after_hooks.log`).

# Round 2 (2026-10-16): stage-review round 1 = more-work-needed; findings and what was done

Review report: `/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/stage3_review_round1.md` (outside the repo).
Sweeps ran on chips 1-3 from a frozen copy of the round-1 code (`TT_VISIBLE_DEVICES=<n>`, Tracy
serialized by a lock); winners were ported into `tt/optimized_decoder.py` and re-measured on the live
code (chip 0). All timings below are device-accurate: Tracy device time or traced replay (no eager
`perf_counter` loops: the round-1 "~230 us sparse floor" was host-bound eager timing).
The round-1 sections above are history; where round 2 supersedes them it says so here.

## R2-1 Batch-1 decode: active-slot bucket (`active_slots`) for SDPA (review P1 #1)
- SDPA decode takes its batch B from the page-table rows and gives each row
  `110 / B` cores (`sdpa_decode_program_factory.cpp:123-127,199-215`): at B = 32, 8 KV heads, a single
  active user gets 2 cores (108 idle). Round-1 limitation #1 ("stage 4 removes it") was wrong.
- `OptimizedDecoder.decode_forward(..., active_slots=b)`: a static per-trace bucket; the caller keeps the
  active users in slots 0..b-1. SDPA gets `page_table[:b]`, `cur_pos[:b]` (Q stays the 32-user
  height-sharded tensor; the op reads rows < b), the output is zero-padded back to 32 rows (idle rows
  become exact zeros). KV updates still use all 32 slots (idle slots: cur_pos -1).
- Correctness vs the 32-row call (`repro/r2_sdpa_bucket_check.py`, `sweeps/r2_sdpa_bucket_check.log`):
  PCC >= 0.99997 for b = 1..16, idle rows 0.0. Interleaved Q is rejected by the op (needs sharded Q).
- Traced timing vs context (`repro/r2_sdpa_bucket_ctx.py`, `sweeps/r2_sdpa_bucket_ctx.log`, incl. the
  slice + pad):

  | ctx | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 (current) |
  |---|---|---|---|---|---|---|
  | 512 | 55 | 55 | 57 | 56 | 90 | 133 |
  | 2048 | 65 | 66 | 81-241* | 136 | 250 | 454 |
  | 32768 | 184 | 342 | 620 | 1718 | 3407 | 6758 |
  | 131000 | 591 | 1181 | 2301 | 6769 | 13486 | 26920 |

  (k_chunk 512; *noisy point, 90 us in `r2_sdpa_bucket_check.log`). Correction (review round 3): k_chunk 512
  is NOT within ~10 % of the best chunk at b=1: ctx 512 kc256 43.4 vs 55.1 us, 2K kc256 56.9 vs 65.3, 8K kc128
  89.7 vs 103.6; 512 only wins from ~32K. Kept one k_chunk (512) for all buckets in stage 3: the b=1 loss at the
  2K bench point is ~8 us per attention layer (~84 us of a ~12.5 ms single-chip token, <1 %), and stage 4 changes
  the SDPA shape (2 KV heads / 8 q heads per chip), so the per-bucket k_chunk choice (a Policy field per bucket,
  free with one trace per bucket) is re-measured there on the sharded shape. Tracy attn_moe b1: SdpaDecode 532 -> 48 us.
- 2 KV heads (stage-4 shape) for reference: b=32 74 us, b=1 27 us at 2K.
- Tests: `test_optimized_bucketed_decode` (b1 short/long, b2, b4 with an idle slot inside the bucket,
  b8, b16; all kinds), `test_optimized_bucketed_full_context` (131069-token prompt, bucket 1),
  perf/determinism with buckets 1 and 4. The serving stages need one trace per bucket and slot compaction.

## R2-2 Batch-1 decode: compact MoE via `sparse_matmul` indexed mode (review P1 #1, P2 #3)
- `moe_compact_max_users=4` (buckets 1/2/4; `active_slots` <= 4): the b*K selected experts run 4b-wide
  with `sparse_matmul(indices=...)`; expert ids and weights from exact 0/1 matmuls on device (no gather,
  no host reads), NaN-safe `where`. b = 8 loses (1240 vs 878 us) -> 64-wide path above 4.
- Device-timed op sweep at 4 active experts (`repro/r2_moe_op_sweep.py`, `sweeps/r2_moe_op_sweep_*.jsonl`,
  1044 configs): gate_up scan/nnz=None 135.6 us -> scan nnz=4 94.7 -> indexed 76.5 us (68 % DRAM BW);
  down 86.3 -> 53.3 -> 42.3 us. Configs: gate_up (8,6,8,1), down (8,6,48,2); in0_block_w 8..48 within
  2 %, only 2 is worse. BFP4/LoFi indexed: gate_up 56.8, down 32.6 us, but BFP4 experts fail the update
  bar on conv_moe (0.9756 / 0.9804) -> BFP8/HiFi2 kept; LoFi = HiFi2 speed.
- nnz=None vs explicit: +41 / +31 us at 4 active, ~0 at 48-64 active; indexed mode beats nnz=4 anyway.
- Packed vs split gate_up re-measured with tuned configs (stage-2 rejection was untuned): packed wins at
  4 (76.5 vs 91.8), 48 and 64 active -> `moe_split_gate_up` stays rejected.
- Tracy (`tracy/r2_moe/`): conv_moe b1 547.9 -> 363.1 us (MoE block 437.7 -> 252.4, glue 152 -> 25 us).
- Blocker found: `expert_gate_up=bfp4` with `expert_down=bfp8` hits `TT_FATAL hybrid_program_factory.cpp:881
  weight_ptr->dtype() == weights_dtype` in the hybrid dispatch op (mixed expert dtypes unusable in prefill).

## R2-3 Decode router (review P2 #6)
- Router chain in DRAM ~100 us/MoE layer (Tracy op list). Candidates on the live code
  (`tracy_r2/conv_moe/b1/*`, decode device us): default 549.8; `router_ds_cores=8` (router as a
  DRAM-sharded matmul, N 64 -> 256 zero-padded and sliced, reading the ffn_norm's 8-core width-sharded
  output directly: no reshape/permute K-split) **521.7**; `router_kth_slice` (k-th score as the last topk
  column instead of min-reduce + FillPad) 557.7 (slower, rejected); all intermediates in L1: no gain in
  the traced microbench (`repro/r2_router_bench.py`: 99.7 vs 102.9 us).
- `router_ds_cores=8` adopted: identical selection and PCC (`sweeps/r2_router.jsonl`), -28 us (b1) /
  -25 us (b32) per MoE layer; replaces `router_ksplit`.

## R2-4 L1 residual (review P2 #6)
- `decode_residual_l1` (decode residual adds to L1): conv_moe b1 551.0 vs 549.8 us -> no gain, rejected
  (the consumers read 128 KB; the adds are 2.4-3.4 us either way).

## R2-5 Conv state update with a full-width slot mask (found in the final tt-perf-report)
- The two ShortConv state `where(slot_active[32,1], ...)` per conv layer broadcast a [32,1] condition:
  8.5 us each vs 3.1 us full-width. `decode_forward(slot_mask=[1,1,B,H])` (built once per step by the
  caller; `DecodeInputs.slot_mask`) -> conv_moe b1 bucket 339.0 -> 326.2 us, conv_dense 323.6 -> 310.9.

## R2-6 BFP4 / LoFi per group on the final geometry (review P1 #2)
Each group alone on real weights, layers 0 (conv_dense), 1 (second dense layer), 2 (attn_moe), 3, 7, 15,
23, 31, 39 (`sweeps/r2_bfp4_bench.jsonl`, `r2_bfp4_cand.jsonl`, `r2_bfp4_combined.jsonl`,
`r2_bfp4_convmoe_layers.jsonl`; out = output PCC prefill/decode, upd = update PCC):
| group | result |
|---|---|
| conv_in BFP4 (HiFi2 / LoFi) | L0 out .99415/.98673 FAIL (LoFi .98155); L15 upd D .947, L31 .923, L39 out .898 FAIL; gain 9 us (33.4 -> 23.5) |
| conv_out BFP4 | L0 out .99272 (HiFi2) / .99062 (LoFi) FAIL; LoFi L3 upd .98819 FAIL; gain 2.4 us |
| conv BFP8 + LoFi | L0 out D .99112 FAIL |
| dense_w13 BFP4 | L0 out .98816 (HiFi2) / .98736 (LoFi) FAIL; L1 PASS (.99444) |
| dense_w2 BFP4 | L0 out .98995 / .98805 FAIL; L1 PASS |
| dense BFP8 + LoFi | PASS, no measurable gain (63.9 vs 65.9 us DS w1) -> not adopted |
| attn_qkv BFP4 | L2 upd D .96773 FAIL |
| attn_out BFP4 | L2 upd .97473 / .97359 FAIL |
| attn BFP8 + LoFi | L2 upd D .98069 FAIL |
No group passes on every layer it is used in -> none adopted globally. Option for the per-layer datatype
sweep (stage 8): layer 1 dense BFP4/LoFi (Tracy 332.7 -> 243.6 us, update PCC decode .99094).
- DRAM-sharded geometry re-swept under each dtype/fidelity with 1/2/3 readers and K/N padding
  (`repro/r2_bfp4_dram_sharded_sweep.py`, `sweeps/r2_bfp4_ds_*.log`): e.g. conv_in bfp4/hifi2 r3 (16,4)
  24.0 us vs bfp8 r3 (32,2) 33.7; w1 bfp8 r3 (32,1) 65.9 (N padded 12288) vs interleaved 68.8.
- Dense w1/w3 decode: explicit 1D configs (per_core_N, in0_block_w 1..32, subblocks) best 71.6 vs TTNN
  default 72.2 us -> no gain (`sweeps/r2_bfp4_w13_1d_*.log`). **Adopted**: DRAM-sharded w1/w3 for decode
  (`ds_dense_w13=(32,2,3)`, N 11776 -> 12288 inert zero padding: silu(0)*0 = 0 against zero w2 rows) +
  K-padded DRAM-sharded decode w2 (`ds_dense_w2_decode=(48,8,2)`). DS w2 with 3 readers via N padding:
  no gain, rejected.
- Prefill keeps the interleaved w1/w3 and the unpadded DS w2: a DRAM-sharded w1/w3 in prefill needs
  full-width output blocks (see R2-8) and costs 901/911 vs 651/655 us per 2048-row matmul
  (`tracy_r2/conv_dense/b1/{fullw_bucket1,w13_interleaved_v2}`). Extra weight copies: ~77 MB per dense
  layer, 2 layers (context_contract.json).
- conv_dense Tracy: decode 333.2 -> 322.7 us, prefill 4391 -> 3689 us (with R2-7).

## R2-7 MoE prefill single/2-core ops, prefill configs, prefill SDPA (review P2 #4)
Traced per-op (`repro/r2_prefill_ops_bench.py`, `sweeps/r2_prefill_ops_{glue,sdpa,mm}.log`, 2048 rows):
| op | before | after | rejected alternatives |
|---|---|---|---|
| rank-scan batched matmul b=256 | 270 us (2 cores) | 22 us (`moe_scan_multicore`: MatmulMultiCoreReuseProgramConfig 8x8, per_core 1x2, exact) | tri-broadcast 307; transposed single matmul 131 (tile reshape 113) |
| dispatch scatter | 410 us (1 row = 1 core) | 139 us (`moe_scatter_rows=32`: pairs split over 32 rows, each scattered into its own zero row, fp32 sum; exact, one write per destination) | R=8 168, R=64 157, R=128 273 |
| BFP8 FFN output -> BF16 RM | 403 us (typecast + untilize) | 182 us (`moe_out_direct_untilize`: one `to_layout`, bit-identical) | BF16 FFN output not exposed by the op (`hybrid_routed_expert_ffn.cpp:81`) |
| prefill SDPA (start 0 / 2048 / 30720) | q128/k128 HiFi4 1375 / 3914 / 39459 us | q64/k512 HiFi2 1160 / 2953 / 28110 us | q256 slower; k1024 1714; HiFi4 q64/k512 1233 |
| QKV 2048x2048x3072 | 230 us (sub 1x1) | 166 us (`prefill_mm_v2`) | 8x8 / 8x10 grids slower |
| conv_in 2048x2048x6144 | 553 us | 330 us (`prefill_mm_v2`) | |
| wo / conv_out 2048^3 | 119 | 120 (already good) | |
In-decoder ablation (`sweeps/r2_prefill_ablate.jsonl`, 2048 chunk, us): scan -269/-262, scatter -276/-278,
direct untilize -243/-227, prefill_mm_v2 -30/-167, SDPA q64/k512 HiFi2 -231 (attn).
Prefill at short and boundary lengths (single fresh chunk, traced; `sweeps/r2_prefill_len_sweep*.jsonl`,
BASELINE / round-1 DEFAULT / round-2, us):
| L | conv_moe | attn_moe |
|---|---|---|
| 1 | 1766 / 744 / 743 | 1999 / 684 / 681 |
| 31 | 6262 / 1748 / 1747 | 5937 / 1613 / 1609 |
| 65 | 12944 / 3844 / 3805 | 11924 / 3518 / 3533 |
| 95 | 16700 / 3685 / 3646 | 15998 / 3677 / 3639 |
| 96 | 18357 / 4004 / 3963 | 17469 / 3731 / 3692 |
| 97 | 18227 / 4094 / 4050 | 16417 / 3787 / 3736 |
| 127 | 22605 / 4197 / 4158 | 22060 / 3896 / 3843 |
| 128 | 21573 / 4142 / 4103 | 21361 / 3683 / 3635 |
| 129 | 22887 / 3294 / 3240 | 22803 / 2986 / 2920 |
| 300 | 53901 / 3638 / 3541 | 52374 / 3315 / 3216 |
| 2047 | 350886 / 7118 / 6235 | 334050 / 7602 / 6662 |
| 2048 | 356229 / 7119 / 6236 | 339605 / 7609 / 6661 |
(< 96 rows is the sparse prefill path, so the dispatch changes do not apply there; the k chunk is capped
at chunk_start + S so short prompts do not pay a 512 k chunk.)

## R2-8 Op bug: DRAM-sharded weight + partial output blocks is nondeterministic
- Porting R2-6 onto R2-7's `prefill_mm_v2` made `test_perf_determinism` fail ("prefill is not deterministic
  for a repeated fresh request", conv_dense). Isolated (`repro/r2_mm_dram_sharded_partial_block.py`,
  `sweeps/r2_mm_dram_sharded_partial_block.log`, 6 repeats each): 2D-mcast matmul [2048x2048]@[2048x12288]
  BFP8 with a DRAM-width-sharded weight is nondeterministic whenever `out_block_w < per_core_N`
  (block 7x7, 7x5, 1x7 of per_core 7x35; in0_block_w 4 or 8); deterministic with full-width blocks (1x35)
  and with the same partial-block configs on an interleaved weight (also without N overhang: 8x8 grid,
  per_core 8x48). Fix: `prefill_2d_config(dram_sharded_w=True)` -> full-width output blocks for every
  DRAM-sharded weight. Upstream issue candidate.

## R2-9 Full-context PCC drop: cause found (review P5)
attn_moe 131069-token prompt, prefill update PCC (`logs/r2_fullctx_*.log`):
| variant | update PCC |
|---|---|
| round-1 DEFAULT | 0.990573 |
| kv_cache bf16 | 0.990574 |
| experts bf16 + unified + hifi2 | 0.990627 |
| attention bf16 | 0.990661 |
| sdpa_prefill_chunk 256 | 0.995979 |
| q64 / k512 HiFi2 (adopted, R2-7) | **0.997574** |
| functional / fused | 0.99647 / 0.99649 |
Cause: k chunk 128 = more online-softmax rescale steps over 131k keys, not the BFP8 KV (round-1 README
was wrong). Microbench start 30720: attention PCC k128 0.99984, k256 0.99990, k512 0.99998.

## R2-10 Router near-ties in the tests (anomaly found while adding bucket tests)
- attn_moe prompt 428 seed 1435 failed the update bar (decode 0.98928) on the stage-1 functional decoder,
  the fused/BASELINE decoder and every precision variant alike (`test_router_near_tie_control`). CPU probe
  (`repro/r2_router_margin.py`, `repro/r2_router_flip.py`): at position 428 the HF fp32 biased scores of
  the 4th/5th experts (44 / 19) differ by 7.3e-4; HF with exactly that swap forced gives output 0.999467 /
  update 0.989381 vs TT 0.999465 / 0.98928. conv_moe slot 13 (prompt 768, seed 1781): gap 2.3e-4 at
  position 769, swap reproduces 0.998307 / 0.94245 vs TT 0.998306 / 0.94201.
- `reference/hf_layer.py::near_tie_alternatives` (gap < 1e-3): the update bar accepts the HF reference
  with that one swap, logged as a WARNING with the gap and both PCCs (never silent). The output bar is
  always against plain HF. Explicit control test with the forced swap: `test_router_near_tie_control`.

## R2-11 Deep layers on real hidden states (anomaly found by the BFP4 sweep)
- DEFAULT failed on conv_moe layers 7/23/31/39 with embedding-row inputs (update decode 0.8995 on L7);
  control: the stage-1 functional decoder fails the same way (L39 prefill output 0.958) -> the inputs are
  out of distribution for deep layers (the residual stream grows through the stack).
- `tests/gen_real_hidden.py`: 4 x 400 tokens of book text, HF layers 0..L-1 (fp32, CPU) -> inputs of every
  layer (`generated/lfm2_real_hidden.pt`). `test_optimized_deep_layers_real_hidden` (layers 1, 6, 7, 15,
  23, 31, 38, 39; optimized and functional): 16/16 pass, min update PCC 0.99724 (L6 decode)
  (`logs/r2_deep_layers_real_hidden.log`).

## R2-12 Smaller fixes
- Idle-slot routing mask `where(slot_active, routing, 0)` (NaN-safe; was multiply).
- `context_contract.json`: KV bytes per token per attention layer 1088 BFP8 / 2048 BF16 (was 2176 / 4096).
- Correction of round-1 text: the round-1 watcher run covered 5 of 16 conv_dense tests (not "all");
  SDPA decode at B=32 runs 4 KV heads per core (not 3); prompt 65 pads to 96 rows and uses dispatch,
  prompts <= 32 use the decode MoE path in prefill.
- `width_shard_layout` builds an 8-wide rectangle for multiples of 8 cores; > 80 cores would need > 10 rows
  (illegal on 11x10). No current caller uses > 80 cores (largest 48); noted.

## Superseded round-1 limitations
- Round-1 "Not adopted / limits": SDPA decode (now bucketed, R2-1), decode MoE glue (now compact, R2-2),
  "BFP4 projections (PCC)" (now per group, R2-6), "DRAM-sharded dense w1/w3 slower" (now padded with 3
  readers, R2-6).

# Round 3 (2026-10-16/17): stage-review round 2 = more-work-needed

Review report: `/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/stage3_review_round2.md` (outside the repo).
Again: sweeps on chips 1-3 from private code copies, winners ported to the live code and re-validated
on the merged code. Tracy runs hold one lock for the whole run (csv pickup included) and record a
worktree md5 in `ops_source.txt`; the helper refuses a csv without the DEVICE ID column.

## R3-1 MoE decode routing glue (review P1)  — `moe_route_v3`, `router_ds_fused`
Routing window (after the ffn_norm up to the first SparseMatmul), Tracy device us (`sweeps/r3_route_tracy/`):
| case | before | after |
|---|---|---|
| b1 bucket | 25 ops, 93.8 us + 21.0 gaps | 19 ops, 67.4 + 14.2 |
| b4 bucket | 32 ops, 117.7 + 27.5 | 21 ops, 76.9 + 15.2 |
| b32 | 19 ops, 99.4 + 12.9 | 18 ops, 86.2 + 11.7 |
Layer decode: conv_moe 325.4 -> 299.4 (b1) / 706.1 -> 664.0 (b4) / 1579.6 -> 1568.2 (b32); attn_moe
419.5 -> 392.0 / 822.8 -> 784.7 / 2034.0 -> 2015.5 us.
- What changed: full-width fp32 bias (no row broadcast); selection from topk *indices* via exact 0/1
  matmuls (idx -> one-hot -> mask; replaces FillPad + min-reduce + [T,1]-broadcast `ge`); full-width eps
  and a bf16-output div; 64-wide idle mask `slot_active @ ones[1,64]` + full-width `where`; compact ids
  untilized straight from topk rows 0..b-1 (relies on the bucket contract, enforced by
  `validate_decode_bucket`), compact weights from exact 0/1 matmuls; sigmoid fused into the DRAM-sharded
  router matmul with the 64 columns sliced from its sharded output. Index constants shared per device.
- Exactness: current vs new on real router inputs (HF ffn_norm outputs, 1600 book-text tokens each for
  layers 2/3/7/15/23/31/39): scores, 64-wide routing (all active / every third row idle), compact b1/b2/b4
  and b4 with an idle slot **bitwise identical in 350/350 batches each** (`sweeps/r3_route_check.log`,
  `repro/r3_route_check.py`).
- Rejected (evidence `sweeps/r3_route_prims*`, `r3_route_bench_v3*`): constants in L1 (hybrid prefill
  expert op claims every L1 byte: OOM, `moe_route_v3_l1` must stay off); batched pick matmul without the
  permute (slower 85.8 vs 80.4 / 118.8 vs 87.2); `ttnn.bcast` divide (not exact: 95.7 % bitwise);
  matmul broadcast of the denominator (tf32 truncation 1.4e-3); `ttnn.repeat` 12.8 us; `ttnn.gather` 35.9;
  tile-layout scatter 23.5; transposed-space normalization (3e-8 fp32 difference: not bitwise);
  `topk(sorted=False)` same 15.5 us; bf16 topk 7.0 us but changes the selection; uint16 index reshape
  unsupported (`reshape_device_operation.cpp:28`).
- Irreducible: fp32 TopK 15.8 us on one core (64-wide x 32 rows is below its multi-core threshold, no
  exact faster alternative), the per-row [T,1] normalization divide 7.6 us, ~2.5 us floor per 110-core
  elementwise op, ~1.4 us gaps around single-core ops.
- b32 64-wide path: `sparse_matmul` zero-fills its expanded output with a FILL(0) over [64,32,3072]
  (21.9 us at b32) unless the output is compact, and compact output needs a static `nnz`
  (`sparse_matmul_device_operation.cpp:497-540`); the b32 active count is data-dependent -> op limit.

## R3-2 Attention decode plumbing (review P1) — `attn_heads_sharded`, `attn_kv_fused_max_bucket=32`, `attn_pad_free`, `attn_bucket_heads`, `attn_residual_sharded`
Tracy attn_moe (`sweeps/r3_attn_tracy/attn_moe/b{1,4,32}/{base,c3,final}`): b1 bucket 420.0 -> 375.0,
b4 824.1 -> 786.4, b32 2034.3 -> 2010.4 us.
| op (b1 bucket) | before | after | how |
|---|---|---|---|
| qkv S2I + create_qkv_heads_decode | 1.24 + 15.05 | 3.39 | heads split straight from the DRAM-sharded qkv output (`DramShardedLinear.decode_sharded`), `slice_size=b` |
| q/k norm round trip | 14.8 | 8.86 | on b users |
| rotary x2 | 4.9 | 2.95 | b users (b-user cos/sin: `FusedRope.decode_cos_sin(batch=b)`, `decoder.bucketed_rope`) |
| PagedUpdateCache x2 | 17.85 | 8.56 | `paged_fused_update_cache` (wins at every bucket at device level, also b32 -8 us; stage 2 rejected it on wall clock at b32 only) |
| SDPA output pad | 6.85 | 0 | concat into a zeroed buffer copied from a DRAM zeros tensor (rows >= b stay 0) |
| wo S2I + DRAM residual add + ffn-norm I2S | 5.7 | ~1.6 | sharded residual add on the wo output layout |
- Rejected: per-head norm without the round trip (`rms_norm` rejects height-sharded,
  `layernorm_device_operation.cpp:226`; matmul formulation exact but +14 us, RSQRT not fusable,
  `repro/r3_attn_norm_micro.py`); sharded SDPA output ("Sharded output not supported for GQA",
  `sdpa_decode_device_operation.cpp:547`); DS wo on the 32-core concat grid (26.7-49.8 vs 14.2 us);
  persistent L1 zeros (hybrid prefill expert op OOM), `ttnn.zeros` L1 sharded (host-created, not
  trace-safe); cos/sin variants (<= 1 us/step). Not tried: `rotary_embedding_llama_fused_qk` (needs
  head-dim-permuted weights model-wide, ~1.4 us).
- Remaining at b1: ~12 us pure data movement, small compute ops (per-head LN 2x3.4, RoPE 2x1.5, residual
  1.6), fused KV update 8.3 us; cos/sin 14.5 us is per step (shared by the 10 attention layers).
- PCC: attn_moe matrix 21 passed + deep layers 6/38 + bucketed/multi-bucket rows identical to base to 6 digits
  (`sweeps/r3_attn_test_final_*.log`, `r3_attn_test_bucketed_*.log`).

## R3-3 BFP4 / LoFi on real layer inputs (review P2) — decision: deferred to the datatype sweep
Real hidden-state inputs (tests/gen_real_hidden.py; L0 = its real embedding input), layers 0, 1, 2, 3, 6, 7,
15, 23, 31, 38, 39 (`sweeps/r3_bfp4_real_matrix.jsonl` 111 rows, `r3_bfp4_real_combos.jsonl`,
`tracy_r3_bfp4/summary.txt`):
| group | real inputs |
|---|---|
| expert gate_up BFP4, decode copy (HiFi2 or LoFi) | PASS all 9 MoE layers (worst update .9913 L23) |
| expert down BFP4 (decode-only or both) | FAIL every conv_moe layer L3-L31 (worst L15 update D .96701) |
| attn_qkv BFP4 / attn_out BFP4 / both | FAIL (L6 .98700 / L2 .98886 / all) |
| attention BFP8 + LoFi | PASS |
| conv_in BFP4 | PASS alone; with expert BFP4 FAIL L15 (.96763), ~2.6 us gain |
| conv_out BFP4 | FAIL L23 (.98762) |
| conv BFP8 + LoFi | PASS on the 4-user real-hidden matrix (worst .99799); FAILS layer 0 in the 32-user stage bench (output decode .99112, see below) |
| dense w13 / w2 BFP4 | FAIL L0 (.99135 / .99201) |
| combined expert gate_up BFP4 + LoFi, attention LoFi, conv LoFi | PASS all 11 layers (worst update .99191 L31) |
Tracy of the combination: conv_moe b1 326.0 -> 303.8, b4 705.6 -> 616.1, b32 1582.4 -> 1337.9 us; attn_moe
b1 420.0 -> 396.8, b32 2027.9 -> 1815.5 us.
On the stage tests' inputs (`sweeps/r3_precision_bisect.jsonl`, 32-user bench, one change at a time):
conv LoFi fails layer 0 (output decode .99112 — layer 0's real input IS the embedding: a genuine fail);
expert gate_up BFP4 fails conv_moe L3 (update .98268 / .98664) and attention LoFi fails attn_moe L2 (decode
update .98069), both on embedding-row inputs that are out of distribution for those layers (R2-11).
**Decision (user, 2026-10-17): keep BFP8 / HiFi2 defaults in stage 3; the passing real-input candidates
(decode expert gate_up BFP4 + LoFi, attention LoFi) go to the datatype-sweep stage, which selects precision
on full-model top-1/top-5 accuracy.** The Policy fields `expert_gate_up_decode` / `expert_down_decode`
(decode copy only; the hybrid prefill dispatch op needs one dtype for gate_up and down) exist for it.

## R3-4 Several decode traces at once (review P2)
`test_optimized_multi_bucket_traces` (all kinds, `TT_METAL_TRACE_ALLOC_TRACKING=1`, program cache
included): one decoder, traces for buckets 1 / 4 / 32 captured after every bucket and every prefill shape
was compiled, sharing the decode inputs, KV cache and ShortConv state; schedule with prefills between
replays, users joining and leaving, a hole (slot 0 freed while slots 1-2 decode) and slot reuse
(replays per bucket {1: 4, 4: 8, 32: 2}). The tracker found that each later trace's output is alive at
an earlier trace's replay; each bucket's output tensor is acknowledged exactly
(`acknowledge_corruptible`) with the reviewed invariant: it is read right after its own replay, before
any other replay or prefill, and was allocated after every persistent buffer, so only the other traces'
intermediates can overwrite it, while it is dead. Serving contract (`tt/optimized_decoder.py`):
`decode_bucket(active slots)` = smallest bucket covering the highest active slot (new requests take the
lowest free slot, so no KV-page or ShortConv-state row ever moves; holes allowed);
`validate_decode_bucket(cur_pos, b)` raises instead of silently zeroing a slot >= b.

## R3-5 Fixes from the round-2 "other concerns"
- `DramShardedLinear.out_mem` now describes the layout the op computes (ceil(N/per_core_N) cores row-major):
  the 416 "Mismatch between computed MemoryConfig" warnings are gone.
- v1 `prefill_2d_config` also forces full-width output blocks for DRAM-sharded weights.
- `_mm_cfg` `("down_prefill", True)`; test docstring (prompt 65 -> dispatch); `near_tie_alternatives`
  restores a caller's `hf_patch`; the deep-layer test generates its input on demand and logs its sha256.
- Near-tie accounting: in the round-2 final logs 3 real-weight acceptances (L15 slot 13 pos 63 gap 2.6e-4,
  L7 functional slot 9 pos 131 gap 5.0e-4, b16 conv_moe slot 13 pos 769 gap 2.3e-4); the other warnings
  are synthetic-weight runs (no update bar) or rejected alternatives.

## R3-6 Review round 3 fixes (2026-10-17)
- `repro/r3_roofline.py` took expert gate_up as BFP4; it now reads every dtype from `DEFAULT_POLICY` (all BFP8).
  Regenerated `sweeps/r3_roofline.txt`: device/roofline attn_moe b1 bucket 3.27x, conv_moe 2.60x, b32 1.58x / 1.39x.
- SLOW tt-perf-report matmul rows dispositioned in README (operation-topology audit). New measurement
  `repro/r3_prefill_mm_l1.py` -> `sweeps/r3_prefill_mm_l1.log` (chip 0, traced replay, live configs):
  | matmul (2048 rows) | in0 DRAM | in0 L1 |
  |---|---|---|
  | qkv x3072 | 164.4 | 162.3 |
  | wo x2048 | 119.1 | 116.6 |
  | conv_in x6144 | 326.0 | 325.6 |
  | conv_out x2048 | 119.1 | 116.5 |
  | w1 x11776 (live, sub 1x1) | 666.6 | 664.6 |
  w1 geometries (in0 DRAM): grid 11x8 blk 4x17 sub 2x1 bw 8: 646.0; 11x8 4x34 sub 2x2: 661.1; 8x8: 870.8;
  N padded 12288 live config: 627.1 (11x8: 707.2, 8x8: 872.8, 8x10 sub 1x4: 817.2). All PCC 0.99997.
  Not adopted: input-0-in-L1 <= 2 %; w1/w3 gains (3-6 %, ~0.1 ms of a ~247 ms 2048 prefill over the two dense
  layers, ~0.16 ms) need the prefill w2 K-padded too -> full-model stage (drop the prefill dense copies, prefill on the padded
  decode weights).
- SDPA decode k_chunk: the R2-1 "within ~10 % everywhere" claim corrected in place (b=1 trade-off recorded);
  per-bucket k_chunk re-measured in stage 4 on the KV-head-sharded shape.
- R3-3 table: conv BFP8 + LoFi row corrected (passes the 4-user real-hidden matrix, fails layer 0 in the 32-user
  stage bench).
- Tests log `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` (next to `TT_METAL_TRACE_ALLOC_TRACKING`) and
  `LFM2_POLICY_OVERRIDE`. No model code changed in this round.
- Superseded exploratory Tracy op tables (`tracy_explore`, `tracy_r2`, `tracy_r2_bfp4` `ops.csv.xz`) removed;
  their perf reports remain.
- Infrastructure: after the 2026-10-17 reboot a Qwen vLLM container held all chips; stopped with the user's
  approval. The first 1x4 open then timed out on ethernet core 31-25 ("Try resetting the board"); one
  `tt-smi -r` recovered it (4x p150b listed, 1x1 and 1x4 mesh smoke OK). Infrastructure only.
- Code check after R3-6: `md5sum -c logs/r3_final_code_md5.txt` -> every tt/ and reference/ file OK; only
  tests/test_functional_decoder.py and tests/test_optimized_decoder.py differ (logging lines only). Logging smoke on the edited
  tests (real weights, chip 0): multi_bucket_traces attn_moe + single_user 1@2048 attn_moe 2/2 passed (`logs/r3r_logging_smoke.log`).
- Host e2e column provenance (`sweeps/r3_final_wall.jsonl`, round 3): per configuration
  `TT_VISIBLE_DEVICES=<chip> LFM2_BENCH_MODE=perf_only LFM2_BENCH_BATCH=<b> LFM2_BENCH_BUCKET=<bucket> LFM2_BENCH_DECODE_ITERS=50
  LFM2_TAG=final_wall LFM2_BENCH_OUT=sweeps/r3_final_wall.jsonl pytest -q tests/bench_policy.py -k <kind>` (50 non-blocking
  replays, one synchronize; the jsonl rows carry no iteration field).
- Round-4 review (Fable): clean-pass; its four P3 doc nits fixed (replay-count provenance above, conv selection-matmul
  SLOW rows, limitation-6 prefill wall per kind, L1/w1 arithmetic).

## Stage 3 commit
`06705dad` (2026-10-17). Pre-commit hooks reformatted the code (black/isort; md5s in `logs/r3_final_code_md5.txt` are the
pre-format provenance). Post-commit device smoke on the reformatted code (chip 0, real weights,
`TT_METAL_TRACE_ALLOC_TRACKING=1`, `-k "multi_bucket_traces or (single_user and blackhole-1-2048) or perf"`): 18 passed
(`logs/r3_post_commit_smoke.log`).
