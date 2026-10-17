# LFM2.5-24B-A2B optimized multichip decoder (stage 5): 1x4 Blackhole p150b ring, TP = 4

Optimized in place: `tt/multichip_decoder.py` (`MC_POLICY`, `DEFAULT_MESH_PLAN`, `MCMoE`, `MCDenseMLP`, `MeshReduce`) plus
default-off Policy fields in `tt/optimized_decoder.py`; tests `tests/test_multichip_decoder.py`.  Start: stage-4 commit
`70eb2c13`.  Plan: `session_state/stage5_plan.md` (Fable planning subagent).  History and every rejected option with its
evidence: `work_log.md` (O-0 ... ).  User context applied: usually a few concurrent users (b1..b8 weighted), 32 slots kept,
context 131072, BFP4/LoFi weight precision decided in the datatype-sweep stage, prefix caching in the vLLM stages.

## Results (Tracy device time per decoder layer, max over the 4 chips; `tracy_final/`, `tracy_final_runs.txt`)
| kind | mode | stage 4 (before) | stage 5 (after) | change |
|---|---|---|---|---|
| attn_moe | decode b1 bucket 1 | 242.2 | 235.9 | -2.6 % |
| attn_moe | decode b4 bucket 4 | 384.4 | 373.4 | -2.9 % |
| attn_moe | decode b8 bucket 8 | 585.4 | 567.0 | -3.1 % |
| attn_moe | decode b32 | 718.3 | 715.3 | -0.4 % |
| attn_moe | decode b1 no bucket | 436.0 | 433.4 | -0.6 % |
| conv_moe | decode b1 bucket 1 | 208.2 | 201.6 | -3.2 % |
| conv_moe | decode b4 bucket 4 | 332.1 | 322.2 | -3.0 % |
| conv_moe | decode b8 bucket 8 | 461.7 | 435.9 | -5.6 % |
| conv_moe | decode b32 | 673.8 | 668.3 | -0.8 % |
| conv_moe | decode b1 no bucket | 368.1 | 363.2 | -1.3 % |
| conv_dense | decode (any) | 150.9 | 141.5 | -6.2 % |
| attn_moe | prefill 2048 (a1 b1 runs) | 3498.4 | 3460.9 | -1.1 % |
| conv_moe | prefill 2048 (a1 b1 runs) | 3185.5 | 3148.4 | -1.2 % |
| conv_dense | prefill 2048 (a1 b1 runs) | 1406.9 | 1436.4 | +2.1 % (chip-1 spread: 1389-1436 across chips; a32 run 1419.7) |

40-layer decode step (2 conv_dense + 28 conv_moe + 10 attn_moe): b1 8.54 -> **8.29 ms** (-3.0 %), b4 13.44 -> **13.04** (-3.0 %),
b8 19.08 -> **18.16** (-4.8 %), b32 26.35 -> **26.15** (-0.8 %); prefill 2048 127.0 -> **125.6 ms** device (-1.1 %).
vs the single-chip stage-3 decoder: b1 12.5 -> 8.29 ms (1.51x), b32 64.4 -> 26.15 ms (2.46x), prefill 247 -> 126 ms (1.96x).
Wall clock (50 traced replays, `logs/final_perf_wall50.log`, us per single-layer replay incl. ~20-30 us launch): b1 bucket attn
261 / conv_moe 236 / dense 173; b4 390 / 361; b8 574 / 465; b32 750 / 698.  Eager prefill wall is host-dispatch bound
(2.8-9.4 ms per layer vs 1.4-3.5 ms device): traced prefill is a full-model item.
"Before" rows: stage-4 `../multichip_decoder/tracy_final/` + the bucket-8 rows measured at the start (`tracy_before/`).
Persistent L1 (L1 is allocated at one address across all banks): the 40-layer stack pins **160 000 B per core at
construction** (all-reduce scratch pair 131 072 B + shared routing constants 26 624 B + 2 304 B), 0 B per additional layer
(`test_multichip_memory_capacity` -> `sweeps/memory_capacity.json`, O-8), plus the attention concat buffers allocated lazily
on the first decode of each bucket: <= 6 buckets x [32, 512] bf16 on the 8-core wo grid = 24 576 B per core -> **<= 184 576 B
per core** once every bucket has run (arithmetic; the memory test does not run a decode).

Per-step attention inputs: of attn_moe's 236.7 us, RoPE cos/sin (10 ops, 13.4 us) and the b-row page-table / position slices
(1.9 us) are per-step inputs that a full model computes once per step for all 10 attention layers (the stacked test does);
the full-model per-layer attention cost is ~221 us at b1 (O-4).

## Adopted (default path; work_log O-1, O-2, O-4, O-5)
| change | where | effect (device us) |
|---|---|---|
| shared routing constants in L1 (`moe_route_v3_l1`; the per-layer bias kept in DRAM, O-8) | MC_POLICY | ~-1 / MoE layer |
| persistent concat output per bucket (`attn_concat_persistent`, new field) | MC_POLICY + OptimizedAttention | -1.7 / attention layer (b1) |
| persistent shared all-reduce scratch, two buffers used alternately (`MeshPlan.persistent_scratch`) | MeshReduce | 0 device; 80 fewer in-trace buffers per step; inter-chip skew safe (O-7) |
| small-bucket expert down config (`moe_down_small_bucket_cfg=(8, (8,6,12,1))`, new field) | MCMoE._mm_cfg | -14 at b8 (down 124 -> 110) |
| ShortConv chain on the out_proj input shards (`MeshPlan.conv_sharded_chain`; B/C/x sliced from the sharded in_proj output, chain intermediates and the slot mask on the 8-core shards, state and taps in DRAM: O-8) | MCShortConv | chain 56.7 -> 52.5 us (conv_moe / dense -4..-9 per layer) |
| concat-free prefill gather table (dummy expert with a zero down projection) | MCMoE._dispatch_forward | prefill -52 / MoE layer |
| compact experts: routing weight before `down`, expert sum written in the all-reduce layout (new fields) | MCMoE._compact_forward | -1.8 / MoE layer (b1..b4) |
| MoE experts read the width-sharded ffn-norm output (`MeshPlan.moe_sharded_input`) | MultichipDecoder.decode_forward | -0.9 (b1) / -2.2 (b8) / -2.7 (b32) |
| packed dense w1 \\| w3 decode matmul (`MeshPlan.dense_w13_packed=(32,2,3)`) | MCDenseMLP | -6 / dense layer |

## Operation-topology audit (decode b1 unless noted; full table with code pointers: `session_state/stage5_plan.md` sec. 2)
| row | item | action | evidence |
|---|---|---|---|
| A1 | q/k/v on the norm output | already one packed DS matmul | stage 3/4 |
| A2 | conv B/C/x on the norm output | packed DS kept; B/C/x sliced straight into the shard grid (adopted, O-7); split into 3 DS matmuls 28.9 vs 16.9 us -> rejected | O-5, O-7 |
| A3 | dense w1, w3 on the same input (2 matmuls + 2 input reshards) | packed: adopted | O-5 |
| A4 | router + experts read the norm output in two layouts | experts read the sharded copy: adopted | O-5 |
| A5 | expert gate_up packed [up \\| gate] | kept (stage-3 R2-2) | - |
| C1 | 2 all-reduces per layer (9 us each) | replicated contract kept; fractured / fused families measured and rejected; persistent scratch | O-1, O-3 |
| C2 | MoE partial reshard before the all-reduce | compact path writes the all-reduce layout directly: adopted | O-4 |
| C3 | prefill composite all-reduce 2 x ~130 us | bfp8 payload / RS+AG / L1 all_reduce_async measured: all slower or infeasible | O-5 |
| R1 | residual entry reshard | single-layer test artifact; the stacked model converts once (stacked test) | stage 4 |
| R2 | q/k per-head norm round trips + RoPE (8 ops, 11.8 us) | `dit_fused_distributed_rmsnorm(per_head_norm)` measured: 16.2 vs 11.1 us, head-major output, trans-mat RoPE convention -> rejected | O-7 |
| R3/R4 | per-step page-table slices, cos/sin | per-step inputs, computed once per step in a full model (accounting above) | O-4 |
| R5 | concat zero buffer copy | persistent: adopted | O-1 |
| R6 | routing constants in DRAM | L1: adopted | O-1 |
| R7 | ShortConv elementwise chain on 110 cores (16 tiles) + S2I / I2S | chain on the 8-core out_proj input shards (state / taps in DRAM): adopted (56.7 -> 52.5 us per chain) | O-7, O-8 |
| R8 | compact MoE weight multiply on [n,32,2048] | moved to [n,32,384]: adopted | O-4 |
| F1/L1 | fused matmul-CCL, fractured residual + distributed norm | measured under the fractured contract: 67.5-97.8 vs 40.5 us per attention boundary (one grid hangs) | O-3 |
| P1 | prefill dispatch reshape copies (105 us / MoE layer) | direct [R, P/R] form measured slower (248 vs 104 us) -> reverted | O-7 |
| P2 | prefill MoE dispatch glue replicated on 4 chips (~1.1 ms) | concat-free gather table adopted (-52 us); bounded local rows not exact in general (max local share 0.395 on real text, worst case 1.0) | O-7 |
| - | TopK 15.8 us (fp32, one core), router 10.5, routing 0/1 matmuls 10.3 | irreducible (stage-3 R3-1), replicated by design | - |

## Families compared (goal: coherent families, not single ops)
- **Residual layout + collective placement + fused matmul-CCL** (O-3, `repro/fractured_family_probe.py`): replicated
  [32,2048] 8-core L1 + one `all_reduce_async` (40.5 us per attention boundary) vs fractured [32,512]/chip with
  `dit_fused_distributed_rmsnorm` + `all_gather_minimal_matmul_async` + `minimal_matmul_strided_reduce_scatter_async`
  (87.6-97.8 us; grid 8x2 hangs: op_issues #4) or with unfused AG / RS around the DS matmuls (67.5 us).  Each chain measured
  under its own contract, eager-correct on all chips.  The fractured family needs 3 fabric round trips per sublayer and the
  fused ops cannot use DRAM-sharded decode matmuls.  Kept replicated (same choice as the qwen3.8 QB2 TP4 default).
- **Activation / CCL dtype** (O-5): decode bfp8 all-reduce payload -2.1 us per all-reduce, PCC 0.999936 vs 0.999963;
  prefill bfp8 payload slower (typecasts).  Precision -> datatype-sweep stage (user decision): candidate recorded.
- **Persistent buffers** (O-1): all-reduce scratch, concat buffer, routing constants: adopted.
- **Packed vs separate projections** (O-5): packed dense w1|w3 adopted; split conv in_proj rejected.
- **DRAM-sharded decode matmuls**: per-role configs on per-chip shapes (stage-4 M-4); re-swept here only where shapes changed
  (packed w13, split conv in_proj) (`repro/local_ds_sweep.py`).
- **Expert path at bucket 8** (O-2): compact (32 indexed slots) vs 64-wide scan: equal weight bytes at ~26 distinct experts;
  the winner flips with the number of distinct experts (conv 461.7 vs 497.4, attn 585.4 vs 556.2); scan kept + tuned down.
- **Precision / fidelity** (weights): not re-swept (user decision); perf-only data for the datatype sweep:
  `sweeps/precision_r33_perf_only.txt` (conv_moe b32 669.8 -> 630.9, b1 203.3 -> 197.8; attn b32 -> 679.3, b1 -> 230.1 us).

## Correctness (final code: `logs/final_code_md5.txt`)
- Full suite `tests/test_multichip_decoder.py`, TT_METAL_TRACE_ALLOC_TRACKING=1: **98/98** (`logs/r4_full_suite.log`): vs HF
  (single user 1..2049, batch 32, slot reuse, increasing lengths through one trace, full 131072 context +/- bucket, bucketed
  b1..b16, multi-bucket traces 1/4/32, deep layers on real hidden states, near-tie control, memory capacity, stacked layers
  1->2->3 in one trace), vs the single-chip TTNN baseline (all kinds, bucket none / 1), perf / determinism / no-host-fallback
  at b32, b1, bucket 1, 4 and 8.  Router near-ties handled as in stage 4 (window 1e-2, one-swap acceptance, logged); the
  attention path changed only in the concat buffer (identical values), so the stage-4 router-noise measurement stands.
- Watcher (`TT_METAL_WATCHER=10`): 39 tests (prefill paths, perf b32/b1/bucket 1/4/8, multi-bucket, stacked, vs single chip,
  slot reuse), 0 error lines (`logs/watcher/`).
- Determinism and trace safety: every perf case asserts bitwise-equal replays (b1/b4/b8/b32); the persistent buffers are
  shared only where dead between uses (all-reduce scratch) or never written outside rows < b (concat buffers per bucket);
  multi-bucket traces pass with them.

## Inter-layer residual contract (for the full-model stage)
- **Decode**: `[1, 1, 32, 2048]` bf16, L1 WIDTH_SHARDED on CoreRange (0,0)-(7,0), shard `[32, 256]`, ROW_MAJOR orientation
  (`MultichipDecoder.resid_mem` = `width_shard_layout(mesh, 8, 256)`), bitwise identical on the 4 chips, idle rows undefined.
  Produced by every layer kind's final residual add (`ttnn.add(h, all_reduce(partial))`), consumed by the next layer's operator
  norm (`ShardedNorm`, same layout: no conversion) and residual add.  The model converts once at entry (embedding output ->
  resid_mem) and once at exit (final norm / LM head input).  No collective, gather or reshard between layers; the two
  all-reduces per layer are inside the layer (after the row-parallel mixer and FFN).
- **Prefill**: `[1, 1, S, 2048]` bf16 DRAM interleaved, replicated.
- **Per-step decode inputs owned by the caller** (shared by all layers): `rope.decode_cos_sin(rope_pos[, batch=b])` (attention),
  `page_table` / `cur_pos` (b rows for bucket b), `slot_active [1,1,32,1]`, `slot_mask [1,1,32,512]` (ShortConv local channels; give it in `decoder.slot_mask_mem` to avoid a per-layer reshard),
  `active_slots` = the static bucket (one trace per bucket; `decode_bucket` / `validate_decode_bucket`).
- Persistent L1 budget the model inherits: <= 184 576 B per core for the decoder stack (160 000 at construction + <= 24 576 of
  lazily allocated per-bucket concat buffers; no per-layer L1); new model-level
  persistent L1 (LM head / sampling buffers) adds to it, and every program's static CBs must fit below the lowest of them.
- Lazy persistent buffers (trace-safety contract): the attention concat buffers (one per (bucket rows, layout)),
  `_ZERO_ROWS`, the routing constants and `_concat_out_buf` sources are allocated on first use, per device; the
  model must run every bucket (and every prefill shape) once before capturing traces (the tests warm all buckets first).
  The all-reduce scratch pair and every weight / KV / state tensor are allocated at construction.
- Prefix caching (vLLM stages): ShortConv state stays two per-slot DRAM tensors `[1,1,32,512]` per chip (row copies snapshot / restore a slot), KV pages use global block ids with replicated page tables; persistent buffers are keyed by shape / bucket only.

## Limitations
1. Batch 1..8 decode remains op-count bound (~50-64 ops per layer at a 1-3 us floor plus ~1 us op-to-op gaps); stage-5 gains
   are 2-5 %.  The remaining larger items are structural: TopK 15.8 us (fp32, single core), the replicated routing glue, the
   two all-reduces (9 us each), and expert sparse matmuls at ~250 GB/s per chip (N = 24 tiles per expert slice).
2. Prefill MoE: the replicated token-dispatch glue (~1.05 ms of 3.15 ms per conv_moe chunk) remains (concat removed; the
   gathers / FFN-output untilize / combine are sized by the static worst case, see O-7 for the bounded-rows data); the hybrid
   routed-expert op cannot coexist with any L1 allocation (unified op used; op_issues #1); eager prefill is host-bound.
   Fractured prefill residual: composite all-reduce 159.5 us vs RS 120 + AG 99 (or RS+AG 203.5, `sweeps/prefill_ar_probe.log`,
   stage-4 M-1): it cannot win at 2048 rows either.
3. Precision (BFP4 / LoFi weights, bfp8 CCL payload) left to the datatype-sweep stage by user decision, with measured candidates.
4. Op issues (`op_issues.md`): hybrid MoE arena sizing; sparse_matmul in0_block_w 64 inside traces; all_reduce_async DRAM;
   fused AGMM / MMRS hang at an 8x2 matmul grid (new).

## Commands
```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v --timeout=4000 models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_multichip_decoder.py
LFM2_MC_DECODE_ITERS=50 pytest -q models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_multichip_decoder.py -k perf
doc/optimized_multichip_decoder/repro/tracy_mc.sh <outdir> <kind> <active> <bucket|None> <tag> [policy-override-json]
LFM2_POLICY_OVERRIDE='{...}' LFM2_PLAN_OVERRIDE='{...}'   # A/B any adopted / rejected switch (logged by the tests)
python doc/optimized_multichip_decoder/repro/{fractured_family_probe,prefill_ar_probe,decode_ccl_dtype_probe,sparse_mm_sharded_in0_probe}.py
```
