# Fused decoder (stage 2) — work log

Branch `autoport/lfm2_5_24b_a2b`, base = stage 1 commit `799a0875`. Hardware: one Blackhole p150b
(1x1 mesh; SDPA compute grid 11x10 = 110 cores).

## 2026-10-15 — op-sequence analysis (from stage 1 Tracy, attn_moe decode, one replay)
| ops | device μs | note |
|---|---|---|
| 2x embedding + 2x reshape (cos/sin) | 11 | |
| LayerNorm (operator_norm) | 33 | 1 core (interleaved) — op config, stage 3 |
| Matmul 32x2048x3072 (wqkv) | 47 | |
| 3x slice + 3x reshape (q,k,v) | 34 | -> `nlp_create_qkv_heads_decode` |
| 2x LayerNorm + 2x mul (q/k norm) | 15 | -> `rms_norm(weight=)` |
| 2x (matmul 64x64 + mul + mac) (rope) | 20 | -> `rotary_embedding_hf` |
| 2x FillPad + 2x InterleavedToSharded | 14 | removed by create_heads_decode (already sharded) |
| 2x PagedUpdateCache | 18 | |
| SdpaDecode | 643 | program config — stage 3 |
| reshape (attn out) | 18 | -> `nlp_concat_heads_decode` |
| Matmul wo + add | 41 | |
| LayerNorm (ffn_norm) | 33 | 1 core — stage 3 |
| router: typecast, matmul fp32, sigmoid, add, topk, gather, sum, add, div, mul, untilize/scatter/tilize/typecast | ~150 | sigmoid fold + x1.0 drop; grouped_topk (rejected: precision) |
| FILL (sparse_matmul output zero-fill) | 73 | inside sparse_matmul |
| SparseMatmul gate/up (64 experts) | 2631 | |
| Transpose (permute E,G) | 71 | G=1 in decode -> reshape |
| swiglu composite (slice, slice, silu, mul) | 172 | -> slices + mul(input_tensor_b_activations=[SILU]) |
| FILL + SparseMatmul down (all 64 experts, nnz=E) | 3395 | -> union-of-selected-experts sparsity |
| permute(routing) + mul + fast_reduce_nc + add | 82 | |

Prefill (attn_moe, 2048): per 512-token group SparseMatmul gate/up ~43 ms, transpose ~1 ms,
swiglu ~2.7 ms, SparseMatmul down (all experts) ~54 ms, combine ~1.2 ms.

## Fusion candidates and measurements (bench_fusions.py: PCC run + wall-clock perf, real weights)
Round 1 leave-one-out vs all (decode μs added when removed; `logs/ablation_round1.jsonl`):
moe_down_union_sparsity +870/+1129 (conv_moe/attn_moe), moe_decode_no_permute +70/+68,
moe_grouped_topk +67/+63, attn heads+rope_hf +40..+53 (attn), moe_swiglu_mul_act +43/+40,
concat_heads_decode +13, qk_norm_weight +5, conv_state_inplace +3, dense_merged_w13 -9 (slower -> rejected).
Round 2 (`logs/ablation_round2.jsonl`, on top of the kept set): moe_prefill_down_sparse prefill
-69/-76 ms (kept); residual_fold decode +9..+21 μs (worse) / prefill -0.6..-3.4 ms (noise level);
dense_minimal_swiglu decode +121 μs / prefill -1.1 ms (rejected: decode worse, prefill-only use
needs a second weight copy); kv_fused_update +2 μs (rejected; matches llama31_8b_qb2 note that the
fused update only wins at batch <= 4).
Round 3 (`logs/ablation_round3.jsonl`): router_fold -2..-6 μs, same PCC (kept); conv_split_in_proj
+46 μs (rejected).
Precision (`logs/precision_conv_moe.jsonl`, conv_moe update PCC out-x prefill): none 0.99387;
+moe_grouped_topk 0.99088; +residual_fold_prefill on top 0.98896 (< 0.99 bar).
Router probe (`repro/router_selection_probe.py`, 512 real-embedding tokens): functional fp32
route differs from torch fp32 in 3/512 (L3) and 1/512 (L20) expert sets, max weight err 0.0031;
moe_grouped_topk 5/512 and 4/512, max weight err 0.0064 -> rejected for accuracy (gain ~65 μs per
MoE layer, ~1% of decode).

## Bug found: SDPA decode with height-sharded Q on a mismatched grid (fixed in our code)
Symptom: with qkv_heads_decode + rope_hf the traced decode replay was not bit-deterministic
(perf test assert), only when another decoder had run first in the process; output PCC still
passed (residual-dominated), `only:qkv_heads_decode,rope_hf` gave decode PCC 0.968.
Bisect: eager runs with host syncs between ops were deterministic; cache-hit vs fresh compile
with shifted L1 allocations differed (`repro/sdpa_decode_sharded_q_grid_mismatch.py`); per-stage
capture showed the first divergence at `paged_scaled_dot_product_attention_decode` with identical
inputs. Minimal repro vs torch: Q DRAM max|err| 0.0060; Q height-sharded on an 8x4 grid with the
full 11x10 SDPA grid: fresh 2.29-3.52, cache hit 1.19, fresh != cached.
Cause: with sharded Q the op places batch b's reducer at (b % grid.x, b // grid.x) of the SDPA
compute grid (`sdpa_decode_program_factory.cpp:309-322`); Q shards on an 8-wide grid vs an
11-wide SDPA grid -> reducers read other cores' L1 (stale data; varies with allocation). The op
does not validate this. Fix: shard decode heads row-major over the full grid width
(`decode_head_mem`), verified: max|err| 0.0060, fresh == cache hit (also 8x8 SDPA grid works).
`nlp_concat_heads_decode` rejects the non-rectangular row-major core set -> its input uses its
own rectangular 8x4 grid.
Test hardening: layer-output PCC hid this (residual-dominated); added the update PCC
(out - x vs ref - x, `delta_pccs`, bar 0.99 from the functional baseline min 0.991) to every
case.

## Deferred to stage 3 (not graph fusion)
- Token-dispatch MoE prefill (`deepseek_prefill.masked_bincount/dispatch/unified_routed_expert_moe/
  combine/post_combine_reduce`): dense-over-experts compute is ~16x the routed work. Blockers found:
  `offset_cumsum` needs >1 device (all_gather), dispatch/combine need fabric initialised even on
  one chip (untested with dispatch_group_size=1), masked_bincount needs T % 64 == 0.
- SDPA decode program config (643 μs), sharded norms (2x 33 μs on one core), matmul configs,
  dtypes (stage 8).

## Synthetic-weight control for the update-PCC bar
`LFM2_REAL_WEIGHTS=0` fused run: 3/15 failed only the new update-PCC bar (batch32 conv_moe slot 3
decode 0.9199; slot_reuse conv_moe/attn_moe 0.9891/0.9861). The functional decoder fails the same
3 tests identically (control). `repro/synthetic_bf16_control.py`: the HF layer itself in bf16 vs
fp32 on the same synthetic inputs gives slot 3 decode update PCC 0.9196 (TT 0.9199) -> a bf16
near-tie expert-selection flip inherent to the synthetic weights, not a TT issue. The update-PCC
bar is therefore asserted for real weights only (output PCC bar still applies to synthetic).

## Stage review round 1: more-work-needed -> round 4 candidates
Findings: slice-after-matmul untried at the dominant MoE gate/up slices; router still untilize/
scatter/tilize; conv prefill tilize/untilize unassessed; residual-fold rejection rationale wrong;
wall-clock noise vs small margins.
New candidates (`logs/ablation_round4.jsonl` wall clock; device time from Tracy A/B in
`tracy_ab/<kind>/<spec>/`, `run_perf_determinism` window, 5 decode replays, all windows complete):
- `moe_split_gate_up` (two half-width sparse_matmuls, no [..,2I] intermediate, no slices):
  with SiLU as the gate matmul's program-config `fused_activation` the MoE output is WRONG
  (conv_moe/attn_moe output PCC 0.98407/0.98960, update PCC down to 0.936); with SiLU applied in the
  multiply instead it is correct (same PCC as default) but decode 8919/8698 μs vs 6064/6157 μs and
  prefill 536/505 ms vs 359/338 ms (wall clock) -> rejected. Cause of slowness: N=I=1536 gives
  `_sparse_matmul_config` 48 cores with 1 output tile each (vs 2 for the packed 2I), and in0 is read
  twice. The fused_activation result is noted as a possible sparse_matmul issue (not pursued).
- `router_tile_mask` (k-th topk value threshold mask, fp32 TILE, no gather/scatter/untilize/tilize):
  selection identical to the functional router (`logs/router_selection_probe.log`: 3/512 L3, 1/512
  L20, max weight err 0.00313/0.00304); Tracy decode conv_moe 6110.6 -> 6055.5 μs, attn_moe
  6193.2 -> 6141.1 μs -> kept.
- `conv_tile_shift` (unaligned row slices in TILE layout instead of untilize + slice + tilize):
  same PCC; Tracy conv_dense prefill 5665.1 -> 5610.7 μs (−54 μs; wall clock had suggested −250) -> kept.
- `residual_fold_prefill` re-measured on device time: conv_dense prefill 5610.7 -> 5139.2 μs
  (−471), conv_moe 355777 -> 355547 μs (−231), attn_moe +50 μs (noise); precision neutral on its own
  (`logs/precision_conv_moe.jsonl` only:residual_fold_prefill update PCC 0.99387 == none; the earlier
  0.98896 was the combination with moe_grouped_topk, now rejected) -> kept (prefill only; decode M=32
  keeps linear + add: round-2 decode variant was +9..+21 μs wall clock).
- `dense_minimal_swiglu` re-measured on device time: conv_dense prefill −1627 μs (−29%), decode
  +116 μs. Only `num_dense_layers` = 2 of 40 layers use it (~3.3 ms per 2048-token prompt, vs
  ~13 s of MoE prefill at this stage); a prefill-only variant needs a second packed copy of w1/w3
  (2 x 96 MiB bf16) -> rejected on cost/benefit; revisit after MoE prefill dispatch (stage 3).
Note on A/B noise: wall-clock decisions within ~±20 μs (kv_fused_update +2, router_fold −2..−6)
are inside run-to-run noise; both are op-count reductions/increases and router_fold is now
subsumed by router_tile_mask's tile-only path.
- `tracy_ab/` was recorded while `residual_fold_prefill` was still in REJECTED_FUSIONS, so its
  `default` dirs exclude the fold and `default_plus_residual_fold_prefill` equals the final default.
  Final before/after (`before_after.txt`, `tracy/{functional,fused}/`) uses the final default set.

## Final evidence (final code)
- `TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v tests/test_fused_decoder.py` -> 39 passed (env logged 36x).
- Synthetic: 15 passed. Watcher (APPEND, 12 tests): 48 attach/48 detach, 15 dumps, 0 error lines.
- Functional control on the same harness: 39 passed (`logs/functional_control_real_weights_all.log`).
- Tracy before/after (`before_after.txt`): conv_dense prefill 5652.7 -> 5145.2 μs, decode
  687.3 -> 684.6 μs; conv_moe 424737 -> 355552 μs / 7102.1 -> 6057.6 μs; attn_moe
  415325 -> 337473 μs / 7501.0 -> 6139.7 μs.

## Stage review round 2: more-work-needed (conv prefill layout ops) -> conv_hybrid_shift
Reviewer: `conv_tile_shift` hides untilize/tilize inside `ttnn.slice` (final Tracy: 2 untilize +
2 tilize), README claim wrong. Added `conv_hybrid_shift`: one untilize of the 32+S-row extended
tensor, the two non-tile-aligned shifts (rows 30, 31) as row-major slice + tilize, the aligned shift
(row 32) sliced directly in TILE. Same PCC (conv_dense 0.99960/0.99931 update 0.99991/0.99984;
conv_moe 0.99984/0.99999 update 0.99387/0.99980).
Tracy A/B (`tracy_ab_conv/`; note: a first attempt was invalid because DEFAULT_FUSIONS includes every
non-rejected flag, so all three specs ran the hybrid — rerun with explicit removals):
- conv_dense prefill: hybrid 5093.1 μs (1 untilize 45.2 + 2 tilize 196.2), tile slices 5147.5 μs
  (2 untilize 92.0 + 2 tilize 195.1), row-major path 5199.1 μs (1 untilize 45.7 + 3 tilize 294.4).
- conv_moe prefill: 355502.8 / 355530.7 / 355601.9 μs. Decode unchanged.
-> conv_hybrid_shift kept, conv_tile_shift marked superseded. ttnn.conv1d not tried (see README).
`_route_tile_mask` docstring now states the exact-tie behaviour.
Final evidence re-run on this code: fused real weights 39 passed (TRACE_ALLOC_TRACKING), synthetic 15
passed, watcher 12 passed (48/48 attach/detach, 14 completed dumps, 0 error lines), Tracy
`before_after.txt`: conv_dense 5652.7 -> 5099.7 / 687.3 -> 685.0 μs; conv_moe 424737 -> 355490 /
7102.1 -> 6056.9 μs; attn_moe 415325 -> 337340 / 7501.0 -> 6142.1 μs. PCC minima unchanged.

## Stage review round 3: clean-pass
Independent reviewer (read-only) re-verified the conv hybrid shift, A/B CSVs, final evidence
timestamps and README claims: `clean-pass`, no required work. Carried notes: sparse_matmul
fused_activation wrong output (not in default path) -> stage 3 note/upstream issue; token-dispatch
MoE prefill and dense_minimal_swiglu revisit in stage 3; conv_moe update-PCC margin ~0.001.
- Commit hooks reformatted stage-2 files (black/isort/whitespace); post-format smoke: 12 fused tests (batch32, slot_reuse, perf_determinism, 65-2048) pass under TT_METAL_TRACE_ALLOC_TRACKING=1.

Stage 2 local commit (not pushed): tt-metal branch `autoport/lfm2_5_24b_a2b` `fee73388`.
