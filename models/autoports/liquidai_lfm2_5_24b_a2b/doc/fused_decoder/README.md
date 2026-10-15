# LFM2.5-24B-A2B — fused decoder (stage 2)

`tt/fused_decoder.py`: `FusedDecoder` (subclass of `FunctionalDecoder`) with the same
prefill/decode contract (see the functional README and module docstring), the op graph rewritten
with dedicated tt-metal ops. Every rewrite is a named fusion; `DEFAULT_FUSIONS` is the measured,
selected set and `REJECTED_FUSIONS` stay switchable for re-measurement. One 1x1 Blackhole p150b.

```python
dec = FusedDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=i, mesh_device=md)  # fusions=DEFAULT_FUSIONS
rope = make_rope(md, args, max_seq_len)   # FusedRope: decode cos/sin height-sharded for rotary_embedding_hf
```

Only API difference: with `rope_hf`, `rope.decode_cos_sin(rope_pos)` returns height-sharded cos/sin
(one user per core, row-major over the full compute-grid width).

## Selected fusions (kept)

| fusion | kind | rewrite |
|---|---|---|
| `qkv_heads_decode` | dedicated op | 3 slice + 3 reshape + 2 pad + 2 reshard -> `nlp_create_qkv_heads_decode` |
| `rope_hf` | dedicated op | `x*cos + (x@R)*sin` (matmul, mul, mac) -> `rotary_embedding_hf` (prefill + decode) |
| `qk_norm_weight` | op merge | `rms_norm` + `multiply(weight)` -> `rms_norm(weight=)` |
| `concat_heads_decode` | dedicated op | decode reshape -> `nlp_concat_heads_decode` |
| `router_fold` | op merge / rewrite | router linear + sigmoid -> `linear(activation="sigmoid")`; drop ×1.0 scale |
| `router_tile_mask` | graph rewrite | topk -> gather -> 3x untilize -> scatter -> tilize -> tile-only mask `scores * (biased >= kth topk value)`, normalized (no layout changes; identical selection to the functional router) |
| `moe_decode_no_permute` | graph rewrite | decode G=1: transpose(E,G) -> reshape |
| `moe_swiglu_mul_act` | op merge | `ttnn.swiglu` composite (slice, slice, silu, mul) -> slices + `mul(input_tensor_b_activations=[SILU])` |
| `moe_down_union_sparsity` | graph rewrite | decode down proj over all 64 experts -> only experts selected by any row (sparse_matmul union mask) |
| `moe_prefill_down_sparse` | graph rewrite | prefill permute + all-expert down -> per-(32-row tile, expert) sparse down in `[G,E,R,I]` layout, no permute |
| `conv_state_inplace` | op merge | `where` + `copy` -> `where(output_tensor=state)` |
| `conv_hybrid_shift` | graph rewrite | conv prefill untilize + 3x (slice, tilize) -> 1 untilize, 2x (slice, tilize) for the unaligned shifts, aligned shift sliced directly in TILE (layout ops 340 -> 241 μs device; `tracy_ab_conv/`) |
| `residual_fold_prefill` | dedicated op | prefill out_proj / w2 matmul + residual add -> `dit_minimal_matmul_addcmul_fused` (decode M=32 keeps linear + add) |

## Rejected (measured)

Device time = Tracy (`tracy_ab/`); wall = `bench_fusions.py` wall clock (noise ≈ ±20 μs decode, ±0.3 ms prefill).

| fusion | result | reason |
|---|---|---|
| `moe_grouped_topk` (`deepseek_prefill.moe_grouped_topk`, ~12 ops -> 1) | −65 μs decode (wall) | precision: 5/512 vs 3/512 selection flips (L3), routing-weight err 0.0064 vs 0.0031, conv_moe update PCC 0.99387 -> 0.99088 (`logs/router_selection_probe.log`, `logs/precision_conv_moe.jsonl`); `router_tile_mask` recovers −52..−55 μs (device) with no precision change |
| `moe_split_gate_up` (slice pushed into 2 half-width sparse_matmuls) | decode +2.8 ms, prefill +170 ms (wall) | slower: N=I gives 1 output tile per core and reads in0 twice; with SiLU as the gate matmul's `fused_activation` the output is wrong (PCC 0.984), see work_log |
| `dense_minimal_swiglu` (`minimal_matmul(fuse_swiglu)`) | conv_dense decode +116 μs, prefill −1.63 ms (device) | decode slower; a prefill-only path needs a second packed copy of w1/w3 (2x96 MiB) for 2 of 40 layers (~3 ms per 2048-token prompt) — cost/benefit, revisit later |
| `dense_merged_w13` (one [w1\|w3] linear) | decode +9 μs (wall) | slower |
| `kv_fused_update` (`paged_fused_update_cache`) | +2 μs (wall, within noise) | no gain at batch 32 (llama31_8b_qb2 notes it only wins at batch ≤ 4) |
| `conv_split_in_proj` (slice pushed into 3 matmuls) | +46 μs (wall) | slower |
| `conv_tile_shift` (all 3 shifts as TILE slices) | conv_dense prefill 5147.5 vs hybrid 5093.1 μs (device) | an unaligned TILE `ttnn.slice` is internally untilize + slice + tilize: 2 untilize + 2 tilize (287 μs) vs the hybrid's 1 + 2 (241 μs) — superseded |
| depthwise `ttnn.conv1d` / `kda.qkv_causal_conv1d_silu` | not run | 3-tap depthwise causal conv: `conv1d` needs ROW_MAJOR/sharded im2col setup for a 3-tap kernel (precedent: `models/demos/multimodal/lfm25_vl/tt/short_conv.py:17-18` uses shifted multiplies, "no im2col/ttnn.conv1d needed"); `qkv_causal_conv1d_silu` forces SiLU and 4 taps (LFM2 has no activation) |

All graph-fusing skill patterns were assessed: activations (silu already
fused as an input activation), softmax/SDPA (already dedicated SDPA ops), RMSNorm (already
`rms_norm`), distributed norm (n/a single chip), split-QKV/create-heads (prefill already
`nlp_create_qkv_heads`; decode applied), concat heads (prefill already `nlp_concat_heads`; decode
applied), RoPE (applied), TopK (already `topk`; grouped variant rejected), shared-LHS matmul (wqkv and
conv B/C/x already packed; w1|w3 rejected), permute-reshape (decode MoE applied), matmul+activation
(router applied), input-arg activation (applied), matmul+bias / conv+bias / batchnorm / pooling /
spatial mean / numeric-stable softmax / scaled-sum (not present in this graph), reduction+reshape
(already keepdim), slice-after-matmul (tried at both sites — conv in_proj and MoE gate/up — both
slower, rejected), tilize/untilize removal (router: all removed; conv prefill: reduced to the
1 untilize + 2 tilize that the two non-tile-aligned shifts need).

## Bug found and fixed during fusing

`paged_scaled_dot_product_attention_decode` with height-sharded Q places batch b's reducer at
`(b % grid.x, b // grid.x)` of its compute grid. Q sharded on an 8-wide grid with the 11-wide
Blackhole SDPA grid made reducers read other cores' L1: wrong output (max|err| 2.3–3.5 vs torch),
run-to-run nondeterminism after program-cache hits. Our fix: `decode_head_mem` shards decode heads
row-major over the full grid width (max|err| 0.0060, fresh == cached). Minimal repro:
`repro/sdpa_decode_sharded_q_grid_mismatch.py`. The op does not validate this (worth an upstream issue).

The bug passed the layer-output PCC (the residual dominates), so every test now also checks the
**update PCC** (`out − x` vs `ref − x`, bar 0.99 from the functional baseline minimum 0.991), real weights.

## Accuracy (real weights; `logs/pytest_fused_real_weights_all.log`, 39 passed under `TT_METAL_TRACE_ALLOC_TRACKING=1`)

| kind | rows | min output PCC prefill / decode | min update PCC prefill / decode |
|---|---|---|---|
| conv_dense | 42 + 12 interleaved | 0.999604 / 0.999309 | 0.999248 / 0.999656 |
| conv_moe | 42 + 12 | 0.999798 / 0.999749 | 0.992301 / 0.991175 |
| attn_moe | 42 + 12 | 0.999737 / 0.999389 | 0.996485 / 0.991647 |

Functional decoder on identical inputs (`logs/functional_control_real_weights_all.log`, 39 passed):
same minima to 4 decimals. Full context: 131069-token prompt + traced decode at 131069..131071 for
all kinds (update PCC ≥ 0.9931). Synthetic weights: 15 passed (`logs/pytest_fused_synthetic_weights.log`);
the update bar is real-weight only (synthetic control in `work_log.md`: HF itself in bf16 scores
0.9196 on one synthetic user).

## Performance (Tracy device time, `before_after.txt`)

Same test (`run_perf_determinism`: fresh 2048-token prefill chunk; traced batch-32 decode, 5 replays;
all windows complete). tt-perf-report tables/CSVs: `tracy/{functional,fused}/<kind>/`.

| kind | prefill 2048 functional → fused (ms) | decode step functional → fused (μs) |
|---|---|---|
| conv_dense | 5.65 → 5.10 (1.11×) | 687 → 685 (1.00×) |
| conv_moe | 424.7 → 355.5 (1.19×) | 7102 → 6057 (1.17×) |
| attn_moe | 415.3 → 337.3 (1.23×) | 7501 → 6142 (1.22×) |

conv_dense decode is unchanged: its decode graph is the in_proj/out_proj/MLP matmuls plus a few
elementwise ops; every fusion candidate there (merged w1|w3, fused-swiglu minimal matmul, split
in_proj, decode residual fold) measured slower, and the in-place conv state saves ~3 μs.

Remaining time is in non-fusion items handed to stage 3: MoE expert matmuls (dense over experts in
prefill; token dispatch ops investigated — `offset_cumsum` needs >1 device, dispatch/combine need
fabric on 1 chip), SDPA decode (~640 μs) program config, single-core norms (2× ~33 μs), dtypes.

## Other checks
- No host fallback: `test_fused_perf_determinism_no_fallback` (guarded prefill, capture, replays) for all kinds.
- Determinism: repeated prefill and replays bit-identical (same test).
- Trace safety: whole suite under `TT_METAL_TRACE_ALLOC_TRACKING=1`; interleaved prefill/decode with one trace.
- Watcher: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1`, 12 tests, 48 attach/48 detach, 14 completed
  dumps, 0 error lines (`logs/watcher/`).
- SDPA repro output: `logs/sdpa_decode_sharded_q_repro.log`.
- Context contract unchanged (bf16 KV and weights; no capacity change).

## Commands
```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_fused_decoder.py
LFM2_FUSIONS=default-<fusion> pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_fusions.py   # A/B
python -m tracy -r -p -v -m pytest -q models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_fused_decoder.py::test_fused_perf_determinism_no_fallback -k <kind>
python models/autoports/liquidai_lfm2_5_24b_a2b/doc/fused_decoder/repro/sdpa_decode_sharded_q_grid_mismatch.py
```
