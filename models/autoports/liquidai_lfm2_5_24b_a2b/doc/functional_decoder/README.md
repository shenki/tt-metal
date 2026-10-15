# LFM2.5-24B-A2B — functional decoder (stage 1)

HF model `LiquidAI/LFM2.5-24B-A2B` @ `1769516ac588c4c558089c94f27d96b0dc67b183`
(`Lfm2MoeForCausalLM`, transformers 5.12.1). Hardware: one Blackhole p150b (1x1 mesh) of a
4-chip QuietBox. tt-metal `main` @ `ac8f82cf3c`, branch `autoport/lfm2_5_24b_a2b`.

## Layer kinds

| kind | layers | mixer | FFN | tested layer |
|---|---|---|---|---|
| `conv_dense` | 0, 1 | ShortConv (K=3, gated) | dense SwiGLU, 11776 | 0 |
| `conv_moe` | 3–5, 7–9, … | ShortConv | MoE 64 experts top-4, 1536 | 3 |
| `attn_moe` | 2, 6, 10, …, 38 | GQA 32q/8kv, hd 64, q/k RMSNorm, RoPE θ=1e7 | MoE | 2 |

## API (`tt/functional_decoder.py`)

```python
dec = FunctionalDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=i, mesh_device=md,
                                        max_batch_size=32, weight_dtype=ttnn.bfloat16)
rope = RopeSetup(md, args, max_seq_len=None)   # cos/sin tables (default: 131072)
dec.mixer.allocate_kv_cache(num_blocks, block_size)   # attention layers only

# prefill: one request, one chunk; x [1,1,S_pad,H], S_pad % 32 == 0
dec.prefill_forward(x, rope=rope, chunk_start=p, page_table=pt, chunk_page_table=cpt,
                    chunk_start_tensor=cst)                       # attention
dec.prefill_forward(x, conv_inputs=ConvPrefillInputs.create(md, slot=s, valid_len=n,
                    padded_len=S_pad, continuation=p > 0))        # ShortConv

# decode: all 32 slots; x [1,1,32,H]
cos, sin = rope.decode_cos_sin(rope_pos)                          # uint32 [1,32]
dec.decode_forward(x, rope=rope, rope_cos=cos, rope_sin=sin, cur_pos=cur, page_table=pt)
dec.decode_forward(x, slot_active=act)    # ShortConv; act bf16 [1,1,32,1] = (cur_pos >= 0)
```

The full contract (padding, page-table width multiple of 8 int32, `num_blocks >=` table
width, `cur_pos = -1` / `slot_active = 0` for idle slots, `chunk_start` a multiple of the KV
block size — asserted — with the SDPA chunk chosen internally, slot reset on a fresh request)
is in the module docstring. Design notes:

* **ShortConv state** lives on device as two `[1,1,32,H]` taps (row = slot). Prefill
  prepends a 32-row tile holding the slot's previous two inputs (zeros for a fresh request),
  computes the causal conv by three shifted slices, and writes the slot's last two rows back
  with one-hot selection matmuls; other slots are untouched. Decode shifts the taps only for
  active slots (`ttnn.where(slot_active, ...)`, exact), so a slot that is mid-prefill, waiting
  to join decode, or free keeps its state while other slots decode.
* **Attention**: paged KV cache (block 64), `paged_fill_cache` + `chunked_scaled_dot_product_attention`
  in prefill, `paged_update_cache` + `paged_scaled_dot_product_attention_decode` in decode.
  RoPE is rotate-half as `x*cos + (x@R)*sin`; decode cos/sin via `ttnn.embedding`.
* **MoE**: fp32 router matmul (HiFi4), sigmoid, + expert_bias, **fp32 `topk`** (bf16 topk
  picked a different expert set for ~5% of tokens — measured 212/4096 rows on layer 2,
  0/4096 with fp32), gather/normalize (+1e-6), row-major fp32 `scatter` into a dense
  `[T,64]` routing tensor; experts via `sparse_matmul` (fused `[up|gate]` + `swiglu`, then
  down), weight by routing and `fast_reduce_nc` over experts. Prefill in groups of 512 tokens.
* All weights bf16, DRAM; matmuls HiFi2 fp32-acc; norms/SDPA/router HiFi4.

## Accuracy (real weights, HF fp32 layer reference)

`tests/` — 39 tests (36 in `test_functional_decoder.py`, 3 in `test_perf_determinism.py`),
all pass, run with `TT_METAL_TRACE_ALLOC_TRACKING=1` (`logs/pytest_real_weights_all.log`).
Each case compares TT prefill rows and decode rows against one causal full-sequence HF
`Lfm2MoeDecoderLayer` forward. **Every decode row comes from `execute_trace` replay** (no
eager decode path is used by any test).

| kind | `run_case` rows | min prefill / decode PCC | interleaved rows | min prefill / decode PCC |
|---|---|---|---|---|
| conv_dense | 42 | 0.999604 / 0.999309 | 12 | 0.999658 / 0.999321 |
| conv_moe | 42 | 0.999817 / 0.999994 | 12 | 0.999798 / 0.999749 |
| attn_moe | 42 | 0.999737 / 0.999667 | 12 | 0.999796 / 0.999390 |

Acceptance bar 0.995 (default). Synthetic-weight CI path (`LFM2_REAL_WEIGHTS=0`, weights
drawn from `tests/weight_stats.json`): 15 selected tests pass (`logs/pytest_synthetic_weights.log`).

Reference check: `reference/hf_layer.py` vs the full HF model's own hidden states:
PCC ≥ 0.999977 for layers 0, 2, 3.

## Coverage

| claim | evidence | remaining risk |
|---|---|---|
| full advertised context 131072 | `test_decoder_full_context`: 131069-token prompt in 2048 chunks + traced decode at positions 131069..131071, all 3 kinds, PCC ≥ 0.999667; KV reserved for 32×131072 (65537 blocks, 8 GiB/layer) | full-model fit across 40 layers is a later-stage check (`../context_contract.json`) |
| non-aligned lengths | prompts 1, 31, 33, 65, 77, 300, 2047, 2049, 4100, 131069; aligned controls 32, 64, 2048 | — |
| chunked prefill | chunk 128 (300 → 3 chunks), 256, 2048; unchunked control (300 in one chunk); SDPA chunk derived inside `Attention.prefill_forward` | prefill is eager (trace not required for this stage) |
| paged cache / page tables | random block permutation per user, scratch block 0, distinct pages per request, table width padded to 8 entries | — |
| current position | 32 slots at mixed positions; idle slots `cur_pos = -1`, `slot_active = 0`, garbage input rows | — |
| batch 32 | `test_decoder_batch32`: 32 users, lengths 1..200, concurrent traced decode | — |
| interleaved prefill/decode (serving-like) | `run_interleaved` (`test_decoder_slot_reuse`, `test_decoder_increasing_lengths_same_instance`): one decoder instance, ONE captured decode trace; 2 background users decode on every replay while foreground requests (700, 33 @ chunk 256; 2047, 2048, 2049, 4100, 32, 33 @ chunk 2048, all in one slot) prefill chunk by chunk with a replay after every chunk, then wait 0–2 replays before joining decode | prefill shapes are compiled before capture (warm-up); a generator must do the same |
| conv state ownership | negative control (all slots forced active = the pre-fix behaviour): conv_dense interleaved decode PCC drops to 0.837 and the tests fail; with `slot_active` masking ≥ 0.999321 | — |
| fresh request reset | slot reuse (700 → 33 in slot 7) and the six requests through slot 9 | — |
| determinism | `test_perf_determinism.py`: repeated fresh prefill bit-identical; two replays from restored state bit-identical | — |
| no host fallback | `no_host_fallback()` around measured prefill, trace capture and replays: `TorchFunctionMode` raising on any torch call, `ttnn.from_torch/to_torch/Tensor.cpu` patched to raise, `throw_exception_on_fallback=True`; guard verified to trip on each | continuation-chunk prefill is not under the guard (same ops as the fresh chunk) |
| watcher clean | `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1`, 12 tests (batch32 ×3, 2049 ×3, slot_reuse/interleaved ×3, perf ×3): 48 attach/48 detach (12 sessions x 4 devices), 18 dumps, no assert/sanitize/overflow/fault/error lines (`logs/watcher/`) | — |
| trace alloc safety | entire suite (39 tests incl. interleaved prefill-after-capture) under `TT_METAL_TRACE_ALLOC_TRACKING=1`, program cache included; env value logged by each case. The tracker caught one real issue during development (the request page table kept alive across replays; now per chunk) | — |

## Performance (warmed, 1 chip, bf16 weights)

Tracy + tt-perf-report 1.4.1. Prefill: one fresh 2048-token chunk. Decode: 32 slots, one
traced step, 5 replays in the window (every op count in the window is a multiple of 5, i.e.
no profiler rows dropped). Time from `tt-perf-report --csv` column `Device Time` (μs), summed
over the signposted window and divided by the pass count.

| kind | prefill 2048 (device μs) | decode step, batch 32 (device μs) | top ops |
|---|---|---|---|
| conv_dense | 5,662 | 687 | dense MLP matmuls |
| conv_moe | 424,846 | 7,103 | `sparse_matmul` down (all 64 experts) + up/gate |
| attn_moe | 415,542 | 7,503 | same, + SDPA decode |

The MoE dominates: the down projection runs every expert densely (all-ones sparsity) and
prefill gate/up uses 32-row groups. This is the functional baseline; optimization is stage 2/3.

Artifacts per kind in `tracy/<kind>/`: `ops.csv` (raw Tracy), `{prefill,decode}_perf_report.txt`
(table), `.csv`, `.console.log`, stacked csv/png.

```bash
python -m tracy -r -p -v -m pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_perf_determinism.py -k <kind>
tt-perf-report tracy/<kind>/ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --csv tracy/<kind>/decode_perf_report.csv --no-advice
tt-perf-report tracy/<kind>/ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-summary --no-advice > tracy/<kind>/decode_perf_report.txt
```

## Limitations

* Prefill takes one request per call; the generator owns chunk planning and padding, keeps
  `chunk_start` a multiple of the KV block size, and must compile every prefill shape it will
  use before capturing the decode trace (each distinct `S_pad` is a new program).
* ShortConv prefill needs small host-built selection tensors per chunk (`ConvPrefillInputs`)
  and decode needs the `slot_active` mask; both are inputs built outside the pass.
* MoE performance (above).

## Commands

```bash
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/                       # real weights if downloaded
LFM2_REAL_WEIGHTS=0 pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/   # synthetic
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=<dir> pytest ... -k "batch32 or 2049-2048 or slot_reuse or perf_determinism"
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v models/autoports/liquidai_lfm2_5_24b_a2b/tests/
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.collect_weight_stats
python models/autoports/liquidai_lfm2_5_24b_a2b/tests/emulate_algorithms_cpu.py
```
