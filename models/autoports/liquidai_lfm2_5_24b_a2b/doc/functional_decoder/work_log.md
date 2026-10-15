# Functional decoder work log — LiquidAI/LFM2.5-24B-A2B

## Environment (2026-10-15)
- Host: tt-quietbox, 4x Blackhole p150 (QuietBox). Stage 1 targets a 1x1 mesh.
- tt-metal: `main` @ ac8f82cf3c10c3fff1adfca8d7eeb796dbb9c288 (shallow clone), local branch `autoport/lfm2_5_24b_a2b`.
- Build: `./build_metal.sh --cxx-compiler-path /usr/bin/clang++ --c-compiler-path /usr/bin/clang --disable-warnings-as-errors --without-distributed -c`
  (host has clang 21 only — no clang-20 toolchain — and no OpenMPI; rebuilt once with the venv active
  because the first configure picked the system Python 3.14 headers.)
- venv: `./create_venv.sh --python-version 3.12` (torch 2.11.0+cpu, transformers 5.12.1 = model's version).
- tt-perf-report 1.4.1.
- HF model: LiquidAI/LFM2.5-24B-A2B @ 1769516ac588c4c558089c94f27d96b0dc67b183.
- Serving pins for later stages: vllm-tt-plugin d84049edcece77c64772a39e92f920f5679efce4 -> vLLM v0.30.0 (docs/install-vllm-tt.sh).

## Model reading notes
- 40 layers: 30 `conv` (ShortConv: in_proj H->3H, B*x, depthwise causal conv K=3, C*conv, out_proj),
  10 `full_attention` (layers 2,6,...,38; GQA 32q/8kv, head_dim 64, per-head q/k RMSNorm before
  rotate-half RoPE theta 1e7). Layers 0,1 dense SwiGLU (11776); 2..39 MoE (64 experts, top-4,
  expert I=1536, sigmoid router, selection-only expert_bias, renormalized weights, scale 1.0).
- Layer kinds: conv_dense (0,1), conv_moe (3,4,5,7,...), attn_moe (2,6,...,38).
- Checkpoint stores experts per expert (w1=gate, w3=up, w2=down); tied embeddings; final norm is
  `embedding_norm` (applied after the last layer).
- Advertised context: max_position_embeddings = 131072.

## Design decisions
- ShortConv decode state lives on device as two persistent [1,1,32,H] taps (row = slot);
  decode shifts them with ttnn.copy (trace-safe). Prefill writes the slot's last two logical rows
  with host-built one-hot selection matmuls + keep-mask (no host round trip, compiled once per
  chunk shape), and pulls the previous carry rows for continuation chunks the same way.
- CPU emulation of exactly these algorithms vs HF Lfm2MoeShortConv / SparseMoeBlock
  (`tests/emulate_algorithms_cpu.py`): ShortConv exact (<1e-3 fp32) for prompts 1,2,3,31,65,300,
  chunk carries, slots 0/5/31 with garbage in other slots; MoE routing/combine max err 6e-7.
- Attention prefill: paged_fill_cache via chunk-local page table, then
  chunked_scaled_dot_product_attention with chunk_start_idx_tensor (one code path for fresh and
  continuation chunks). Decode: paged_update_cache + paged SDPA decode with cur_pos tensor.
- MoE: dense routing tensor doubles as sparse_matmul sparsity (nnz=None, Blackhole deadlock note
  from qwen36/gpt_oss); router logits/sigmoid/bias in fp32.

## Hardware availability
- 2026-10-15: all 4 chips held by qwen3.8-flash-next.service; user asked not to stop it. Device
  runs are pending the user freeing the chips.

## 2026-10-15 — weights downloaded, host-side reference checks
- Pinned checkpoint `1769516ac588c4c558089c94f27d96b0dc67b183` fully downloaded (45 GB, HF cache).
- `python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.collect_weight_stats` → `tests/weight_stats.json`
  (layers 0/2/3 + embedding; embed std 0.0193).
- HF CPU sanity (bf16, greedy, chat template, "capital of France"): coherent output; the model opens
  with a reasoning preamble ("We need to answer: ...") — expected for LFM2.5 and relevant for later
  qualitative checks (allow enough new tokens).
- Reference wrapper `reference/hf_layer.py` (fp32, layer-only) vs the full HF model's own hidden
  states (bf16): PCC layer 0 = 0.999977, layer 2 = 0.999996, layer 3 = 0.999997.
- Device work still blocked: all 4 chips held by `qwen3.8-flash-next.service` (user decision: do not stop it).

## 2026-10-15 — device bring-up (1x1 Blackhole p150b)
User stopped `qwen3.8-flash-next.service` ~17:55. `tt-smi -ls --local`: 4x Blackhole p150b.
Mesh smoke (bf16 64x64 matmul) on MeshShape(1,1) and (1,4): OK.

Fixes found on device (in order):
1. `ttnn.scatter` rejects fp32 TILE (`scatter.cpp:109`) -> scatter in ROW_MAJOR, back to TILE.
2. SDPA: `page_table_stick_size % 32 == 0` -> page-table rows padded to a multiple of 8 int32
   entries (scratch block 0) (`tests/harness.py::pad_page_table_row`).
3. `paged_update_cache`: page-table width must be <= num_blocks -> cache sized to at least the
   table width in the harness.
4. RoPE prefill slice past `max_seq_len` for tile-padded chunks -> `RopeSetup` rounds its table
   length up to a tile multiple.
5. MoE selection precision: probe (`route_probe`, 4096 real-embedding tokens, layers 2/3/20):
   bf16 topk on biased scores selected a different expert set for 212/210/182 rows; fp32 topk
   0/0/0. Before: min batch32 prefill PCC 0.995452 (attn_moe), conv_moe decode 0.997834; after
   fp32 topk: 0.999737 / 0.999994. Router now does topk in fp32.

Results (final code):
- `pytest -v models/autoports/liquidai_lfm2_5_24b_a2b/tests/` -> 36 passed in 393 s
  (`logs/pytest_real_weights_all.log`). Min PCC: conv_dense 0.999604/0.999309,
  conv_moe 0.999835/0.999994, attn_moe 0.999737/0.999745 (prefill/decode).
- Full context (`-k full_context`): 131071-token prompt + decode at 131071, all kinds pass;
  KV 65537 blocks (32 x 131072 tokens, 8 GiB) allocated for the attention layer
  (that log was superseded and removed; see the final run below; host RSS 23 GB for the HF fp32 reference).
- Synthetic weights (`LFM2_REAL_WEIGHTS=0`): 12 selected tests pass (`logs/pytest_synthetic_weights.log`).
- Perf/determinism/no-fallback (`tests/test_perf_determinism.py`): 3 pass; guard checked to
  trip on torch.*, tensor methods, ttnn.from_torch, ttnn.to_torch.
- Tracy: `python -m tracy -r -p -v -m pytest .../test_perf_determinism.py -k <kind>`, CSVs
  copied to `tracy/<kind>/ops.csv`, tt-perf-report tables/CSVs alongside. Per-pass device time:
  conv_dense prefill 5.7 ms / decode 0.67 ms; conv_moe 425 ms / 7.08 ms; attn_moe 415 ms / 7.45 ms.
  MoE sparse_matmul dominates (down proj runs all 64 experts) — stage 3 target.
- Watcher: `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=logs/watcher`, 9 tests pass, watcher.log
  has 4 dumps and 0 assert/sanitize/overflow/fault/error lines.
- `TT_METAL_TRACE_ALLOC_TRACKING=1` (program cache included): batch32 + perf tests, 6 pass
  (`logs/trace_alloc_tracking.log`); the checker raises in execute_trace on unsafe buffers.
- `doc/context_contract.json`: advertised 131072, supported 131072, no reduction.

## 2026-10-15 — stage-review round 1: more-work-needed, fixes
Reviewer findings (P1 ShortConv decode shifted every slot's state; P2 SDPA chunk set by the
caller / chunk_start unchecked; P2 "all decode traced" untrue for the eager same-slot runner
and no prefill-after-capture coverage; P2 watcher log kept only the last session; also
incomplete attn_moe profiler window, stale full_context.log, no 32-token case).
Fixes:
- `ShortConv.decode_forward(x, slot_active)`: taps update via `ttnn.where(slot_active, new, old)`
  (probe: where with [1,1,32,1] broadcast condition is bit-exact vs torch.where).
  `DecodeInputs` keeps a persistent `slot_active` derived from `cur_pos >= 0`.
- `Attention.prefill_forward` asserts `chunk_start % block_size == 0` and derives the SDPA
  chunk itself (`prefill_sdpa_chunk`); harness no longer touches `mixer.sdpa_chunk`.
- New `run_interleaved` (replaces the eager `run_sequential_same_slot`): one decode trace,
  background users decode after every foreground prefill chunk, join delays 0–2, slot reuse,
  prefill shapes warmed before capture. First run under TRACE_ALLOC_TRACKING failed:
  "Found 1 device buffer(s) still alive before trace replay ... [op: ttnn.to_device]" — the
  request's full page table lived across replays; now created per chunk. Then 6/6 pass.
- Negative control (`-p unmasked_plugin`, all slots forced active): conv_dense interleaved
  decode PCC 0.994/0.989/0.891/0.923/0.970/0.984/0.898/0.837 -> 2 tests fail; conv_moe min
  0.996266. With the fix: >= 0.999321.
- Tests: added 32-token case; full context now 131069 prompt + 3 traced decode steps
  (positions 131069..131071).
Final evidence (same code for all):
- `TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v .../tests/` -> 39 passed in 380 s
  (`logs/pytest_real_weights_all.log`; env value logged 36x).
- `LFM2_REAL_WEIGHTS=0 ... -k "batch32 or 2049-2048 or 65-2048 or slot_reuse or perf_determinism"` -> 15 passed.
- Watcher (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1`, 12 tests) -> 12 passed; watcher.log
  48 attach / 48 detach / 18 dumps (36 start+completed lines) / 0 error-like lines.
- Tracy re-profile with DECODE_ITERS=5: all window op counts are multiples of the pass count.
  conv_dense 5,662 / 687 μs, conv_moe 424,846 / 7,103 μs, attn_moe 415,542 / 7,503 μs
  (prefill 2048 / decode step). Source CSV paths in `tracy/<kind>/ops_source.txt`.
- Negative-control artifacts: `negative_control/unmasked_plugin.py`, `negative_control/negative_control.log`
  (`PYTHONPATH=$PWD:<dir> pytest -p unmasked_plugin .../test_functional_decoder.py -k "(slot_reuse or increasing) and conv"`).

## 2026-10-15 — stage-review round 2: clean-pass
Independent reviewer subagent (read-only, no device access) re-verified all round-1 findings
against code and artifacts: verdict `clean-pass`, no required work. Noted: dump count wording
(fixed: 18 dumps), negative control ran ~30 s before the final env-logging-only test edit,
conv_moe layer-output PCC is a weak detector for conv-mixer bugs (conv_dense catches it; consider
mixer-level PCC in later stages), MoE dense work and per-S_pad prefill programs (stage 2/3 and
generator concerns).
- Commit hooks (black/isort/autoflake/whitespace) reformatted stage files and dropped unused imports; post-format smoke: 12 selected tests (batch32, 65-2048, slot_reuse, perf_determinism) pass under TT_METAL_TRACE_ALLOC_TRACKING=1.

Stage 1 local commit (not pushed): tt-metal branch `autoport/lfm2_5_24b_a2b` `799a0875`.
