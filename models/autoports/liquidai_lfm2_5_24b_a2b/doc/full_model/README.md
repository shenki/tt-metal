# LFM2.5-24B-A2B full model (stage 6) — 1x4 Blackhole p150b, TP = 4

## Headline (all 40 layers, batch 1, prompt 128 / generate 128, greedy, wall clock, tracking / profiler off)

Source: `logs/final_r2.{log,json}` (final code; earlier run `logs/final_perf.json` agrees within 0.1 %).  Memory /
trace-region figures: `logs/final_all_runs.json` (measured after buckets 1 / 4 / 8 / 32 were captured).

| metric | value | boundary |
|---|---|---|
| TTFT | **222 ms** | `generate` start -> first token on host: eager chunked prefill (one 128-row chunk) + traced prefill-sampling replay + 128-byte readback |
| token-out decode | **106.0 t/s/u** (9.43 ms/step) | model trace + sampling trace + history-append trace per step, non-blocking, one history readback per request; no EOS polling |
| teacher-forcing decode | **101.9 t/s/u** (9.82 ms/step) | same replays + per-step 128-byte token readback + host callback + (if forced) one token write |
| traced decode, logits only | **9.06 ms/step** | model-trace replays only (no sampling / feedback): the PERF-style comparison |
| token-out, EOS polling every 16 steps (default free-running `generate`) | 105.4 t/s/u | as token-out + one history readback per 16 steps |
| batch token-out (buckets 4 / 8 / 32) | 13.83 / 19.26 / 26.24 ms/step | slots 0..b-1 active, mixed prompt lengths |

Readiness (fresh AIME24 chat-template reference, 157-token prompt + 100 reference tokens, all 40 layers):

| gate | top-1 | top-5 | top-100 |
|---|---|---|---|
| `run_prefill_check` | 98 % (98/100) | **100 %** | **100 %** |
| `run_teacher_forcing` (traced token-out path, `enable_trace=True`) | 98 % (98/100) | **100 %** | **100 %** |

Bars: top-5 >= 98 %, top-100 = 100 %: met.  The readiness teacher-forcing runner's own timer reports TTFT 450 ms (the
157-token prompt pads to the 256 bucket, and this first request of the process includes the deferred first capture of
the bucket-1 / prefill-sampling / history traces: 243 ms, `logs/final_all_runs.json` `tf_last_perf.capture_s`) and
103.5 t/s/u decode.  The headline TTFT above is measured after that one-time capture.

Free-running (`run_autoregressive`, 256-token budget) and the shared qualitative suite: coherent, no repetition /
language drift / control-token leakage; details in "Accuracy and qualitative verdict".

## What was built

* `tt/model.py` — `Lfm2Model`: device-only compute.  Embedding (tied `embed_tokens`, bf16 ROW_MAJOR replicated) ->
  40 x stage-5 `MultichipDecoder` (unchanged `MC_POLICY` / `DEFAULT_MESH_PLAN`) -> final norm (HF `embedding_norm`,
  applied after the stack: `modeling_lfm2_moe.py` `Lfm2MoeModel.forward`) -> vocab-parallel tied LM head.
  `decode_forward_device(bucket)`, `prefill_chunk(...)`, KV allocation / binding / zeroing, ShortConv state reset /
  snapshot / restore (prefix-caching hooks for stage 9), `logits_to_host` (compat / readiness only).
* `tt/generator.py` — `Lfm2Generator(readiness_check.contract.Generator)` + `build_generator(model_dir, mesh_device,
  **kw)`: low-level `prefill_forward` / `decode_forward` with explicit cache / page table / slots / positions / prompt
  lengths, high-level `prefill_logits` / `generate` / `reset`, sampling parameters per row, counters.
* `tests/test_full_model.py` (18 tests, reduced probe), `tests/readiness_driver.py` (readiness runners + perf + memory +
  qualitative on one 40-layer build), `tests/perf_full_model.py`, `tests/qualitative.py`, `tests/hf_probe.py` (HF CPU
  reference of a layer subset incl. embedding / final norm / LM head), `tests/test_full_model_perf.py` (Tracy probe),
  `tests/gen_reference.py` (readiness reference + metadata sidecar).

### Carried-forward decoder contract (nothing relaxed)
Stage-5 `MultichipDecoder` with `MC_POLICY` and `DEFAULT_MESH_PLAN`: weight / activation / KV (BFP8 paged cache, block
64) / CCL (bf16 `all_reduce_async` on persistent scratch) policy, the dtype / fidelity exceptions and the rejection
ledger (`doc/optimized_multichip_decoder/README.md`, `doc/optimized_decoder/README.md`) are unchanged — the model passes
the same objects.  Inter-layer residual: decode `[1, 1, 32, 2048]` bf16 width-sharded L1 on cores (0,0)-(7,0), shard
[32, 256] (`resid_mem`), identical on the 4 chips; the embedding output enters it once, the layers pass it on without
conversion, and the final norm reads it in place (`ShardedNorm` on the same 8-core grid).  Prefill: DRAM interleaved.
No replicated-stream or single-chip fallback.  Router near-ties (decode router-score error <= ~7e-3 vs HF fp32, stage 5)
are why the top-1 of the readiness gates is 98 %, not 100 %; top-5 / top-100 are complete.

### Embedding / final norm / LM head
* Embedding: one bf16 ROW_MAJOR `[128000, 2048]` table per chip (0.49 GiB); decode embeds all 32 rows (one program for
  every bucket; idle rows hold a valid id), `to_memory_config(resid_mem)` once.  Rejected: hidden-sharded table +
  per-step `all_gather` (one extra CCL per step on an op-count-bound decode).
* LM head = `embed_tokens.T`, vocab-parallel DRAM-sharded matmul, bfp8 / HiFi2 (the stage-3/5 projection policy;
  bf16 / bfp4 heads are datatype-sweep candidates).  Per-chip columns must be a multiple of 32 x 8 DRAM banks x readers:
  the unpadded 32000-column shard admits only 1 reader per bank (393.6 us); the vocab is therefore padded at its TAIL to
  4 x 32256 (zero weight rows) so chip d's column j is still token 32256 d + j, and the sampler masks the padded tail
  (`padded_vocab_size = 129024`).  Config sweep (`repro/head_sweep.py`, `logs/head_sweep_bfp8*.log`): (64 cores,
  in0_block_w 1, 3 readers) 175.6 us in a traced loop (169.4 us device in the probe profile, ~412 GB/s on 70 MB/chip);
  (32, 2, 3) 180.9; (16, 4, 2) 252.7; 32000-column (16, 1, 1) 393.6; 8 cores with 2/3 readers do not fit L1.
* Final norm: decode `ShardedNorm` on `resid_mem` (= the residual layout, HiFi4), prefill interleaved `rms_norm`.

### Sampling (split, canonical)
Both common samplers were read against this model:

| | `models/common/sampling/tt_sampling.py::TTSampling` (**chosen**) | `models/common/modules/sampling/sampling_1d.py::Sampling1D` (rejected) |
|---|---|---|
| state | device-resident k / p / temp / seeds refreshed by host copy (trace-stable; mode switch = param refresh, same graph) | k / p / temp passed per call (caller-owned device tensors; no persistent-state benefit here) |
| greedy exactness | `_adjust_values_for_tiebreak`: lowest-index tie-break for k=1 rows | no tie-break (unstable top-k order decides ties) |
| 1x4 ring | split path: local top-k + composite `ttnn.all_gather` of candidates; force-argmax `all_gather_async` Ring | argmax path forces Linear topology below 8 devices (`_get_argmax_all_gather_config`) |
| logprobs | yes | only on 8- / 32-device meshes |
| `tt_out_tok` | yes | yes |
| ecosystem | TT vLLM plugin model runner / sampling contract written against it; deployed on the same mesh class (qwen38 QB2 TP4) | experimental users only |

Token-out decode = model trace (40 layers + norm + head -> vocab-sharded bf16 logits `[1, 1, 32, 32256]` per chip +
`plus_one(cur_pos, skip_negative_entries)` + `plus_one(rope_pos)`) then sampling trace (`TTSampling(logits,
tt_out_tok=model.state.tokens)` + `plus_one(seeds)`): local top-32 per chip, all-gather of 4 x 32 candidates, device
index offsets, `ttnn.sampling`.  Greedy = k=1, p=0, temp=1 on that path (semantically greedy, tile-shaped candidates).
Measured alternatives (`repro/sampler_ab.py`, probe, each variant on its own generator over one model,
`logs/sampler_ab_probe.log`; all-layer: sampling trace 0.340 ms vs force-argmax 0.903 ms, token-out 106.05 vs 100.15 t/s/u, same tokens, `logs/final_r2.json`): split without padding (32256-wide top-k:
`topk_large_indices` route) 375.4 us / step for the sampling trace, split with `pad_logits_to_power_of_2` (32768-wide
top-k) **340.2 us (default)**, force-argmax (full-vocab `all_gather_async` + untilize + argmax) 868.5 us — identical tokens in all three.  Top-k / top-p
/ temperature use the same traces (`set_sampling_params`; tested greedy -> top-k -> greedy without recapture).
Prefill first token: last-row logits of each prefilled sequence accumulate (rows of other slots are exactly 0) into a
persistent `[32, 32256]` buffer sampled by a third trace into a separate token buffer (read once per request, then
written into the slot's decode token input at the request boundary without disturbing other slots).

### Traces, warm-up, lifetime
* Keys: model + sampling trace pair per decode bucket b in {1, 2, 4, 8, 16, 32} (`decode_bucket`: smallest bucket
  covering the highest active slot), + one prefill-sampling trace + one history-append trace.  Captured lazily on
  first use of a bucket (after phase 1), never recaptured or released during request handling; `release_traces` /
  `close` only at teardown.  Mode switches (greedy / sampled) and bucket switches select, never recapture (tested).
* Phase 1 (`prepare`): every prefill bucket (32 .. 2048) x {first chunk with all-logits head, continuation chunk with
  last-row head}, the prefill-sampling accumulate / sampler, every decode bucket + sampler, history append, slot reset /
  snapshot / restore, the whole-cache `reset()` zeroing (`ttnn.fill` of the BFP8 KV and the ShortConv state: the cache is
  zeroed once here) — on the scratch block and the scratch slot; tokens / positions / seeds / history cursor /
  ShortConv state restored afterwards.  6.6 s for 40 layers with a warm kernel cache (32.5 s cold).
* Prefill RoPE: per-chunk cos/sin gathered with `ttnn.embedding(positions, cos_rm)` (runtime positions; one program per
  bucket) and handed to the 10 attention layers through `_ChunkRope`; the decoder's compile-time `prefill_cos_sin`
  slice (one program per chunk start) is never used.  ShortConv prefill tail selection is runtime data (`windowed=False`).
  Chunked SDPA programs depend on the chunk start only through divisibility (first chunk vs 2048-aligned continuation).
* Trace region: 256 MiB requested (`open_mesh`); measured 68.1 MiB of 256 MiB with buckets 1 / 4 / 8 / 32 + the sampling /
  history traces captured (~17 MB per 40-layer model trace), so all six buckets fit.
* **Trace allocation**: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest
  tests/test_full_model.py`: 21 passed, no unsafe allocation (`logs/test_full_model_tracker_r2.log`).  The first test
  makes the generator's first `reset()` / `reset_slots()` happen after a capture; negative control (reset warm-up
  removed from `prepare`): the tracker fails on the late-compiled BFP8 `FILL` program buffer
  (`logs/test_reset_after_capture_negative_control.log`).  The test
  order exercises every key with prefill / continuation chunks / slot resets / page-table changes / sampling-mode switches /
  bucket switches between replays.  Acknowledged (exact tensor, `_acknowledge` in `_capture`): the per-bucket model-trace
  output `logits[b]`, written by model trace b and read only by sampling trace b, which is replayed immediately after it
  (and before any other replay or eager op) in every path that replays model trace b; other traces may reuse its address
  as scratch.  In host-sampling compat mode the logits are read right after model trace b, again before anything else.
  No other acknowledgement or allocation scope.  Device-side tensors created after a capture and kept alive across a
  replay would be unsafe — e.g. the outputs of `snapshot_conv_state` (prefix-caching hook): stage 9 must preallocate
  those before capture (the tests take host-side snapshots for that reason).
* Tracker coverage is on the reduced probe (same generator / model code paths as 40 layers, one layer of each kind);
  the all-layer runs ran without the tracker.  Each all-layer run logs Metal's allocator warning "Allocating device
  buffers is potentially unsafe due to the existence of an active trace" once: these are the eager prefill /
  request-boundary input tensors created between replays and freed before the next replay — the pattern the probe
  tracker run covers.
* Benchmark-only second generators (force-argmax in `perf_full_model.py`, the variants in `repro/sampler_ab.py`) are
  built only after the previous generator's last replay, and that generator is never replayed again (perf runs last in
  the driver).

### Host work per step (counters, `tests/perf_full_model.py`)
Per 128-token request (token-out): `model_replays 127, sampling_replays 127, history_replays 127, history_readbacks 1, token_readbacks 1, prefill_sampling_replays 1, prefill_chunks 1, token_refreshes 1, position_refreshes 1, rope_refreshes 1, mask_refreshes 2, history_cursor_refreshes 1`.  i.e. per decode step exactly one model replay + one
sampling replay + one history replay; zero token / position / RoPE / page-table / mask writes, zero syncs, zero
readbacks in the steady state.  Request boundary: 1 position + 1 RoPE + 2 mask refreshes, 1 token write, 1 prefill-
sampling replay + 1 token readback, 1 history readback.  Teacher forcing adds 1 token readback per step (inherent) and a
token write only when the forced token differs from the prediction.  The page table is copied only when it changes
(tested: unchanged -> 0 copies, changed -> 1 copy, output changes; restored -> bitwise previous output).

### Lower-bound accounting (b1 token-out, ms/step)

| item | ms | source |
|---|---|---|
| 40-layer decoder stack (stage-5 Tracy device sum, b1 bucket) | 8.09 | `doc/optimized_multichip_decoder/README.md` |
| - cos/sin once per step instead of per attention layer (9 x 13.4 us) | -0.12 | |
| embedding + entry reshard + cos/sin gathers | 0.022 | probe Tracy (`perf/probe_decode_breakdown.txt`) |
| final norm + LM head (bfp8, 70 MB/chip DRAM read) + logits to DRAM | 0.186 | probe Tracy |
| position advance (2 x `plus_one`) | 0.002 | probe Tracy |
| sampling trace (tail mask 28 us, pad 17 us, top-k 124 us, 2 all-gathers 16 us, tie-break / offsets ~60 us, seed 19 us, sampling 22 us) | 0.288 | probe Tracy |
| **device lower bound** | **8.47** | |
| measured token-out (wall) | 9.43 | |
| measured logits-only model trace (wall) | 9.06 | |

The probe's 3 decoder layers cost 573.2 us device vs 567.7 us from the stage-5 per-layer numbers (decoder carried over
intact).  Wall minus device sum is the in-trace op-to-op dispatch gap (164 us over 171 ops in the probe step; the
40-layer step has ~1,850 ops).  Full-model-only costs to optimize in stage 7: the sampler's 32 x 32768 top-k (124 us:
only b rows are active in bucket b), the vocab tail mask slice / concat and pad (~45 us), `manual_seed` (19 us), the
LM head (DRAM-bound; bfp4 is a datatype-sweep candidate), and op-to-op gaps.  The sampler is ~3.6 % of the step: not
dominant.  Probe `tt-perf-report`: `perf/tracy_probe/{decode,prefill}_perf_report.{txt,csv}` (+ `ops.csv.xz`).

### Prefill
Eager, chunked: chunks of 2048 at 2048-aligned starts, the last chunk padded with `pad_token_id` to the next physical
bucket (32 .. 2048).  Pad rows: their K/V land in the sequence's own blocks past its logical end (or scratch block 0),
causal SDPA keeps valid rows from attending to them, the ShortConv tail selection takes the last valid rows, outputs are
sliced to the logical length (all-logits) or selected by a one-hot `[32, S_pad]` matmul (last row, exact).  TTFT is
host-dispatch bound (eager over 40 layers; probe device time of a 128-token prefill: 4.18 ms for 3 layers + head +
sampling, ~55 ms projected for 40 layers vs 221 ms wall): a traced first-chunk prefill per bucket is the first stage-7
item (the bucketed design makes it an addition, no API change).

### Accuracy and qualitative verdict
* Reference: `bringup/references/lfm2_5_24b_a2b_aime24_chat.refpt` (sha256 3dc10f0e…dbe2), generated fresh by
  `tests/gen_reference.py` (packaged `readiness_check.generate` + a transformers-5 shim: `apply_chat_template(tokenize=True)`
  returns a `BatchEncoding`, which the packaged helper iterated as ids) with metadata sidecar `*.meta.json` (model id,
  pinned revision 1769516a…, tokenizer, chat-template sha256, prompt source aime24 / index 0, gen 100, top-k 100,
  command, versions).  The chat template ends the generation prompt with `<think>`: the 100 reference tokens are
  reasoning text.
* Readiness: table above (`logs/final_r2.json`; identical in `logs/final_all_runs.json` and r1).
* `run_autoregressive` (packaged raw-text prompt, 256 tokens): TT stops at `<|im_end|>` after 174 tokens; the text is a
  coherent continuation of the story, then the model's own `</think>` + a short refusal — fluent English, no repetition
  (adjacent duplication 0.0, trigram loop 0.042; `check_degenerate_output` passes).  First token differs from the
  runner's HF output (" a" vs " something") and the texts diverge from there.  Cause, verified: the checkpoint's
  `generation_config.json` sets `repetition_penalty: 1.05`, which HF `generate` applies even with `do_sample=False`;
  HF's own raw step-0 distribution is " a" 0.544 / " something" 0.374 (bf16 forward, forward-with-mask, use_cache, and
  `generate`'s raw logits all agree; fp32 0.540 / 0.377), i.e. TT's pure greedy picks what HF's logits pick, and HF
  `generate` with `repetition_penalty=1.0` produces " a tiny" = TT's first two tokens (`repro/ar_step0_control.py`,
  `logs/ar_step0_control.log`).
* Qualitative suite (`readiness_check/vllm_prompts.txt`, chat template, 128 tokens; `qualitative/`): HF control = CPU
  bf16 pure greedy (`repetition_penalty=1.0`).  TT greedy vs HF greedy first divergence at tokens 27 / 5 / 102 / 57 / 68
  / 21 (prompts 0-5); all 12 TT outputs (greedy + top-k 20 / top-p 0.9 / temp 0.7) are coherent English reasoning on
  topic (haiku syllable counting, supervised vs unsupervised with examples, story planning, the thermodynamics laws,
  "Bonjour" options for the French translation, iterative Fibonacci), no echo, no doubled tokens, no control-token
  leakage.  Prompt 1 diverges at token 5 by wording ("supervised vs unsupervised learning in simple terms. First person,
  concrete" vs "the difference between supervised and ... The user wants") — a near-tie, both fluent.  Verdict: pass.
* Probe tests vs HF (layers 0, 2, 3 + embedding + norm + head, fp32 CPU): prefill PCC >= 0.997 at every tested length;
  batch decode b2 / b8 / b32: every TT token in HF top-5, top-1 18/18, 72/72, 287/288.

### Batch and context contract
32 fixed physical slots; a request keeps its slot; idle rows `cur_pos = -1` (`slot_active = 0`), bucket = smallest of
1 / 2 / 4 / 8 / 16 / 32 covering the highest active slot (standalone `generate` uses slot 0 by default).  Tested (probe):
slots {0, 1} (bucket 2), {0, 5} (bucket 8, holes), {0, 1, 2, 4, 5, 6, 9, 13} (bucket 16, holes), all 32; mixed prompt lengths; idle slots' ShortConv state bitwise unchanged;
same prompt in slots 0 and 17 -> identical tokens; all-layer token-out timing at b1 / b4 / b8 / b32.
Context: 131072 (HF `max_position_embeddings`), no reduction: all 40 layers + full 32 x 131072 BFP8 cache + traces =
23.22 GiB of 31.58 GiB per chip (`doc/context_contract.json` `full_model`).  Prompt lengths 1 .. 131072 - generation
through the public API; tested non-aligned lengths 1, 31, 33, 65, 77, 300, 2047, 2049, 4100 and 131069 (+ decode to
131071); `generate` beyond the context raises.  Host memory: `return_all_logits` returns fp32 `[B, L, 128000]` (a
131069-token all-logits readback would be 67 GB; the readiness prompts are ~260 tokens).

Serving row semantics (`test_serving_row_semantics`): `tokens` / `start_pos` rows are slots; a `tokens` vector shorter
than 32 keeps the device-resident tokens of active slots it does not cover (one readback; bitwise-checked), and a seed
set for one request keeps the other slots' device-advanced RNG state (read-modify-write).  `generate()` is the
standalone owner of the slot -> block mapping and refuses an externally bound cache (use the low-level API).

### Runtime integrity
* Fallback audit: `decode_forward_device` / the sampling and history traces contain only TTNN device ops; the
  steady-state loop is checked with `ttnn.from_torch` / `to_torch` / `copy_host_to_device_tensor` / `Tensor.cpu`
  patched to raise and `throw_exception_on_fallback` on (`test_determinism_no_fallback[1, 32]`).  Host <-> device
  transfers exist only at request boundaries (positions / tokens / masks / page table / sampling params), in eager
  prefill inputs (token ids, positions, page-table rows, chunk start, ShortConv selections, last-row selector: freed
  before returning), in `prefill_logits` / `return_all_logits` and host-sampling compat (full-logits readbacks,
  counted), and in `reset` (state zeroing on device with `ttnn.fill`; host writes of idle positions / tokens / seeds).
* Host-sampling compatibility mode (`host_sampling=True` on `build_generator` / `generate` / `decode_forward`): model
  trace replay + full-logits readback + host argmax + token write; greedy only; never in measured paths; equal tokens to
  the device path (32 steps, `test_host_sampling_compat`; 131069-token context).
* Determinism: token-out histories and logits bitwise equal across two runs at b1 and b32; slot 0 == slot 17.
* Watcher (`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`; without NOINLINE the fabric ERISC kernel exceeds the
  kernel-config buffer at mesh open): 5 probe tests (split-sampling trace, b8 batch, prefill 77 / 2049, host compat)
  passed, 0 error lines in 30,420 records (`logs/watcher/`).
* Cache ownership: standalone `allocate_cache()` (full cache, fixed slot -> blocks), or `bind_cache(kv, page_table)`
  before the first capture for external owners (`prepare()` zeroes the bound cache once: bind an empty cache); `reset()` zeroes KV and ShortConv state in place (buffers and traces
  kept).

### Limitations / follow-ups
* Model build 400-413 s for 40 layers (host weight conversion of the expert copies); weight caching is a stage-7 item.
* Eager prefill (TTFT host-bound); traced first-chunk prefill per bucket is the first stage-7 item.
* Prefill continuation chunks must start at multiples of 2048 (block aligned); arbitrary 64-aligned starts (prefix
  caching, stage 9) need the extra chunked-SDPA k-chunk variants warmed.
* Prefix-caching hooks exist (`snapshot_conv_state` / `restore_conv_state` / `reset_conv_slots`, KV pages addressed by
  page table); their outputs must be preallocated before capture when kept across replays (see Trace allocation).
* No repetition-penalty / presence-penalty support on device (TTSampling has none); readiness and serving greedy are
  pure argmax.
* Readiness packaged-tool issue (not fixed in the package): `readiness_check.generate` fails with transformers 5
  (`BatchEncoding` from `apply_chat_template`); worked around in `tests/gen_reference.py`.

## Commands (from `tt-metal/`)
```bash
source python_env/bin/activate; export TT_METAL_HOME=$PWD
export PYTHONPATH=$HOME/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PWD
M=models/autoports/liquidai_lfm2_5_24b_a2b
# reference (CPU, ~1 min)
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.gen_reference --hf-model LiquidAI/LFM2.5-24B-A2B \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 --output $PWD/bringup/references/lfm2_5_24b_a2b_aime24_chat.refpt
# all-layer gates + perf + memory + qualitative on one build (~20 min)
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.qualitative --hf          # HF CPU control (once)
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.readiness_driver --runs prefill,tf,perf,memory,qual,ar --json $M/doc/full_model/logs/final_all_runs.json
# probe tests with the trace allocation tracker (~3 min)
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest -q $M/tests/test_full_model.py
# probe tt-perf-report
$M/doc/full_model/repro/tracy_probe.sh $PWD/$M/doc/full_model/perf/tracy_probe
# stage gate
MODEL_DIR=$M bash $HOME/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/06-full-model.check.sh
```
Hardware: QuietBox, 4 x Blackhole p150b (1x4 mesh, FABRIC_1D_RING, router payload 8192 B, l1_small 24576, trace region
256 MiB).  HF `LiquidAI/LFM2.5-24B-A2B` @ 1769516ac588c4c558089c94f27d96b0dc67b183, transformers 5.12.1, torch 2.11.0.
Branch `autoport/lfm2_5_24b_a2b`; base commit 59366b94 (stage 5).
