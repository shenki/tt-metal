# LFM2.5-24B-A2B optimized full model (stage 7) — 1x4 Blackhole p150b, TP = 4

## Headline: before (stage 6) / after (stage 7), all 40 layers, warmed, wall clock, tracking / profiler off

Workload: batch 1, chat-template AIME24 prompt cut / repeated to the prompt length, greedy.  Before =
`doc/full_model/logs/final_r2.json`, after = `logs/final_r2.json` (final code; `tests/perf_full_model.py`).  Harness
change: stage 7 synchronizes the device after each request's `reset()` before starting the TTFT timer (`_fresh`), so a
request does not pay the previous request's whole-cache zeroing (~22 ms; with traced prefill the TTFT(128) read 87.5 ms
without the sync, `logs/allayer_p1_perf.json`); stage 6's eager, host-dispatch-bound prefill hid that work behind
dispatch.

| metric | stage 6 | stage 7 | boundary |
|---|---|---|---|
| **TTFT, prompt 128** | 221.6 ms | **65.4 ms** | `generate` start -> first token on host (prefill + prefill-sampling replay + 128 B readback); request boundary synced first |
| TTFT, prompt 1024 / 2048 / 4100 | (eager, not measured) | 97.5 / 134.9 / 296.6 ms | 1 / 1 / 3 chunks |
| first request of a process (incl. deferred captures) | 446 ms (readiness runner) | 558.6 ms | bucket-1 decode + prefill-bucket captures happen once |
| **token-out decode** | 106.05 t/s/u (9.430 ms) | **106.88 t/s/u (9.356 ms)** | model trace + sampling(+history) trace per step, non-blocking, 1 history readback per request |
| **teacher-forcing decode** | 101.85 t/s/u (9.818 ms) | **102.03 t/s/u (9.801 ms)** | + per-step 128 B token readback + host callback |
| traced decode, logits only | 9.061 ms | 9.146 ms | model trace replays only |
| sampling trace | 0.340 ms | 0.315 ms | |
| token-out b4 / b8 / b32 | 13.83 / 19.26 / 26.24 ms | 13.75 / 19.20 / 26.20 ms | slots 0..b-1, mixed prompt lengths |
| 40-layer model build | 412 s | 81 s (warm weight cache; 409 s cold build that writes it, once per key) | |
| readiness prefill / TF top-1 / 5 / 100 | 98 / 100 / 100 % | 98 / 100 / 100 % | fresh AIME24 chat-template reference (stage 6) |

Free-running (`run_autoregressive`) and greedy qualitative outputs are token-identical to stage 6 (`readiness_autoregressive/`,
`qualitative/tt_greedy.json`); `07-optimized-full-model.check.sh` exit 0 (`logs/check_07.log`).

## What changed

| item | change | effect | evidence |
|---|---|---|---|
| P1 traced prefill | `tt/model.py` `prefill_host_inputs` / `alloc_prefill_inputs` / `load_prefill_inputs` / `prefill_chunk_device`: every chunk-dependent quantity (token ids, positions, page-table row, chunk block row, chunk start, ShortConv carry / tail / keep, last-row selector) is a persistent per-bucket device input refreshed by host copy; `tt/generator.py` `_prefill_chunk` / `_capture_prefill`: one trace per (bucket, first / 2048-aligned continuation, head "last" / "none"), captured on first use, never released before `close()`; the sampled last chunk accumulates its logits into `pf_logits` inside the trace | TTFT(128) 221.6 -> 65.4 ms; probe 128-row prefill op-to-op gaps 28.5 ms -> 0.23 ms | `test_prefill_traced_vs_eager[1, 33, 300, 2049, 4100]` (logits, token, ShortConv state, KV blocks bitwise), `test_prefill_trace_reuse`, `test_prefill_allocation_free`, probe Tracy |
| W1 weight cache | `tt/weights.py::WeightCache`: during the weight construction (globals, RoPE, layers; `stop()` before the final-norm config / decode state) every large, non-zero device-bound `ttnn.from_torch` goes through `ttnn.as_tensor(cache_file_name=...)` (one hook, no decoder edits); mutable state and zero constants are never cached; key = weight-defining code (decoder modules, weights.py, model_config.py, model.py) + checkpoint fingerprint (resolved snapshot dir, shard sizes / mtimes) + policy / plan / fusions / head / mesh / max_seq_len | build 412 -> ~81 s | `tests/test_weight_cache.py`: probe at max_seq_len 131072 built cache-off / cold / warm: all **222** device tensors reachable from the model identical on all chips incl. shape / dtype / layout / memory config (160 cached, the rest bypassed) (`logs/test_weight_cache.log`); all-layer readiness / AR / greedy outputs unchanged.  (The first version of this test walked no tensor — module filter — and passed vacuously; caught in review, fixed.) |
| D1 hoisted bucket slices | `decode_forward_device` slices the b-row page table / positions once per step and hands them to the 10 attention layers (the attention decode already consumed b-row tensors) | 18 slice ops fewer per step at b < 32 | `tests/test_full_model_ab.py` (eager logits bitwise, buckets 1 / 4 / 8 / 32) |
| S2 history in the sampling trace | per bucket a sampling + history-append trace for free-running decode | 3 -> 2 trace replays per step | counters, tracker suite |
| S3 stable local top-k | `sampler._topk_stable = True` (TTSampling's own switch; this call is not on the routed large-index path) | top-k 124.3 -> 99.1 us device | `repro/topk_stable_bench.py` / `logs/topk_stable_bench.log` (99.5 vs 124.2 us, lowest tied index first in both); tie-break pass kept; greedy tokens identical |
| trace region | 256 -> 768 MiB (`open_mesh`, driver `--trace-region-mb`) | room for all prefill keys + decode buckets | 256 MiB overflowed at the b32 capture (first `allayer_p1_perf` attempt); used 323 MB after the perf run |

Unchanged (carried forward from stage 5/6): decoder `MultichipDecoder` + `MC_POLICY` + `DEFAULT_MESH_PLAN` (dtypes,
fidelities, BFP8 paged KV, bf16 `all_reduce_async` on persistent scratch, rejection ledger), the width-sharded L1
inter-layer residual, embedding / final norm / LM head (bfp8 DS 32256 / chip, (64, 1, 3)), canonical TTSampling split
sampling with `tt_out_tok`, on-device `plus_one` position / RoPE advance, changed-only page tables.

## Trace keys and lifetime
* Decode: per bucket b in {1, 2, 4, 8, 16, 32}: model trace, sampling trace, sampling + history trace.  Prefill: per
  (S in 32 .. 2048, continuation, out in {last, none}) — reachable keys: (S, first, last) and (S, cont, last) for every
  S, (2048, first, none) and (2048, cont, none).  Prefill-sampling trace, history-append trace.  All captured lazily on
  first use with program-cache misses forbidden (`set_program_cache_misses_allowed(False)` around every capture: model,
  sampling, sampling + history, prefill-sampling, history-append and prefill traces), at
  most once, released only by `close()` / `release_traces()`.  Trace sizes: prefill 2.1-4.3 MB / bank, decode
  ~2.2 MB / bank (`logs/final_r2.json` `perf.prefill_trace_bytes_per_bank`).
* Prefill program families: the chunk start passed to the decoder is a representative (0 for a first chunk,
  PREFILL_CHUNK for any 2048-aligned continuation); the runtime start is the `cst` input.  The decoder uses the integer
  only for the alignment assert and the SDPA q / k chunk choice, which depends on the start only through divisibility
  while both caps (q 64, k 512) are powers of two dividing 2048 — asserted in `prefill_chunk_device`.  The eager path uses
  the same representative, so the traced-vs-eager test cannot see a wrong family; coverage of continuation correctness is
  `test_prefill_lengths[2049, 4100]` (vs HF) and `test_prefill_continuation_interleaved`.
* Trace-region budget: prefill traces 2.1-4.3 MB / bank each, decode bucket ~2.2 MB / bank (model) + samplers; the full
  reachable set (16 prefill + 6 x 3 decode/sampling + 2) is ~71 MB / bank of the 96 MiB / bank region (768 MiB per chip);
  `final_r2.json` `perf.trace_region_used` after the perf run.
* Program set: `prepare()` compiles every prefill bucket x {first, continuation} with the all-logits and the last-row
  heads, the sampling accumulate, every decode bucket, the samplers, history append, slot / cache reset ops — the
  "none" head variant and the first-chunk last-row head reuse those programs (no compile after a capture).
* Persistent inputs: decode state (stage 6) + per-bucket prefill inputs allocated by `prepare()` before any capture;
  a traced prefill creates no device tensor from the host between replays (`test_prefill_allocation_free`).
  Acknowledged trace output: `logits[b]` only (stage 6 invariant: written by model trace b, read by its sampling trace
  replayed immediately after).  Prefill traces write their logits only into the pre-capture `pf_logits`.
* Tracker: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest tests/test_full_model.py`:
  28 passed (`logs/test_full_model_tracker_r2.log`; first test makes the first reset after a capture; traced prefill of
  every tested length, continuation chunks interleaved with decode replays, batch, determinism with host transfers
  forbidden, full context 131069 + decode).  A negative control (bucket-2048 prefill inputs allocated lazily after
  captures, `logs/test_prefill_inputs_lifetime_negative_control.log`) did not trigger the tracker: the late inputs were
  captured into their own new trace and did not overlap an earlier trace's footprint, so it is inconclusive; the
  guarantee is structural (allocation in `prepare`, KeyError on a missing bucket) and tested directly
  (`test_prefill_allocation_free`).

## Host work per request (token-out, prompt 128 / gen 128, `final_r2.json`)
`model_replays 127, sampling_replays 127, history_appends 127` (one combined trace), `prefill_replays 1,
prefill_input_refreshes 11` (one bucket's inputs), `prefill_sampling_replays 1, token_readbacks 1, history_readbacks 1`,
request boundary `position / rope 1, mask 2, token 1, history_cursor 1`.  Steady state: 2 non-blocking replays per
token, no host writes, syncs or readbacks.

## Roofline / device / end-to-end (b1, `perf_summary.json`, `perf/roofline_b1.json`)
`repro/roofline.py`: bytes read per chip per b1 token from the config and MC_POLICY dtypes (attention q/k/v/o + KV at
position 192, ShortConv in/out, router + 4 active quarter-experts per MoE layer, dense w13/w2, LM head 70.2 MB) = 693 MB
-> **1.35 ms** at 512 GB/s (tt-perf-report's Blackhole DRAM bandwidth).  Device lower bound 8.62 ms (below), end-to-end
token-out 9.35 ms: the step is ~6x above the DRAM roofline — latency / op-count bound (~1,850 device ops per 40-layer
step, 80 all-reduces), not bandwidth bound.

## Lower bound (decoder stack from stage-5 per-layer latencies + measured terminal work)
`LB(b) = 2 L_dense + 28 L_conv_moe + 10 (L_attn - 13.4 - 1.9 us) + T` (stage-5 `tracy_final` per-layer device max over
chips; the attention figures include cos/sin and b-row slices that the full model computes once per step), T = probe
Tracy terminal work (`perf/probe_decode_breakdown.txt`: entry + embedding + cos/sin + hoisted slices 29.6 us, final norm
+ LM head 185.6, position advance 1.8, sampling + history 267.8).

| bucket | stack | terminal | LB | measured token-out | over LB |
|---|---|---|---|---|---|
| 1 | 8.134 ms | 0.485 | 8.619 | 9.347 | +8.5 % |
| 4 | 12.886 | 0.485 | 13.370 | 13.750 | +2.8 % |
| 8 | 18.005 | 0.490 | 18.495 | 19.197 | +3.8 % |
| 32 | 25.995 | 0.490 | 26.485 | 26.202 | -1.1 % (stage-5 per-layer maxima over chips summed: slightly pessimistic) |

Within the goal's 10 % band at every bucket.  The b1 gap (0.73 ms) is in-trace op-to-op dispatch: the probe step has
174.1 us of gaps over 174 ops (decoder layers 88.9 us over 121 ops); the 40-layer step has ~1,850 ops.  Roofline: the
step is op-count / latency bound, not DRAM bound (probe DRAM 92 GB/s average; the dominant single ops are the LM head
168.9 us at ~415 GB/s and 80 all-reduces).  `perf_summary.json` holds the table, workload and named limitations.

## Full-model terminal path audit (probe Tracy before -> after, us device per step)
| op group | before | after | action / evidence |
|---|---|---|---|
| token embedding (1 core, 32 rows) | 10.9 | 10.9 | kept (multi-core RM embedding + tilize would save ~5 us; not pursued) |
| entry reshard, cos/sin (slice, 2 x embedding / tilize-pad / transpose / I2S) | 11.0 | 11.0 | kept (D3 rewrite < 0.1 % of the step: not pursued) |
| b-row page-table / position slices | 20 x ~1 (2 per attention layer) | 2 x ~1 | D1 (bitwise) |
| final norm + reshard + LM head (DS bfp8 [2048, 32256]) + S2I | 186.0 | 185.6 | head config sweep (stage 6); 32768-wide head 202.8 us best (`logs/head_bias_sweep.log`): slower than 178.0 + the 45 us of sampler pad / tail mask it would remove only partly; fused DS-matmul bias for the vocab mask lands on the wrong columns (`repro/ds_matmul_bias_repro.py`, `logs/ds_matmul_bias_repro.log`): not used |
| sampler vocab tail mask (slice, slice, add, concat) + pad to 32768 | 44.8 | 44.9 | kept (see head row) |
| local top-k [32, 32768] k=32 | 124.3 | 99.1 | S3 stable top-k |
| candidate all-gathers (2) | 15.6 | 14.9 | composite `ttnn.all_gather` (no shared semaphores) |
| tie-break / offsets / untilize (~20 ops) | ~60 | ~60 | kept (greedy exactness) |
| manual_seed + sampling | 40.6 | 39.5 | kept; `ManualSeed` 17.7 and the `SamplingDeviceOperation` dispatch gap are op-level (no greedy-only fork of the canonical sampler) |
| history append | separate trace | 9.1 in the sampling trace | S2 |

Sampler benchmark (all 40 layers, same tokens in every variant): split TTSampling 0.315 ms / step, 106.90 t/s/u;
force-argmax 0.924 ms, 100.62 t/s/u (`final_r2.json` `perf.force_argmax`) -> split kept; probe A/B
`doc/full_model/repro/sampler_ab.py` (stage 6): unpadded 375.4, padded 340.2, force-argmax 868.5 us.  No sampler op
dominates: sampling trace 3.4 % of the token-out step.

## $multichip / $optimize checklist (full-model scope)
* Sharding / CCL of the stack: stage-5 tables (`doc/optimized_multichip_decoder/README.md`: residual layout family,
  all-reduce placement, persistent scratch, fused CCL rejections with the 8x2 hang repro) — unchanged.
* Terminal path: embedding replicated (no CCL), final norm on the residual grid (no reshard), vocab-parallel head (no
  logits all-gather; sampler gathers 4 x 32 candidates only), DS head config swept (stage 6) + 32768 / bias variants
  (stage 7), sampler variants measured.
* Matmul program configs: head sweep tables; decoder configs stage 5.  Kernels: sampler top-k stable path selected by
  measurement.  Trace replay: 2 replays / token, non-blocking; prefill traced.  Host boundaries: counters above.
* Tracy / `tt-perf-report`: reduced probe only (layers 0, 2, 3 + full terminal path): `perf/tracy_probe/`
  (`{prefill,decode,decode_b8}_perf_report.{txt,csv}`, `ops_windows.csv.xz` = the three signpost windows,
  `ops_source.txt`), before = `perf/tracy_probe_before/` (stage 6).  Probe: decode b1 1070.4 -> 1049.8 us device
  (not like for like: the stage-7 window replays the sampling + history trace, +9.1 us of history ops (IndexedFill 4.9,
  Copy 3.5, PlusOne 0.7) that stage 6 ran in a separate trace outside its window; like for like ~1040.7 us),
  decode b8 1625.1 us, traced 128-row prefill 4224.6 us device with 225.4 us op-to-op (eager: 28.5 ms).

## Rejected / not pursued (with reasons)
* 32768-wide LM head (pow2 shard, no sampler pad / mask): best 202.8 us vs 178.0 us (only 1-2 readers per bank fit
  128 tiles); net loss.  Fused vocab-mask bias in the DS head: bias misplaced (repro above).  Mask add fallback: 16 us for
  a 28 us saving -> ~12 us, not worth the extra graph.
* Top-k over only the b active rows: the [32, 32768] logits are one tile row; TopK / DS head are width-bound, no gain.
* Sub-devices for terminal overlap: strictly serial chain at b1.
* Removing the greedy tie-break pass (would rely on cross-chip tie order inside `ttnn.sampling`): not verified, kept.
* D2 / D3 (embedding multi-core, cos/sin op rewrite): < 0.1 % each.
* Datatype / fidelity changes (bfp4 head, LoFi, bfp8 CCL payload): stage 8 by decision; fused AGMM / MMRS: stage-5 hang.
* Prefill MoE expert FFN (~874 us per MoE layer at 128 rows): the stage-5 prefill rows P1 / P2
  (`doc/optimized_multichip_decoder/README.md`, ledger: dispatch reshape copies, replicated dispatch glue; work_log O-7)
  hold; the expert FFN op itself is op level.  64-aligned continuation starts (prefix caching): stage 9.

## Correctness and qualitative verdict
* Readiness (final code, all 40 layers, `logs/final_r2.json`): prefill and teacher forcing 98 / 100 / 100 % (bars met).
  The runner's TF TTFT (787 ms) is the first request of the process and includes the deferred captures of the
  256-bucket prefill trace, the decode bucket-1 traces and the samplers.
* `run_autoregressive`: TT tokens identical to stage 6 (174 tokens, stops at `<|im_end|>`; stage-6 HF control analysis
  of the token-0 difference: repetition_penalty 1.05 in the checkpoint's generation_config).  `check_degenerate_output`
  pass.
* Qualitative suite (`qualitative/`, chat template, same HF pure-greedy control as stage 6: `doc/full_model/qualitative/
  hf_greedy.json`, `qualitative_prompt_format.json`): TT greedy outputs token-identical to stage 6 for all 6 prompts
  (first divergence vs HF at 27 / 5 / 102 / 57 / 68 / 21, unchanged).  Top-k (20 / 0.9 / 0.7, seed 1234): 5 of 6
  identical; prompt 1 differs (stable top-k reorders the candidate list the seeded draw indexes) and is a fluent, on-topic
  explanation of supervised vs unsupervised learning.  Control: with `LFM2_STABLE_TOPK=0` all 6 top-k outputs are
  token-identical to stage 6 (`qualitative_unstable_topk_ab/`, `logs/qual_unstable_topk_ab.json`).  Verdict: pass.
* Probe: 28 tests (above); weight cache bitwise; D1 bitwise; traced vs eager prefill bitwise.
* Watcher (`TT_METAL_WATCHER=10`, NOINLINE): 5 probe tests incl. traced prefill 2049 passed, 0 error lines in 7,779
  records (`logs/watcher/`, `logs/watcher_probe.log`).
* Fallback audit: decode / sampling / history / prefill traces contain device ops only; steady-state decode checked with
  host transfers patched to raise (`test_determinism_no_fallback`); traced prefill creates no device tensor from host
  (`test_prefill_allocation_free`); host <-> device work only at request boundaries (counters), in the eager all-logits
  readiness path (`prefill_logits`) and host-compat mode.

## Context / capability
131072 context, 32 slots, no reduction (`doc/context_contract.json` `optimized_full_model`): DRAM 23.22 GiB allocated of
31.08 GiB per chip with the 768 MiB trace region; trace region used 323 MB after the perf run; persistent L1 189,696 B
per core.  Non-aligned prompt lengths 1 .. 131069 through the traced prefill (`test_prefill_lengths`, `test_full_context`).

## Limitations
* TTFT(128) 65 ms vs ~55-60 ms device: per-chunk host refresh of ~11 input tensors + prefill-sampling replay + readback.
* First request of a process pays the deferred captures (~0.3-0.55 s per prefill key, 0.24 s per decode bucket);
  `prepare()` could pre-capture the common keys for serving (stage 9 / 10 decision).
* Continuation prefill starts must be multiples of 2048 (prefix caching at 64-token blocks needs more SDPA variants:
  stage 9).  Prefix-caching hooks unchanged (`snapshot_conv_state` outputs must be preallocated before capture).
* Weight cache: 54 GB on disk for the 40-layer configuration (3.9 GB for the probe) under
  `~/.cache/tt_autoport_weights/lfm2_5_24b_a2b/<key>`; a cold build writes it once (400 s); a change to a weight-defining
  file, the policy / plan / head configuration or `max_seq_len` selects a new key (old keys are not garbage-collected).

## Evidence files
`.log` files are gitignored repo-wide; the stage's logs cited here are force-added to the commit (`git add -f`); the
all-layer driver logs (650-770 KB) are committed as `<name>.log.xz` next to their JSON results.

## Commands (from `tt-metal/`, environment as in `doc/full_model/README.md`)
```bash
M=models/autoports/liquidai_lfm2_5_24b_a2b
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.readiness_driver --runs prefill,tf,memory,qual,ar,perf \
  --json $M/doc/optimized_full_model/logs/final_r2.json        # with LFM2_QUAL_OUT=$PWD/$M/doc/optimized_full_model/qualitative
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest -q $M/tests/test_full_model.py
pytest -q $M/tests/test_weight_cache.py $M/tests/test_full_model_ab.py
$M/doc/optimized_full_model/repro/tracy_probe.sh $PWD/$M/doc/optimized_full_model/perf/tracy_probe
python $M/doc/optimized_full_model/repro/{head_bias_sweep,topk_stable_bench,ds_matmul_bias_repro}.py
MODEL_DIR=$M bash $HOME/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh
```
Results of the runs above: `logs/final_r2.json` (final code); `logs/final_r1.json` is the pre-review run (same numbers within 0.5 %).
A/B switches: `LFM2_EAGER_PREFILL=1`, `LFM2_HOIST_SLICES=0`, `LFM2_STABLE_TOPK=0`, `LFM2_WEIGHT_CACHE=0`.
Hardware: QuietBox 4 x p150b, 1x4 FABRIC_1D_RING, router payload 8192 B, l1_small 24576, trace region 768 MiB.  Base:
stage 6 6227a175.
