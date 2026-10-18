# Stage 10: optimized vLLM serving, LFM2.5-24B-A2B on QuietBox 4x Blackhole p150b (1x4, TP4)

**What is served.** The adapter `tt/generator_vllm.py::TTLfm2MoeForCausalLM`, through `tenstorrent/vllm-tt-plugin`
(local branch `lfm2/text-only-mamba-align-chunked-mm`, `108d500`) on upstream vLLM 0.30.0.

**Configuration.**
- Precision C1 (stage 8).
- `max_model_len` 131072, `max_num_seqs` 32.
- Automatic prefix caching on.
- Serving config selected in this stage: `max_num_batched_tokens` 2048 and tt `decode_interleave_prefill_steps` 1.

Stage 9 (`doc/vllm_integration/`) is the starting point.

**What changed in this stage:**
- device penalties;
- 13x faster host-sampling logits readback;
- a host seed mirror (no blocking readbacks at transitions);
- the decode-interleave setting;
- a 512-token ShortConv prefix-cache snapshot grid built with in-chunk taps;
- accuracy re-measured after the stage-9 all_gather race fix.

**Requests with no sampling parameters run on the host sampler.** With the checkpoint `generation_config` (temperature
0.1, top_k 50, repetition 1.05), a request that sets no sampling parameters is host-sampled, because the device top-k is
capped at 32 (section 2). That traffic gets 34.7 t/s/u single-user and 5.5 t/s/u in a 32-request burst. A client that
sends `top_k <= 32` (with or without penalties) is sampled on device: 85.7 t/s/u single-user with repetition 1.05,
108 t/s/u without penalties.

## Headline: primary single-user serving (final code, runner-launched server)

| metric | stage 9 (before) | **stage 10 (after)** | workload |
|---|---|---|---|
| TTFT (median) | 74.7 ms | **72.0 ms** | 128 in / 128 out / 1 request, concurrency 1, greedy (`--temperature 0.0 --repetition-penalty 1.0`), `ignore_eos` |
| TPOT (mean) | 9.24 ms | **9.25 ms** (P99 9.25) | same |
| ITL (median / P99) | 9.23 / 9.84 ms | **9.24 / 9.52 ms** | same |
| output throughput | 102.5 tok/s | **102.6 tok/s** | same (1 request, includes TTFT) |
| **decode t/s/u (1000 / mean TPOT)** | 108.3 | **108.1** | same |

Evidence:
- after: `readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json`;
- before: `readiness_vllm/superseded_stage9/`;
- runner commands: `logs/runner_final_r2_{serve,sampling,qual_bench}_cmd.txt`. This is the run after review round 1;
  the round-1 runner artifacts are in `readiness_vllm/superseded_s10r1/` with logs `logs/runner_final_*`.

Greedy decode was already at the standalone traced decode speed in stage 9, and it stays there. The stage-10 gains are
in the traffic that stage 9 left on the host, and in mixed-traffic latency (the next table).

**Comparison with the full model (same C1 config, same code).** `logs/perf_full_model_final.json`: standalone traced
token-out batch-1 step 9.26 ms (108.0 t/s/u), TTFT 128 / 1024 / 2048 / 4100 = 64.2 / 95.7 / 132.9 / 296.2 ms.
- Serving decode 9.24 ms/step is the same as standalone traced decode.
- Serving TTFT is 72.0 vs 64.2 ms; the difference is HTTP, scheduling, tokenization and the first-step readback.

### Before / after across sampling modes (same server command line, same workload generator)

The before set ran on committed HEAD `d28b25fb` with the stage-10 changes stashed (`before/`). The after set is the
final code (`after/`). Both used `session_state/scratchpad_tools/bench_set.sh`: `vllm bench serve` random dataset,
`ignore_eos`, single user as 5 sequential requests (n5).

| workload | sampling path before -> after | TTFT p50 before -> after | TPOT mean before -> after | **t/s/u before -> after** |
|---|---|---|---|---|
| greedy 128/128/1 | device -> device | 78.5 -> 73.1 ms | 9.26 -> 9.23 ms | 108.0 -> **108.3** |
| greedy 128/128 n5 | device -> device | 73.5 -> 77.6 ms | 9.25 -> 9.23 ms | 108.1 -> **108.3** |
| top_k 20, temp 0.1, 128/128 n5 | device -> device | 73.7 -> 73.2 ms | 9.24 -> 9.25 ms | 108.2 -> **108.1** |
| **top_k 20, temp 0.1, repetition 1.05**, 128/128 n5 | **host -> device** | 170.7 -> 106.3 ms | 109.0 -> 11.7 ms | 9.2 -> **85.7** |
| **checkpoint defaults** (no params: temp 0.1, top_k 50, rep 1.05), 128/128 n5 | host -> host (top_k > 32) | 168.1 -> 90.7 ms | 106.8 -> 28.9 ms | 9.4 -> **34.7** |
| CI burst greedy 100/100/32 | device -> device | 1430 -> 1432 ms | 28.3 -> 28.7 ms | 35.4 -> 34.9 |
| CI burst top_k 20 + rep 1.05, 100/100/32 | host -> device | 2249 -> 2020 ms | 282.7 -> 29.1 ms | 3.5 -> **34.3** |
| CI burst checkpoint defaults 100/100/32 | host -> host | 2200 -> 2060 ms | 262.0 -> 180.5 ms | 3.8 -> 5.5 |

The `after/` set ran before review round 1 changed the device penalty order (section 1). The round-2 tracked run
(`tracked/`, same QUICK workloads, allocation tracker on) shows the same penalized performance: top_k 20 + rep 1.05
85.5 t/s/u, TPOT 11.70 ms, against 85.7 / 11.67 ms in `after/`. Greedy is 108.3 t/s/u in both.

Summaries: `after/summary.json`, from
`python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.summarize_vllm_bench after --compare before`.

**Which sampler ran, by workload.** The adapter events (`after/events.jsonl.xz`) give a per-workload count of device vs
host steps:
- every step of the greedy, top_k-20 and penalty workloads was device-sampled (penalty workloads: 635 / 635 penalized
  device steps);
- every step of the checkpoint-default workloads was host-sampled;
- 0 stale penalty replays and 0 seed readbacks.

### Mixed traffic, prefix caching, and the CI burst

| metric | stage 9 code (before set) | final | workload |
|---|---|---|---|
| decode pauses > 250 ms | 4 (stage 9 final report, `readiness_vllm/superseded_stage9/mixed_shape`) | **0** | `tests/vllm_mixed_shape_check.py` (unseen lengths 129-131, chunked 3000 / 5000 prompts arriving while 3 requests decode, batch-order repeats, sampling-mode switches, prefix hits) |
| ITL P99 / max | 236 / 343 ms | **158 / 184 ms** | same |
| 12k-token shared prefix: cold owner TTFT | 1.40 s | **1.02 s** | `tests/vllm_prefix_ab.py --concurrent` (owner keeps decoding while 3 hits land) |
| 12k-token shared prefix: 3 hits while owner decodes | 0.64 / 0.65 / 0.65 s | **0.18 / 0.43 / 0.42 s** | same |

**Secondary: CI serving burst (runner; capacity / nightly-parity evidence, not a headline decode number).**

| metric | stage 9 | stage 10 | workload |
|---|---|---|---|
| TTFT median / P99 | 1256.7 / 1930.9 ms | 1321.8 / 1929.1 ms | 100 in / 100 out / 32 requests all admitted at once, greedy + `--repetition-penalty 1.0`, `ignore_eos` |
| TPOT mean / P99 | 28.94 / 31.08 ms | 29.14 / 38.73 ms | same |
| ITL median / P99 | 24.27 / 63.81 ms | 24.26 / 104.16 ms | same |
| output throughput | 723.4 tok/s | 721.8 tok/s | same |
| TPOT-derived t/s/u | 34.6 | 34.3 | same |

The burst's P99 TPOT / ITL rise is the cost of 1 prefill step per decode step. Admitting 32 prompts now interleaves a
decode step after every prefill step, so each prefill step waits behind a decode step. The trade is deliberate (see
the interleave matrix below). Throughput is within 1 %.

## Accuracy and sampling evidence

- **Accuracy after the stage-9 all_gather race fix** (stages 7 and 8 were measured while the race was present).
  `tests/datatype_sweep.py --run-one configs/C1.json` gives, on the main and extended AIME24 references, results
  identical to the three stage-8 C1 runs:
  - traced teacher forcing top-1 / 5 / 100: 0.99 / 1 / 1;
  - prefill: 0.98 / 1 / 1;
  - extended references top-1: 1.00 / 0.98 / 0.99.

  The stage-6 probe suite under the trace allocation tracker passed 28 / 28. Qualitative first divergence vs HF is
  27 / 5 / 102 / 57 / 38 / 21, equal to stage 8. Files: `accuracy/`.
- **Sampling suite (plugin `tests/tt`, full profile, runner).** Round 2 (final code): 71 passed, 4 failed, 46 skipped
  (`readiness_vllm/sampling_tests.log`). Round 1: 72 / 3 / 46 (`readiness_vllm/superseded_s10r1/sampling_tests.log`).
  Stage 9: 74 / 1 / 46.
  - Round 2 failures: `test_bad_words`, `test_mixed_params_batch`, `test_temperature_varied_between_batches`,
    `test_repetition_penalty_mixed_batch`. All are in the flaky classes below and are flaky on both codes. Three of them failed in the stage-9-code control;
    `test_temperature_varied_between_batches` passed there but is a probabilistic host-path test:
    - `test_temperature_varied_between_batches`: host-sampled (inherits top_k 50). It needs the first *character* of
      10 temperature-2.0 completions to vary, and all 10 began with `' '`. It passed in both one-at-a-time runs on
      both codes.
    - `test_repetition_penalty_mixed_batch`: same outlier text `' 0.1\n2e-4 -'` vs majority `' 0.5? ...'` with host
      penalties on stage-9 code (slot 1) and in the host-penalty A/B (slot 5). It's a near-tie of the first generated
      tokens, and which slot hits it depends on admission order.
  - Round 1 failures:
  - `test_bad_words`: unchanged vLLM `bad_words` semantics (stage 9 `analysis/bad_words_failure.md`).
  - `test_seeding_and_variety::test_topk[15]` and `test_tt_penalties::test_repetition_penalty_mixed_batch`: greedy
    near-tie batch variance. **Not a stage-10 regression**; controlled:
    - Both tests compare identical greedy requests that were admitted at different times, so they took different
      decode-bucket histories.
    - Greedy outputs of this model depend on the batch composition: the stage-9 standalone control (no vLLM,
      `doc/vllm_integration/qualitative_parity/standalone_batch_variance.json`) shows solo vs 6-slot batched first
      divergence at tokens 17-109.
    - A dedicated A/B ran the two tests twice on each path (`debug_pen/{device,host}/tests_r{1,2}.log`). With **host**
      penalties (`LFM2_DEVICE_PENALTIES=0`, the stage-9 path) one of them fails in both rounds. With device penalties
      both pass in both rounds, with single excused near-ties.
    - The failure has the same signature as in the runner run: same prompt, same alternative token (`<|end|>` vs
      `Question`), same position (index 1). The stage-9 runner pass was luck of the draw.
  - **Control, one test at a time** (`tests/vllm_suite_sampling_paths.py`; production generation config;
    1 prefill step per decode step; the output of every non-passing test is saved under `failures/`):

    | code | passed / failed / skipped | tests with device-sampled steps |
    |---|---|---|
    | stage-9 code (stage-10 changes stashed) | 72 / 6 / 1 | 0 |
    | final code, run 1 | 71 / 7 / 1 | 17 |
    | final code, run 2 | 76 / 2 / 1 | 17 |

    Artifacts: `suite_attribution/{stage9_code,.,final_r2}`.
    - The flaky set also fails on stage-9 code: `test_mixed_params_batch`, `test_same_seeds_reproduce_across_batches`,
      `test_temperature_varied_in_batch`, `test_repetition_penalty_mixed_batch`. These are host-sampled or near-tie
      batch-variance tests.
    - The two deterministic failures fail on both codes:
      - `test_bad_words`;
      - `test_tile_size`: `cannot import name 'current_platform'`, the plugin import cycle when that file runs alone,
        documented in stage 9.
    - On the final code 17 tests sample on device (stage-9 code: 0). These include every `test_tt_penalties` test and the
      seeded / top-k tests with `top_k <= 32`.
- **Device penalties vs vLLM's host penalties (fp32)** (`penalty_parity/penalty_parity.json`). Each penalized greedy
  request ran twice: plain (device penalties) and with `logprobs=1` (the plugin then samples on the host with vLLM
  penalties over the same logits). Events confirm 1323 host steps and 1323 penalized device steps.
  - Prompts: the 6 qualitative prompts and 1 random-id prompt. Settings: {rep 1.3 + presence 0.5 + frequency 0.3,
    rep 1.05, frequency 1.0}.
  - Final code (vLLM order): in 18 / 21 pairs the 64 greedy tokens are identical. The other 3 are single-penalty
    settings, where the order cannot matter: repetition-only diverges at tokens 35 and 20, frequency-only at token 44.
    Those are bf16 vs fp32 penalty arithmetic on near-ties.
  - Round 1, with the common order: 17 / 21 identical, and the combined setting diverged at token 14 on the random-id
    prompt (`superseded_r1_penalty_parity/`).
  - The penalties change the output in 16 / 21 pairs vs unpenalized greedy.
- **Qualitative** (`readiness_vllm/vllm_qualitative_outputs.json`, chat template, 256 tokens). All 12 completions are
  coherent, on topic, English, with no loops or leakage. The degenerate check is clean (`logs/check_10.log`).
  - Greedy is identical to stage 9 for 3 / 6 prompts. The other 3 diverge 107-845 characters in at near-ties.
  - Reason: the runner's chat requests inherit the checkpoint `repetition_penalty` 1.05. Stage 9 applied it on the host
    in fp32; stage 10 applies it on device in bf16.
- **Serving greedy parity** (pure greedy, `repetition_penalty` 1.0, 128 tokens): see
  `qualitative_parity/vllm_greedy_parity.json`, which compares against the standalone TT run and HF.
- **Gate:** `10-optimized-vllm.check.sh` exits 0 (`logs/check_10.log`): no degenerate output, and the context
  contract holds (131072 = HF context).

## What was done

### 1. Device penalties (presence / frequency / repetition)

**Problem.** In stage 9 every request with a penalty was host-sampled (`supports_device_penalties` False): 9.2 t/s/u.

**Implementation** (`tt/generator.py`; the plugin contract is unchanged). The model reuses
`models/common/sampling/tt_penalties.py::TTPenalties`:
- vocab-sharded prompt masks, output masks and counts as persistent device buffers (`[32, 32256]` int32 per chip);
- replicated gathered counts;
- but not its `apply_penalties` op sequence. That function subtracts presence and frequency first and tests the logit
  sign afterwards; vLLM (`vllm/model_executor/layers/utils.py::apply_penalties`) applies the repetition scaling first,
  on the raw logit sign, then subtracts `frequency * counts` and `presence * mask`. For a token that is in the prompt or
  output and was generated before, the two differ by `(f * count + p)(1 - 1/r)` logit units.
  `Lfm2Generator._apply_penalties` runs the same bf16 ops in vLLM's order. It was found by stage review round 1; the
  first implementation used the common order, and its test reference mirrored that order.

**Traces.** Per decode bucket there is one more sampling trace, `pen_traces[b]`. It applies the penalties to the
bucket's logits, samples, and counts the sampled tokens (`update_output_tokens` on a clone of the persistent token
buffer, which that function deallocates). A penalized prefill sampler `pf_sample_pen_trace` does the same without the
count. 7 extra traces in all, captured with the others during warm-up. Decode replays the penalized trace only when an
active slot has non-neutral penalties. Neutral rows are exact no-ops: the reduced-model test shows they are bitwise
equal to plain greedy.

**State commands (contract v1).**
- The plugin sends `prompt_tokens` / `output_tokens` (`[32, L]`, -1 padded) on every penalized decode step. The adapter
  uses them only on `reset_sampling_state` steps. There it rebuilds every row's masks and counts from host histograms
  (host, because a device `scatter_add` races on repeated ids). On all other steps the device counts tokens itself.
- At prefill the prompt mask of each penalized slot is built from the full prompt, and its output counts are zeroed.
  The prefill's first token is counted by the rebuild at the next decode step. That step is always a transition: the
  plugin invalidates the decode chain after any prefill (`async_decode.py: note_prefill_submitted`).
- A `pen_stale` flag counts any penalized replay without a rebuild since a prefill, slot move or host-sampled step.
  It is 0 in every run (mixed-shape check, penalty parity, benchmark sets).

**Host cost of a rebuild.** Measured in `logs/bench_penalty_resets.log`:
- `TTPenalties`' mesh-mapper `from_torch` cost 6-10 ms per buffer. The generator now builds the per-chip tile shards
  itself (`ttnn.from_host_shards`, about 1.25 ms per shard).
- The replicated buffer is tiled once.
- The prompt-mask upload is skipped when the mask is unchanged; the generator keeps a host mirror that always equals
  the device buffer.
- Result: penalized TTFT went 160 -> 106 ms, against 73 ms greedy.

**Tests** (`tests/test_generator_vllm.py`):
- `test_penalty_op_matches_vllm_order`: logit-level check of the device op on a known state (prompt and output
  tokens with repeats; presence 0.5, frequency 1.0, repetition 1.3). The device op equals vLLM's order within bf16
  rounding, and on the tokens in both masks it is more than 0.5 logit units away from the common order.
- `test_device_penalties_match_reference`: 24 steps against `_vllm_penalties` (fp32, vLLM's order, written in the
  test) on host logits teacher-forced along the device tokens. All tokens are within one bf16 ulp of the reference maximum; at least
  21 / 24 match the exact argmax.
- In the same test: a mid-stream history rebuild leaves the stream bitwise unchanged, so rebuild equals on-device
  counting; a neutral row next to a penalized row equals plain greedy; and the penalties do change the output.

**Measured.** top_k 20 + rep 1.05 single user goes 9.2 -> 85.7 t/s/u (TPOT 11.7 ms; the penalty ops plus the
non-overlapped plugin path cost 2.4 ms per step). CI burst 3.5 -> 34.3 t/s/u.

### 2. Device top-k stays capped at 32 (user decision; evidence)

`TTSampling(max_top_k=64)` builds and replays, but it **cannot** sample from more than 32 candidates.
- `ttnn.sampling`'s compute kernel hard-codes its final top-k to one tile: `nearest32_K = 32`, `logk = 5` in
  `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp:459-460`. The bitwise merge keeps
  "the larger 32 values".
- Evidence: with `max_top_k=64`, `top_k` 50 / 64 at temperature 200 drew no token above rank 29 in 31 draws
  (`logs/topk64_capped_at_32_evidence.log`, `test_device_topk_above_32`).
- So declaring `max_device_top_k` 64 would silently turn top_k 50 into top_k 32.

The user chose exact semantics: the cap stays 32 (`LFM2_MAX_DEVICE_TOP_K`, default 32). top_k 33-64 requests,
including the checkpoint defaults (top_k 50), stay host-sampled. Extending the kernel to 64 candidates (top-k merge,
softmax / top-p over 2 tiles) was rejected for this stage as kernel engineering. Consequence and its cost: checkpoint-default traffic runs at 34.7
t/s/u single-user (host sampler ~9 ms + readback 6.6 ms + plugin per step) instead of ~86.

### 3. Host-sampling logits readback 13x faster

**Problem.** In stage 9 every host-sampled step cost about 107 ms. The adapter's own decode call took 1.1 ms and
vLLM's CPU sampler about 9 ms (`tests/bench_host_sampler.py`, `logs/bench_host_sampler.log`: 128000 vocab, 1 row:
6-9 ms over two runs; 32 rows: 155-165 ms).

**Root cause.** `ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor)` on the `[32, 4 x 32256]` bf16 tile logits takes
**83.9 ms**. A per-device `to_torch` plus `torch.cat` is bit-identical and takes **6.6 ms**
(`tests/bench_logits_readback.py`, `logs/bench_logits_readback.log`).

**Fix.** `Lfm2Generator._logits_to_host` is used for decode host-logits steps, the prefill host-logits buffer and
readiness all-logits. The adapter's per-step token readback now reads chip 0's replicated copy.

**Effect.**
- Checkpoint defaults 9.4 -> 34.7 t/s/u, TTFT 168 -> 91 ms.
- Intermediate 2048-token prefill chunks, which are host-logits steps, went from about 230 ms to about 142 ms per
  step. That shortens every decode pause.

A device untilize + slice would save another 4 ms (2.5 ms measured). It was not adopted: it adds device ops between
trace replays on the host path.

### 4. Transition-step readbacks removed (host seed mirror)

**Problem.** Stage 9 read the device RNG seeds back (blocking) whenever `set_row_seeds` or `permute_slots` changed some
rows.

**Fix.** `Lfm2Generator.seed_host` mirrors the device seeds. Every sampler replay `plus_one`s all rows, mirrored by
`_advance_seeds()` at each replay site. Host-logits steps do not replay the sampler and do not advance the mirror.

**Evidence.**
- `test_seed_mirror_matches_device`: after prefill, decode, a slot remap and host steps, one verification read equals
  the mirror, with 0 readbacks during the sequence.
- `seed_readbacks` = 0 in every server run.
- The remaining token readback is the prefill's sampled first token, which must go to the host. A slot remap arrives
  with `reload_inputs` on device-sampled steps, so no token readback happens on remap.

### 5. Decode pauses: interleave and batched-token budget (matrix)

Three configurations, same code, QUICK set (`matrix/m{1,2,3}_*`):

| config | pauses > 250 ms | ITL max | mixed TTFT p50 | CI 100/100/32 t/s/u (ITL P99) | 12k prefix cold owner |
|---|---|---|---|---|---|
| m1: budget 2048, 2 prefill steps per decode step (plugin default) | 3 | 294 ms | 144 ms | 35.4 (67 ms) | 0.99 s |
| **m2: budget 2048, 1 prefill step per decode step** | **0** | **183 ms** | **109 ms** | 34.8 (109 ms) | 1.01 s |
| m3: budget 4096, 1 prefill step per decode step | 3 | 294 ms | 107 ms | 40.1 (31 ms) | 0.95 s |

**m2 is selected:** the target is a few interactive users, and it is the only configuration without decode pauses.

m3 was rejected: its 4096-token steps (two 2048 chunks, about 290 ms) bring the pauses back. The burst throughput it
wins is a secondary metric.

A 1024-token budget was not measured. It would need `prefill_chunk_alignment` 1024 and twice as many prefill steps for
long prompts, and m2 already removes the pauses.

### 6. 512-token ShortConv snapshot grid (user decision: build the true grid)

**Problem.** In stage 9 a prefix hit at P recomputed `[Q, P)` from the last 2048-aligned snapshot Q (up to 2047
tokens).

**Implementation.**
- Chunk starts may now be any multiple of `CHUNK_ALIGN` = 512 (`tt/model.py`). The SDPA q / k chunk caps (256 / 512)
  divide 512, so every aligned start selects the programs captured for the representative start, and no new trace keys
  appear.
- Every prefill chunk of 1024 or 2048 rows carries two runtime selections `tap0` / `tap1`: one-hot rows at extended
  rows `32 + b - 2` and `32 + b - 1` for each inner boundary b (`tap_boundaries`).
- Every `FusedShortConv` copies those state rows into per-layer persistent tap buffers (`tap_buf`). These are the same
  one-hot matmul as the tail selection, 2 per ShortConv layer per chunk.
- After the chunk the adapter copies each tap row into a snapshot-pool entry (`snapshot_taps_into`, an exact
  fp32-accumulated one-hot matmul) keyed by the 512-block hash chain.
- The walk-down now steps by 512. The pool grew to 256 entries (0.49 GiB per chip).
- Taps exist only when the generator has a snapshot pool (serving). Standalone programs are unchanged:
  `perf_full_model_final` equals stage 9.
- The `model.py` and decoder edits changed the weight-cache key: `a59f9200eb71ecb83b9b` for the evidence runs, then
  `150065a5ca55fabe41cc` after pre-commit formatting. The 40-layer adapter suite passes on the formatted code
  (`logs/adapter_tests_full_r5_postformat.log`).

**Tests** (40 layers, `logs/adapter_tests_full_r4.log`):
- `test_tap_snapshot_equals_chunk_end_state`: the state tapped at 3584 inside chunk [2048, 4096) is **bitwise** the end
  state of a chunk [2048, 3584) of the same bucket.
- `test_prefix_hit_and_evicted_walkdown`: with the 4096 snapshot evicted, the hit walks down to the 3584 tap,
  recomputes 512 tokens, and is **bitwise** equal to the same prompt prefilled with the same chunk split.
- `test_prefix_hit_on_512_grid`: a hit at P = 2624 (neither a chunk end nor on the old grid) restores the 2560 tap and
  recomputes 64 tokens from a 512-aligned continuation start. It matches B from scratch (top-1, PCC > 0.999); no
  bitwise control exists, because A's K/V of [2048, 2560) come from a 2048-row chunk. A's blocks are untouched.
- Two-level eviction (all grid points evicted) and the non-mutation tests pass unchanged.

**Served** (tracked mixed-shape run):
- continuation starts 64 / 128 / 2048 / 2624 / 2688 / 4096 / 4160;
- recompute starts 0 / 2048 / 2560 / 4096;
- 13 tap snapshots stored; 0 recaptures.

**Effect.** 12k-token shared prefix while the owner decodes: hits 0.64 s -> 0.18-0.43 s. This includes the fast
readback and the interleave change.

## Trace, async and fallback evidence (final code)

- **Trace reuse under the allocation tracker.** `tracked/`: server with `TT_METAL_TRACE_ALLOC_TRACKING=1
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; the adapter logs `trace allocation tracking ON`. Extended mixed-shape
  check:
  - pass, 0 recaptures. Warm-up captured 27 decode-side traces (6 buckets x {model, sampling, sampling + history
    append, penalized sampling} + prefill sampler, history append and penalized prefill sampler) and 16 prefill
    traces;
  - greedy determinism across batch compositions, including the penalized greedy request and the 512-grid hit;
  - 110 penalized decode steps, 0 stale penalty replays, 0 seed readbacks;
  - 0 pauses > 250 ms;
  - new allocations: tap buffers, tap inputs and penalty buffers are all allocated before the first capture
    (`prepare()` / generator `__init__`).
- **Allocator warning.** Every non-tracked server log, including stage 9's, prints once after `prepare()`:
  `Allocating device buffers is potentially unsafe due to the existence of an active trace` (`allocator.cpp:140`). The
  control is the `tracked/` run: the allocation tracker was on, program-cache allocations were included, and the
  between-replay allocations (penalty slot resets, tap-snapshot temporaries) were exercised. It found 0 unsafe
  allocations and 0 recaptures.
- **Async decode** is unchanged from stage 9:
  - `decode_forward(read_from_device=False)` returns the device token buffer;
  - `read_decode_output(async_read=True)` = `cpu(blocking=False)` + `record_event`;
  - `execute_trace(..., blocking=False)` for model, sampling and penalized-sampling traces;
  - `supports_async_decode` True and exercised (plugin resident decode modes in `after/mixed_shape`).
  - Penalized steps use the plugin's async submit + read but not its overlapped fast path: the plugin excludes steps
    carrying histories (`async_decode.py:624-628`). That costs part of the 2.4 ms per penalized step. A plugin patch was
    not pursued (see Limitations).
- **Persistent inputs.**
  - Token, position, RoPE, page-table, mask, sampler (k / p / temp, seeds) and penalty (params, masks, counts) inputs
    are persistent device tensors.
  - They are refreshed only on reload / reset / remap commands (event counters).
  - Stale-input tests still pass: `test_stale_inputs_resident_vs_reload`, `test_page_growth_page_only_refresh`,
    `test_slot_remap_swap`, `test_remap_exact_move`.
- **No host fallback on device-sampled steps.** No host argmax, no full-logits readback, no eager sampling. Host-logits
  steps are exactly the plugin's host-sampling steps:
  - logprobs / min_p / bad_words / structured output;
  - top_k > 32;
  - intermediate prefill chunks.

  The attribution counters show 0 host steps in the greedy, top_k-20 and penalty workloads.
- **Cleanup.** `pgrep` showed no vLLM / EngineCore processes after every run. One exception: the runner's held server
  did not exit within 180 s of SIGINT. It was stopped by exact-PID SIGINT to the API server and then SIGTERM to the
  runner (`logs/runner_final_driver.log`).

## Rejected / not done

- **Device top-k 64** (section 2): kernel limit; the user kept the exact 32 cap.
- **Budget 4096** (section 5): pauses return.
- **Device untilize + slice for host logits** (section 3): extra 4 ms not worth device ops on the host path.
- **Plugin patch to overlap penalized steps.** It would require changing the plugin's overlap-eligibility rule for
  steps with histories. The device-side cost (11.7 vs 9.2 ms) was judged acceptable for few users. It remains a
  candidate.
- **Single-slot penalty uploads.** The full `[32, V]` prompt-mask upload (about 5 ms) is kept for simplicity; skipped
  when unchanged.

## Limitations

- Checkpoint-default requests (top_k 50) are host-sampled: 34.7 t/s/u single user, 5.5 t/s/u in a 32-request burst
  (the CPU sampler takes 155-165 ms for 32 rows). Clients that send `top_k <= 32` get device sampling.
- Device penalties follow vLLM's order but use bf16 arithmetic. For example, repetition 1.05 is applied as a bf16
  multiply / divide (effective 1.047 / 1.049). Near-tie greedy choices can therefore differ from vLLM's fp32 host
  penalties (3 / 21 pairs over 64 tokens, section "Accuracy and sampling evidence").
- Greedy batch variance (pre-existing, stage 9 control) makes two plugin determinism tests flaky.
- Admitting a penalized request costs about 30-45 ms for the history rebuild at the next decode transition, paid by all
  running rows on that step.

## Artifacts

| path | content |
|---|---|
| `before/`, `after/`, `after_v1/`, `after_v2/` | benchmark sets (vllm bench JSON + logs, adapter events, mixed shape, prefix A/B); `after/summary.json` |
| `matrix/` | interleave / budget matrix |
| `tracked/` | allocation-tracker server: QUICK set + extended mixed-shape check |
| `penalty_parity/` | device vs host penalties |
| `debug_pen/` | A/B of the two flaky plugin tests, device vs host penalties |
| `suite_attribution/` | per-test sampling-path attribution of the plugin suite |
| `qualitative_parity/` | serving greedy parity vs standalone TT and HF |
| `accuracy/` | post-fix accuracy (AIME24 refs, probe suite, qualitative) |
| `logs/` | adapter tests (reduced r1-r9, 40-layer r1-r6; final: `adapter_tests_reduced_r9`, `adapter_tests_full_r6`), microbenchmarks, runner logs and commands (`runner_final_r2_*` final), `check_10.log`, `perf_full_model_final.{json,log}` |
| `scripts/` | the shell launchers used for the non-runner evidence (server, benchmark set, tracked run, runner flow, reruns, A/B) |
| `readiness_vllm/` | final runner artifacts (stage-9 ones in `superseded_stage9/`) |
| `doc/context_contract.json` | `optimized_vllm` block |

Commands: `work_log.md`. Launchers (copied from `session_state/scratchpad_tools/`): `scripts/{serve_lfm2,run_server_bench,bench_set,runner_final,run_pen_tests,run_suite_attr,rerun_after_review1}.sh`.
