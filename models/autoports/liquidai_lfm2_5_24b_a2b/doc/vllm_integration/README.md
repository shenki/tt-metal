# Stage 9 — vLLM integration: LFM2.5-24B-A2B on QuietBox 4x Blackhole p150b (1x4, TP4)

Adapter `tt/generator_vllm.py::TTLfm2MoeForCausalLM`, served through the standalone `tenstorrent/vllm-tt-plugin` and
upstream vLLM 0.30.0, selected precision **C1** (stage 8), `max_model_len` 131072, `max_num_seqs` 32, automatic
prefix caching **on**.  This README describes the **final code** after stage-review rounds 1 and 2 (prefix-hit
recompute into a full page-table row of adapter scratch blocks; `top_k > 32` rejected in the device path) and after the
root-caused tt-metal fix of a multicast all_gather completion race that made prefill intermittently nondeterministic
(below).  All headline artifacts were regenerated on this code.

## Headline (final code, one runner-launched server)

| | result | workload / evidence |
|---|---|---|
| **sampling suite (plugin `tests/tt`, full profile)** | **74 passed, 1 failed (`test_bad_words`, vLLM bad_words semantics, see below), 46 skipped** (41 speculative-decode tests: server not speculating; 1 spec-artifacts; 3 chunked-prefill tests skipped by the runner, run separately: **3 passed**; 1 all-vocab logprobs: plugin clamps `max_logprobs` to 20) | `readiness_vllm/sampling_tests.log`, `logs/runner_final3_sampling{_cmd.txt,.log}`; skip reasons `logs/sampling_full_rs.log`; `logs/sampling_chunked_prefill.log` |
| **which sampler the suite exercised** | with the checkpoint `generation_config` the whole suite is host-sampled (0 device steps out of 262 prefill + 3149 decode steps); per-test attribution with `--generation-config vllm`: **16 tests had device-sampled steps** (table below) | `logs/events_full_r2.jsonl`; `analysis/suite_sampling_paths_genconfig_vllm.json` |
| **qualitative verdict** | **pass**: greedy and sampled answers to all 6 chat prompts are coherent, on topic, English, no repetition / gibberish / leakage; the 6 greedy completions are identical to rounds 1 and 2; degenerate check: *No degenerate output detected* | `readiness_vllm/vllm_qualitative_outputs.json` (chat template, `/v1/chat/completions`, 256 tokens), `logs/degenerate_check.log` |
| **serving parity vs standalone TT and HF (final code)** | vLLM solo greedy == standalone TT **and** vLLM 6-concurrent greedy == standalone 6-slot batched control, **token for token** (128 tokens, all 6 prompts); first divergence vs HF greedy **27 / 5 / 102 / 57 / 38 / 21** (= stage 8) | `qualitative_parity/final/vllm_greedy_parity.json`, `qualitative_parity/standalone_batch_variance.json` |
| **TTFT (median)** | **74.7 ms** | primary single-user: 128 in / 128 out / 1 request, concurrency 1, greedy (`--temperature 0.0 --repetition-penalty 1.0`), `ignore_eos` |
| **TPOT (mean)** | **9.24 ms** (P99 9.24 ms) | same |
| **ITL (median)** | **9.23 ms** (P99 9.84 ms) | same |
| **output throughput** | **102.5 tok/s** | same (1 request, includes TTFT) |
| **decode t/s/u (1000 / mean TPOT)** | **108.3 t/s/u** | same; `readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json` |

Serving decode vs the full-model references (same selected config C1): standalone post-selection token-out 107.97 t/s/u
(stage 8, `doc/datatype_sweep/logs/final_selected.json`) / 107.96 t/s/u after the all_gather fix
(`logs/perf_after_allgather_fix.json`) and traced teacher forcing 103.70 t/s/u (lower bound; selection metric).
Serving matches the standalone token-out path (108.3 vs 107.96, within run-to-run noise): no avoidable vLLM-specific
decode overhead remains in steady decode (resident token / position feedback, non-blocking trace replay, one 128-byte
token readback per step, no page-table copy on unchanged steps).  The benchmark server ran without adapter event
logging, so device sampling of the benchmark steps is **inferred**: the requests carry `temperature 0` +
`repetition_penalty 1.0` (no host-only parameter, so the plugin selects device sampling) and the TPOT matches the
standalone device token-out path (a host-sampled step adds an 8 MB logits readback plus the host sampler, which is not
consistent with that match).  No per-step sampler log exists for these two runs.

### Secondary: CI serving burst (vLLM-nightly shape; not a headline decode number)

| metric | value | workload |
|---|---|---|
| TTFT median / P99 | 1256.7 ms / 1930.9 ms | 100 in / 100 out / 32 requests, no max concurrency (all 32 admitted at once), greedy + `--repetition-penalty 1.0`, `ignore_eos` |
| TPOT mean / P99 | 28.94 ms / 31.08 ms | same |
| ITL median / P99 | 24.27 ms / 63.81 ms | same |
| output throughput | 723.4 tok/s (32/32 completed) | same |
| TPOT-derived t/s/u | 34.6 | same; burst admission + 2048-token chunked prefill inflate TPOT |

Files: `readiness_vllm/vllm_ci_serving_benchmark.json`, raw `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.log`.

### How the final evidence was produced

One server launched by the runner on the final code (`run_vllm_server --stages serve`, held open;
`logs/runner_final3_serve_cmd.txt`, `logs/runner_final3_serve.log`, server log `readiness_vllm/server.log`: started
16:29, `trace allocation tracking off`, `KV pool 65568 blocks x 64 tokens (+2048 adapter scratch)`), then the runner
attached with `--server-url` for `--stages sampling` (full profile, `logs/runner_final3_sampling_cmd.txt`) and
`--stages qualitative,benchmark` (`logs/runner_final3_qual_bench_cmd.txt`).  The full-flow mode would stop after the
single sampling failure and skip qualitative + benchmark, hence the attached stages.  `degenerate_check.log` and
`check_09.log` (exit 0) were regenerated afterwards.  Superseded runner artifacts: `readiness_vllm/superseded_r1/`
(round 1), `readiness_vllm/superseded_r2/` (round 2), `logs/runner_final{,2}_*`, `logs/runner_qual_bench*`,
`logs/server_final_bench.log`.

## Serving stack

| item | value |
|---|---|
| plugin | `tenstorrent/vllm-tt-plugin` `main` @ **6afed54** (2026-10-16, #219), checkout `/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin` (detached), editable install `vllm-tt-plugin 0.1.0`, **plus one local patch** (below) |
| upstream vLLM | **0.30.0** (`vllm==0.30.0+empty`), the version the pinned plugin's `docs/install-vllm-tt.sh` recommends (`VLLM_TARGET_DEVICE=empty`, `--no-binary` sdist, `requirements/common.txt` of v0.30.0 + `docs/vllm-overrides.txt`, torchvision 0.26.0 CPU, numba 0.61.2) |
| interpreter | tt-metal `python_env` (Python 3.12.14), torch 2.11.0+cpu, transformers 5.12.1, numpy 2.3.5 (unchanged) |
| install | `source docs/install-vllm-tt.sh` unchanged; log + rollback freeze in `session_state/stage9/` |
| packages changed | huggingface-hub 1.16.1 -> 1.33.0, pydantic 2.9.2 -> 2.14.1 (+ core etc.), click 8.5.0, lark 1.2.2, opencv-headless 4.11.0.86, sse-starlette 3.5.0, numba 0.61.2 (from 0.68.0) |
| `uv pip check` | 2 incompatibilities, both outside the model path: vllm wants opencv >= 4.13 (the plugin override pins 4.11), fiftyone (tt-metal dev dep, unused) wants sse-starlette < 1 |
| tt-metal (non-autoport) | `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp`: `axis_load_balance = false` (all_gather race fix, rebuilt `build_Release`); stage-owned, not upstreamed |

**Local plugin patch** (`src/vllm_tt_plugin/platform.py` `_apply_chunked_prefill_policy`, +2 host tests in
`tests/test_chunked_prefill_policy.py`; 13 passed, `pre-commit` clean; not pushed): upstream vLLM classifies the
checkpoint (`Lfm2MoeForCausalLM`, `IsHybrid`) as hybrid, so with prefix caching it sets `mamba_cache_mode="align"` and
`VllmConfig` asserts that chunked MM input stays allowed; the plugin set `disable_chunked_mm_input=True` for every
chunked-prefill model, which made EngineCore initialization fail (`logs/server_reduced_r1.log`).  The flag only
governs multimodal items, so the patch leaves it unset for text-only models under `align` mode (gated on the resolved
cache mode, not a model allowlist).  The adapter emits no MambaSpec, so vLLM's align-mode state logic is inert.

## Registration

`TT_MODEL_CLASS_OVERRIDES="TTLfm2MoeForCausalLM=models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm:TTLfm2MoeForCausalLM"`
(HF architecture `Lfm2MoeForCausalLM`; the plugin prefixes `TT`; there is no built-in TT Lfm2 class, so a missing
override cannot silently select another implementation).  Every process logs `Applied TT_MODEL_CLASS_OVERRIDES:
TTLfm2MoeForCausalLM -> models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm:TTLfm2MoeForCausalLM`, EngineCore
included, followed by the adapter's `TTLfm2MoeForCausalLM: precision C1 (.../selected_precision_config.json), layers
40, max_model_len 131072` (`readiness_vllm/server.log`).

## Adapter

`TTLfm2MoeForCausalLM` composes `Lfm2Generator` (`tt/generator.py`); it only translates plugin keyword arguments.

| plugin call | delegates to |
|---|---|
| `initialize_vllm_model(hf_config, mesh, 32, max_seq_len)` | `Lfm2Model(mesh)` with the default precision `tt/precision.py::resolve()` -> `doc/datatype_sweep/selected_precision_config.json` (C1: decode all-reduce payload BFP8, KV BFP8, weights BFP8 HiFi2, head BFP8); model always built for the trained 131072 context |
| `get_max_tokens_all_users` | `max_num_seqs x max_model_len` = 32 x 131072 (plugin adds 32 blocks headroom -> 65568 blocks) |
| `allocate_kv_cache(shape, dtype, 10)` | `Lfm2Model.allocate_kv_cache(num_blocks + 2048)` + `Lfm2Generator.bind_cache`: vLLM addresses blocks `[0, 65568)`; the 2048 blocks after them (one full page-table row) are adapter-owned scratch, the prefix-hit recompute target; per chip `[67616, 2, 64, 64]` BFP8 per attention layer (vLLM's dtype only drives its byte accounting); `kv_cache` identity checked on every call |
| `warmup_model_prefill/decode(enable_trace=False)` | `Lfm2Generator.prepare()` (phase 1: compiles every prefill bucket / continuation / head / sampler / decode bucket / snapshot / permute op; allocates the snapshot pool) |
| `warmup_model_prefill(enable_trace=True)` | `_capture_prefill` of the 16 reachable prefill keys (7 buckets x {first, continuation} with the sampled last-row head, plus the full 2048 chunk without head) |
| `warmup_model_decode(enable_trace=True)` | `_trace(b)` for buckets 1, 2, 4, 8, 16, 32: model + sampling + sampling-with-history traces, plus prefill sampler and history append = 20 traces |
| `prefill_forward` | `Lfm2Generator.prefill_forward(..., slots=empty_slots, start_pos, sample / host_logits, on_chunk)` (traced 2048-chunked prefill) |
| `decode_forward` | `permute_slots` (slot_remap) -> `set_sampling_params` / `set_row_seeds` -> `Lfm2Generator.decode_forward(..., read_from_device=False, active_slots=B, return_logits=host)` |
| `read_decode_output(async_read=True)` | `tensor.cpu(blocking=False)` + `ttnn.record_event` (persistent token buffer; read enqueued before the next submission) |
| `process_decode_output_host` | `[B, 1]` token ids, or `[B, 1, 128000]` fp32 logits (host steps) |
| `release_request(slot)` | `Lfm2Generator.reset_slots` (idle position: no KV writes through a stale page-table row; zero ShortConv rows) |
| `release_persistent_capture` | `Lfm2Generator.release_traces` (all 36 traces; logged as `release_traces 20 16` at every shutdown) |

Generator / model changes (outside the traced graphs): `prefill_forward(host_logits=, on_chunk=)`,
`decode_forward(return_logits=)`, `set_row_seeds`, `permute_slots`; `Lfm2Model.alloc_conv_snapshot_pool`,
`snapshot_conv_state_into`, `restore_conv_state_from`, `permute_conv_slots` (one-hot fp32-accumulated matmul, exact).

### Capability declarations

| key | value | evidence |
|---|---|---|
| `supports_sample_on_device` | True | device prefill + decode sampling for greedy / top-k <= 32 requests without host-only parameters (mixed-shape decode modes; 16 suite tests with device steps) |
| `max_device_top_k` | 32 | `TTSampling` local top-32 per chip; a device-sampling row with `top_k` outside 1..32 raises (was a silent clamp; the plugin already routes such rows to the host) |
| `supports_device_penalties` | False | penalties are host-sampled (stage 10 adds device penalties) |
| `supports_chunked_prefill` | True | 2048-token chunks, continuation starts 2048 / 4096 under concurrent decode (mixed-shape), `test_chunked_prefill.py` 3 passed |
| `supports_prefix_caching` | True | A/B identical outputs (sequential and concurrent cold-owner), hit-correctness / shared-block / two-level-eviction tests, mixed-shape prefix hits |
| `supports_async_decode` | True | contract v1, split submit / readback, resident token + position (below) |
| `fabric_config` | `{config: FABRIC_1D_RING, router_config: payload 8192}` | |
| `decode_input_update_contract` | 1 | legacy `reset_batch` rejected |
| `tt_supported_decode_batch_sizes` | (1, 2, 4, 8, 16, 32) | = captured decode buckets |

tt-config: `{"sample_on_device_mode": "all", "trace_region_size": 805306368, "l1_small_size": 24576,
"prefill_chunk_alignment": 2048}`; `--block_size 64 --max_num_seqs 32 --max_model_len 131072`; chunked prefill with
`max_num_batched_tokens` 2048 (vLLM default; user decision: compare larger budgets in stage 10); the plugin's decode
interleave (at most 2 prefill steps before a decode step) is on.

### Decode reload contract v1

Commands are executed exactly: `slot_remap` first (ShortConv rows, k / p / temperature rows, RNG seeds unless reset
on the same step, and the resident token / position / page-table rows unless reloaded), then `reload_inputs` (tokens,
positions / masks, page table) or `reload_page_table` (page table only; the generator copies it only when it changed)
or nothing (device-resident token feedback and positions), then `reload_sampling_params` / `reset_sampling_state`.
Device steady state = model trace + sampling trace replays with no host token / position writes (round-1 mixed-shape
run, 406 decode steps: 364 device-resident, 5 page-only refreshes, reload+reset at layout changes, 15 host steps).
Adapter tests (`logs/test_generator_vllm_full_r5.log`, 40 layers, **23 passed**; reduced `logs/test_generator_vllm_r5.log`
23 passed): stale tokens / positions with `reload_inputs=False` == host-authoritative reloads; lazily grown page
tables with page-only refreshes == full reloads and unchanged steps copy no page table; `slot_remap` mid-decode with
and without reload keeps every request's tokens identical; async read == sync.  Transition steps (reload with partial
seed rows, `slot_remap` without reload) still contain small blocking readbacks (seed / token read-modify-write for the
rows not given); removing them is a stage-10 item.

### ShortConv state across prefill calls and prefix hits

30 of the 40 layers keep a per-slot ShortConv state.  The plugin picks a prefill row's state slot per call
(`_alloc_prefill_state_slots`; no remap between prefill chunks), so a continuation chunk can land in another slot,
and a prefix-cache hit starts a request at `start_pos > 0` with no state at all.  The adapter therefore carries conv state by content: after every chunk ending on
the 2048-token grid it snapshots the slot into a preallocated DRAM pool (128 entries x 1.9 MB per chip, allocated
before trace capture) keyed by a blake2b chain hash of the token history; a chunk starting at `P` uses the highest grid
boundary `Q <= P` whose state is already in its slot or in the pool and restores it (exact copy).  If `Q < P` the
tokens `[Q, P)` are recomputed to rebuild the ShortConv state, and their K/V is written into the **adapter-owned
scratch row** (2048 blocks past vLLM's pool = one full page-table row, ~35.7 MB per attention layer per chip,
~0.33 GiB per chip; the row's page-table entries `[Q/64, P/64)` point to scratch for this prefill call only, so any
walk-down depth fits; a range that does not fit raises `ValueError`).  vLLM's cached blocks — which other live
requests may share — are **never written by a hit**; the chunk `[P, E)` and all later decode steps read the real cached
blocks for `[0, P)` through vLLM's page table.  Worst case `Q = 0`.  Only 2048-aligned starts have prefill programs,
so no new SDPA variants or trace keys exist.  User decision: 2048 grid in stage 9, 512 grid in stage 10.

### Prefix caching (user requirement)

| run (final code, allocation tracking on unless noted) | shared prefix | TTFT per request | outputs |
|---|---|---|---|
| sequential, caching on (`prefix_ab_on_final3.json`) | 12045-12052 prompt tokens (6000-word system prompt), 4 chat requests | 1.461 s (cold), then **0.235 / 0.233 / 0.236 s** | identical to the round-1 run (4/4) |
| sequential, `--no-enable-prefix-caching` (round 1, `prefix_ab_off.json`) | same | 1.472 / 1.377 / 1.405 / 1.377 s | identical to the cached run (4/4) |
| **concurrent cold-owner, caching on** (`prefix_ab_concurrent_on_cold3.json`, tag `session-c9`: the owner computes its new tail fresh and decodes 160 tokens while 3 requests arrive and hit mid-prompt, recomputing `[Q, P)` via the scratch row) | 12050-12057 prompt tokens | owner 1.447 s; hits **0.646 / 0.671 / 0.671 s** | — |
| concurrent cold-owner, `--no-enable-prefix-caching` (`prefix_ab_concurrent_off_cold3.json`, `logs/server_full_nopc3.log`, no tracking) | same | 1.464 / 1.450 / 2.873 / 4.305 s | **identical** to the cached concurrent run (4/4) |

Files under `readiness_vllm/prefix_cache/` (greedy, temperature 0, repetition_penalty 1.0, 48 tokens; the concurrent
owner 160 tokens); hit rows in `logs/events_full_track3.jsonl`.  Earlier rounds (`prefix_ab_on*.json`,
`prefix_ab_concurrent_on*_cold.json` round 2) are kept; `prefix_ab_concurrent_on.json` and `..._on_fresh.json` are
**not** cold-owner runs (the owner itself hit the earlier cache; one answer differs from the sequential run from
character 160: concurrency / decode-bucket numerics).

Adapter tests (`logs/test_generator_vllm_full_r5.log`): prefix hit at 4096 in a new slot == the same prompt from
scratch (top-1 / PCC); after evicting the 4096 snapshot the adapter recomputes exactly 2048 tokens from the 2048
snapshot; `test_prefix_hit_does_not_mutate_shared_blocks`: owner A (5000 tokens), B hits at 4160 -> A's KV blocks
bitwise unchanged, A's greedy continuation == control, B's logits match B from scratch (negative control with the
scratch substitution disabled **fails** on the block compare, `logs/test_shared_blocks_negative_control.log`);
`test_prefix_hit_two_level_eviction`: 4096 and 2048 snapshots evicted, hit at 4160 -> `Q = 0`, 65 scratch blocks,
owner blocks bitwise unchanged, hit top-1 / PCC vs from-scratch.

### Sampling

* Device sampling: traced `TTSampling` split path (greedy = k 1, p 0, temp 1) writing straight into the next decode's
  token input; prefill first token sampled by the traced prefill sampler.
* Seeds are position-derived: `seed = base(request seed) + position of the sampled token`, re-aligned from
  host-authoritative positions on reset / reload steps and advanced once per sampled token on device, so a seeded
  stream does not depend on slot, batch composition or async timing (adapter test: seeded top-k request alone in slot 0
  == next to two other requests in slot 2).  Unseeded requests get a fresh random base.
* Host sampling (plugin-selected, explicit compat path, never on the measured path): logprobs, penalties, min_p,
  bad_words, logit_bias, allowed tokens, structured output, top_k > 32, and steps containing an intermediate prefill
  chunk.  Prefill: last-row logits read from the traced chunks' persistent logits buffer (8 MB); decode: model trace
  only (sampler not replayed, device RNG untouched), vocab-sharded logits read asynchronously.
* **Checkpoint generation_config** (`temperature 0.1, top_k 50, repetition_penalty 1.05`) is applied by vLLM to every
  request field left unset; with no device penalties those requests are host-sampled.  User decision: keep the
  checkpoint defaults (`--generation-config auto`), run the benchmarks as pure greedy with `--repetition-penalty 1.0`
  sent explicitly, add device repetition / presence / frequency penalties and top_k up to 64 in stage 10.  Consequence:
  the runner's qualitative requests and the whole plugin sampling suite are host-sampled on the production server.

### Which evidence exercises the device sampler

* adapter device tests (`test_generator_vllm.py`, 23): device prefill sampler + split decode sampler, greedy, top-k,
  seeded;
* mixed-shape checks, parity checks, prefix A/B runs, benchmarks (requests with temperature 0 / top-k <= 32 and
  repetition_penalty 1.0);
* plugin suite on a server with `--generation-config vllm` (`logs/server_full_genvllm.log`, script
  `tests/vllm_suite_sampling_paths.py`, `logs/suite_sampling_paths_genvllm.log`,
  `analysis/suite_sampling_paths_genconfig_vllm.json`, events `logs/events_full_genvllm.jsonl`; round-2 code, before the
  all_gather fix): 79 non-speculative tests run one at a time, **77 passed, 1 skipped** (all-vocab logprobs, clamp 20),
  **1 error** (`test_tile_size.py::test_plugin_tile_matches_ttnn`: collection-order import cycle when run standalone,
  documented in the plugin's `AGENTS.md`; it passes in every full-suite run); `test_bad_words` passed there.  13 tests
  send no request, 50 were host-sampled only, 16 had device-sampled steps:

| test | prefill device / host | decode device / host (device resident) |
|---|---|---|
| `test_chunked_prefill::test_a_solo_split_prefill_recalls_its_needle` | 1 / 1 | 23 / 0 (22) |
| `test_chunked_prefill::test_a_long_split_prefill_recalls_its_needle` | 1 / 4 | 23 / 0 (22) |
| `test_chunked_prefill::test_prefills_sharing_a_step_each_recall_their_own_needle` | 12 / 12 | 78 / 0 (57) |
| `test_request_isolation::TestBatchIsolation::test_mixed_params_batch` | 1 / 3 | 0 / 18 (0) |
| `test_seeding_and_variety::test_seeding` | 1 / 2 | 0 / 18 (0) |
| `test_seeding_and_variety::test_topk[15]` / `[19]` / `[32]` | 3 / 0 each | 18 / 0 (15) each |
| `test_seeding_and_variety::test_top1_is_greedy` | 3 / 0 | 18 / 0 (16) |
| `test_structured_output_dp1::test_dp1_full_capacity_mixes_structured_and_plain_requests` | 0 / 2 | 2 / 14 (0) |
| `test_structured_output_dp1::test_dp1_reuses_slots_with_mixed_requests` | 3 / 7 | 8 / 19 (4) |
| `test_tt_penalties::TestRepetitionPenalty::test_repetition_penalty_mixed_batch` | 1 / 1 | 0 / 9 (0) |
| `test_tt_penalties::TestPresencePenalty::test_presence_penalty_changes_output` | 23 / 8 | 184 / 184 (176) |
| `test_tt_penalties::TestPresencePenalty::test_presence_penalty_is_per_request` | 2 / 2 | 19 / 38 (18) |
| `test_tt_penalties::TestFrequencyPenalty::test_frequency_penalty_changes_output` | 20 / 8 | 184 / 184 (176) |
| `test_tt_penalties::TestFrequencyPenalty::test_frequency_penalty_mixed_batch` | 1 / 1 | 0 / 14 (0) |

(device-step counts: adapter events in each test's time window; penalty tests get device steps from penalty-free requests.)

### Sampling suite details (production server)

`readiness_vllm/sampling_tests.log` (runner attached, `--tt-max-num-seqs 32`): 1 failed, 74 passed, 46 skipped in
447.0 s; identical counts in rounds 1 and 2 (`readiness_vllm/superseded_r{1,2}/sampling_tests.log`) and in the `-rs`
rerun with skip reasons (`logs/sampling_full_rs.log`).

* **Failed `test_host_only_params::test_bad_words`** (`analysis/bad_words_failure.md`), the same response 4 in every
  round (`'... ("Say a hello to me") ...'`): the request is host-sampled by vLLM's own bad_words logits processor; the
  banned token ids ('hello' 52071, ' hello' 63532, 'Hello' 35808, ' Hello' 59959) were never generated in 3 x 5
  replays; the word comes from the non-canonical split `' hell' + 'o'` (`[(94866,'Say'), (267,' a'), (18069,' hell'),
  (87,'o'), (310,' to')]`), reproducibly for seed 4 in the 5-request batch.  vLLM bans exact token sequences; the test's
  whole-word string check is stricter than that contract.  Not a TT serving defect (the same test passes on the
  `--generation-config vllm` server, where the sampled text differs).
* Skipped: 41 speculative-decode tests ("the server is not speculating"), 1 spec test needing `--tt-spec-artifacts`,
  3 `test_chunked_prefill.py` tests that need `--tt-chunked-prefill-budget` (the runner does not pass it; run directly
  with `--tt-chunked-prefill-budget=2048`: **3 passed**, `logs/sampling_chunked_prefill.log`), 1 all-vocab logprobs
  test (the plugin clamps `max_logprobs` to 20: *"max_logprobs=128000 exceeds TT device limit of 20, clamping to 20"*,
  `logs/server_full_r2.log`).
* Smoke profile on every server variant (incl. final code + tracking, `logs/sampling_smoke_full_track3.log`): 3
  passed, 1 skipped (same all-vocab logprobs skip).

## Qualitative verdict ($qualitative-check)

Prompt format: the checkpoint has a chat template, so the runner sends the shared 6-prompt suite through
`/v1/chat/completions` (`prompt_mode: chat` in `vllm_qualitative_outputs.json`), greedy (temperature 0) and sampled
(temperature 0.7), 256 tokens, other fields from the checkpoint generation_config (repetition_penalty 1.05, top_k 50;
host-sampled).  All 12 completions are coherent, on topic, English, with the model's reasoning-style preamble, no
repetition loops, no gibberish, no prompt echo beyond quoting the task, no cross-request leakage.  Greedy completions
are identical to rounds 1 and 2:

| id | prompt | greedy (start) | sampled (start, final run) |
|---|---|---|---|
| 0 | haiku about ML | "We need to write a haiku about machine learning. Haiku: 5-7-5 syllables ... 'Data flows like streams / Models learn from the patterns found' ... Too many. Need 7." | "... Example: 'Data flows like streams / Patterns emerge from the noise / Learning from the past.' Check syllables ..." |
| 1 | supervised vs unsupervised | "We need to answer: 'Explain the difference ...' ... definitions, examples ... labeled data vs unlabeled data ..." | "We need to respond with an explanation of the difference between supervised and unsupervised learning, in simple terms ..." |
| 2 | complete the story | "1. **Analyze the Request:** ... *Idea 3:* He discovered a way to capture dreams or thoughts using gears and steam." | "1. **Analyze the Prompt:** ... **Key Elements:** Faraway kingdom, curious young inventor, discovery (the hook) ..." |
| 3 | laws of thermodynamics | "... Zeroth Law: Thermal equilibrium. First Law: Conservation of energy ... Third Law: Absolute zero" | "... Zeroth Law: Thermal equilibrium. First Law: Conservation of energy (Energy cannot be created or destroyed) ..." |
| 4 | translate to French | "... 'Hello': ... 'Bonjour' (used during the day) ... 'how are you': Formal: 'Comment allez-vous ?'" | "The user wants me to translate ... 'Hello' -> Standard greeting. 'how are you' -> Inquiry about well-being ..." |
| 5 | Fibonacci in Python | "The user asks: 'Write a Python function ...' ... iterative, recursive, using memoization ..." | "... Could include both iterative and recursive approaches, maybe also using memoization or generator." |

Degenerate check (`check_degenerate_output.py --scope vllm`, final run): no degenerate output (max trigram loop
fraction 0.127).

**Controls** (`tests/vllm_qual_parity.py`: the same chat-rendered token ids, `doc/datatype_sweep/qualitative/
prompts.json`, to `/v1/completions`, greedy with `repetition_penalty 1.0`, 128 tokens; final code
`qualitative_parity/final/vllm_greedy_parity.json`):

| prompt | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| vLLM solo vs standalone TT (stage 8 selected run) | identical | identical | identical | identical | identical | identical |
| vLLM 6 concurrent vs standalone 6-slot batched control | identical | identical | identical | identical | identical | identical |
| vLLM solo first divergence vs HF greedy (CPU bf16, rep 1.0) | 27 | 5 | 102 | 57 | 38 | 21 |
| stage 8 C1 (`doc/datatype_sweep/logs/qual_C1.log`) | 27 | 5 | 102 | 57 | 38 | 21 |
| 6 concurrent: first divergence vs solo | 50 | 17 | 109 | 57 | 38 | 100 |
| 6 concurrent: first divergence vs HF | 27 | 5 | 102 | 81 | 62 | 21 |

## Mixed-shape trace reuse (`tests/vllm_mixed_shape_check.py`)

Workload on one live server, generator-wide preparation only: first encounters of 129 / 130 / 131 tokens; 3000- and
5000-token prompts arriving (0.4 s stagger) while 77 / 300 / 1500-token requests decode (300 sampled); repeats in
other compositions and orders; greedy -> sampled -> host (logprobs 3) -> greedy; a 4200-token prompt sharing 4096
tokens with the 5000-token one (twice).

**Final code, allocation tracking on** (`logs/mixed_shape_full_track3/mixed_shape_report.json`, events
`logs/events_full_track3.jsonl`, server `logs/server_full_track3.log`):

* traces before the workload: 20 decode / sampling + 16 prefill captures (trace ids recorded in the `traces_ready`
  event; 71.2 MB per bank of the 768 MiB trace region measured in round 1); after: **20 + 16, 0 recaptures**; at
  shutdown `release_traces 20 16`.
* chunk rows: continuation starts 64, 128, 2048, 4096, 4160 (2048 / 4096 = chunked 3000 / 5000 prompts; 64 / 128 =
  prefix hits on repeats, recomputed from 0; 4096 / 4160 = prefix hits of the 4200-token prompt, recomputed from the
  4096 snapshot); recompute starts 0 / 2048 / 4096; snapshots 3 stores, 2 restores, 3 in-slot continuations, no
  evictions; 9 rows redirected to scratch.
* greedy determinism across compositions: u129 / u130 / u131 / d77 / p4200 all identical.
* TTFT p50 / p95 / p99 0.146 / 0.709 / 0.710 s (first encounters included); ITL p50 / p95 / p99 10.8 / 97.2 /
  238.8 ms; 4 decode pauses > 250 ms (max 347 ms).

Also passed in rounds 1-2 and on the reduced target (`readiness_vllm/mixed_shape/`, `logs/mixed_shape_*`), 0 recaptures.

## Trace allocation safety

`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` are set in the server environment by the
launch command (recorded in `work_log.md`, "commands"), before TTNN import; the adapter logs `trace allocation
tracking ON` from inside EngineCore (`logs/server_full_track3.log` final code; also `server_full_track{,2}.log`,
`server_reduced_track2.log`).  Workloads: mixed-shape check, prefix-cache requests (sequential and concurrent
cold-owner), smoke sampling.  No `still alive` / unsafe-allocation report, no traceback.  The only reviewed
corruptible buffer is the stage-6/7 acknowledged decode logits buffer (model trace -> sampling trace of the same
bucket).  The snapshot pool, the scratch KV row (part of the KV allocation), prefill inputs and history buffers are
allocated before the first capture; snapshot / restore / permute temporaries are freed before the next replay.

## Context, prompt lengths, concurrency

* Served `max_model_len` 131072 = `doc/context_contract.json` (`vllm_integration` block); stage gate
  `09-vllm.check.sh`: *Context contract OK ... target=131072, supported=131072 (full HF context)* (`logs/check_09.log`,
  exit 0, final run).  KV pool 65568 blocks x 64 tokens (32 x 131072 + headroom) + 2048 adapter scratch blocks, BFP8,
  10 attention layers.
* The longest served request in this stage is ~12.2k tokens (prefix runs); the full 131072 context is backed by the
  context contract and the full-context evidence of stages 1-8 (same model, KV layout and page-table width), not by a
  131072-token serving request.
* Non-aligned prompts: adapter prefill of 1, 31, 33, 129, 130, 131, 2047, 2049, 4100 tokens == eager whole-prompt
  logits (top-1 equal, PCC > 0.999); served 129 / 130 / 131 / 77 / 300 / 1500 / 3000 / 4200 / 5000 / ~12050 tokens.
* Concurrency: `max_num_seqs` 32 everywhere; CI burst 32 concurrent requests (32/32 completed); sampling suite with
  `--tt-max-num-seqs 32`.
* Cleanup: every server stopped with SIGINT on its exact PID; afterwards `pgrep EngineCore|vllm.entrypoints` empty and
  no holder in `/proc/driver/tenstorrent/*/pids`.

## Anomaly ledger

| observed | evidence | control / investigation | resolution |
|---|---|---|---|
| **Prefill nondeterminism (found re-testing after round 2)**: `test_determinism_slots_and_repeats` failed once on 40 layers (`logs/test_generator_vllm_full_r3.log`, passed on rerun `full_r4`); the same 300-token prompt gave different last-row logits in 9/149 prefills with decode interleaved, 10/149 with a sync before each prefill, 1/149 without decode, 1/149 eager; layers 0-19 4/149, 20-39 2/149, 3-layer and layer-2-only 0/149 (`logs/diag_*`, `analysis/diag_determinism_stress.py`); large diffs (max abs 1-4, top-1 flips) | AutoDebug (`analysis/AUTODEBUG.md`) H1, verified against source: the multicast `all_gather` used by the prefill all-reduce for the 64..512-row buckets on the 1x4 ring alternates data packets over primary / alternate routes but sends the completion credit only on the primary route, so a chip can finish before the 2-hop chip's alternate-route packets land and the next op reads stale tiles; control: bucket 1024 (unicast all_gather) unfixed 0/199 (`logs/diag_stress_len900.log`) | **root-caused + fixed** (`analysis/AUTOFIX.md`): `all_gather_multicast_factory.cpp` `axis_load_balance = false`; model stress 0/299 (len 300) and 0/199 (len 100); new op regression `tests/test_ccl_ring_all_gather.py` (traced all_gather -> add, fresh random inputs, 1000 replays x 4 chips, rows 128 / 512): unfixed 37/4000 stale at 512 rows, fixed 2 passed; perf unchanged (TTFT 128/1024/2048/4100 65.6/97.4/135.1/296.1 ms vs 65.7/97.5/135.0/295.9; token-out 107.96 vs 107.97 t/s/u; b4/b8/b32 13.63/19.08/26.10 ms) |
| **P1 round 1**: a prefix hit recomputed `[Q, P)` into vLLM's (possibly shared) cached blocks with bucket-dependent numerics | review; old claim "identical K/V rewritten into the request's own blocks" | recompute redirected to adapter scratch; `test_prefix_hit_does_not_mutate_shared_blocks` + negative control (fails without the fix); concurrent cold-owner run == caching off | **fixed** |
| **P1 round 2**: 32 scratch blocks could overflow on deeper walk-downs | review | scratch = one full page-table row (2048 blocks), explicit `ValueError`; `test_prefix_hit_two_level_eviction` (`Q = 0`, 65 scratch blocks); 23/23 reduced + full | **fixed** |
| **P2 round 1**: with the checkpoint generation_config the plugin suite is host-sampled only | `logs/events_full_r2.jsonl`: 0 device steps | per-test attribution on a `--generation-config vllm` server: 16 tests with device steps (table above) | **resolved** (production server keeps the user's generation_config decision) |
| 6 concurrent greedy requests differ from the same requests alone | `qualitative_parity/` | deterministic over repeats; standalone generator reproduces solo and batched bitwise (final code too); batched never further from HF than solo | controlled: decode-bucket numerics (bucket 8 vs 1 kernels), not serving |
| `test_bad_words` failure | `readiness_vllm/sampling_tests.log` | token-level replay | controlled: vLLM bad_words semantics (`analysis/bad_words_failure.md`) |
| 4 decode pauses > 250 ms in the mixed-shape runs (max 343-347 ms) | `mixed_shape_report.json` | coincide with the phases containing 2048-token prefill chunks and the host-sampled logprobs step; attribution remains phase-level | stage-10 item (batched-token budget, host-sampling cost) |
| host-sampled logprobs request TTFT ~0.7 s (and the greedy request sharing its step) | mixed-shape `switch_host_logprobs` | host sampling path; first-use cost not isolated | expected compat cost; stage 10 |
| blocking readbacks on transition steps | `set_row_seeds`, `permute_slots` | steady decode has none (TPOT == standalone) | stage-10 item |
| reduced-target text repeats; weight cache rebuilt once (`model.py` in the key); plugin assertion with prefix caching | `logs/server_reduced_*`, `test_generator_vllm_full_r1.log` | — | not a quality signal; by design; local plugin patch |

## Limitations

* **Earlier-stage accuracy numbers (stages 4-8) were measured with the all_gather race present** (pre-existing since
  the stage-4/5 `MeshReduce.prefill`; rare, ~1/149 per prefill without decode, ~9/149 with decode interleaved).  The
  stage-9 runs before the fix are affected the same way; all headline stage-9 evidence was regenerated after it.
* The all_gather fix is a local tt-metal change (`all_gather_multicast_factory.cpp`), not upstreamed (user's call); it
  disables route alternation in the multicast all_gather (no measurable perf change here).
* The checkpoint's default sampling settings (repetition_penalty 1.05, top_k 50) are host-sampled until stage 10 adds
  device penalties and top_k 64; clients sending their own params with no penalty and top_k <= 32 use device sampling.
* Prefix reuse of ShortConv state is on a 2048-token grid: a hit recomputes up to 2047 tokens (into scratch), or more
  after snapshot evictions; a shared prefix shorter than 2048 tokens is recomputed from 0 (vLLM still reuses its
  attention KV blocks for decode); finer grid in stage 10.  Snapshot pool of 128 entries (LRU).
* `max_logprobs` is clamped to 20 by the plugin; logprobs and penalties force host sampling.
* `max_num_batched_tokens` 2048 (vLLM default) bounds decode stalls but costs TTFT on long prompts; compare in stage 10.
* The local plugin patch is not upstream; any reinstall from the pinned commit needs it.
* No Tracy / perf-report in this stage (per skill); device-op context comes from stages 7 / 8.

## Commands

```bash
source session_state/env.sh
export TT_MODEL_CLASS_OVERRIDES="TTLfm2MoeForCausalLM=models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm:TTLfm2MoeForCausalLM"
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin HF_HUB_OFFLINE=1
LFM2_VLLM_LAYERS=all pytest -q $M/tests/test_generator_vllm.py          # reduced default: layers 0,2,3
pytest -q $M/tests/test_ccl_ring_all_gather.py                          # all_gather race regression
# final evidence: one runner-launched server, stages attached (logs/runner_final3_*_cmd.txt)
python -m readiness_check.run_vllm_server --stages serve --model-dir $M --hf-model LiquidAI/LFM2.5-24B-A2B \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 131072 \
  --tt-config '{"trace_region_size": 805306368, "l1_small_size": 24576, "prefill_chunk_alignment": 2048}'
python -m readiness_check.run_vllm_server --stages sampling --server-url http://localhost:8000 --model-dir $M \
  --hf-model LiquidAI/LFM2.5-24B-A2B --max-num-seqs 32 --sampling-profile full
python -m readiness_check.run_vllm_server --stages qualitative,benchmark --server-url http://localhost:8000 \
  --model-dir $M --hf-model LiquidAI/LFM2.5-24B-A2B --max-num-seqs 32 --max-model-len 131072 \
  --additional-benchmark-args "--repetition-penalty 1.0"
python -m pytest $VLLM_TT_PLUGIN_ROOT/tests/tt/test_chunked_prefill.py --tt-server-url=http://localhost:8000 \
  --tt-model-name=LiquidAI/LFM2.5-24B-A2B --tt-max-num-seqs=32 --tt-chunked-prefill-budget=2048
# check servers (session_state/scratchpad_tools/serve_lfm2.sh mirrors the runner's launch command)
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 LFM2_VLLM_EVENTS=<jsonl> serve_lfm2.sh <log>
LFM2_VLLM_EVENTS=<jsonl> serve_lfm2.sh <log> --no-enable-prefix-caching    # or --generation-config vllm
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.<vllm_mixed_shape_check|vllm_prefix_ab|vllm_qual_parity|batch_variance_control> ...
python "$TT_MODEL_BRINGUP_ROOT/runtime/readiness_check/check_degenerate_output.py" --hf-model LiquidAI/LFM2.5-24B-A2B --missing-artifacts critical --scope vllm
MODEL_DIR=$M bash $TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/09-vllm.check.sh
```

## Artifacts

| path | content |
|---|---|
| `tt/generator_vllm.py`, `tt/generator.py`, `tt/model.py` | adapter; generator / model hooks |
| `tests/test_generator_vllm.py` (23), `tests/test_ccl_ring_all_gather.py` | adapter device tests; all_gather race regression |
| `tests/vllm_mixed_shape_check.py`, `vllm_prefix_ab.py`, `vllm_qual_parity.py`, `vllm_suite_sampling_paths.py`, `batch_variance_control.py` | serving checks |
| `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp` (tt-metal) | all_gather race fix |
| `readiness_vllm/` | final runner run (server, sampling, qualitative, both benchmarks); `prefix_cache/`; `mixed_shape/` (round 1); `superseded_r1/`, `superseded_r2/` |
| `doc/vllm_integration/logs/` | adapter tests (`test_generator_vllm_*`, negative control), op regression (`test_ccl_ring_all_gather_{unfixed,fixed}.log`), stress (`diag_*`), `perf_after_allgather_fix.{json,log}`, servers, events, `mixed_shape_*`, sampling logs, runner logs + commands (`runner_final3_*` final), degenerate / gate checks |
| `doc/vllm_integration/qualitative_parity/` | parity (`final/` on final code), repeats, standalone batch-variance control |
| `doc/vllm_integration/analysis/` | `AUTODEBUG.md`, `AUTOFIX.md`, `diag_determinism{,_stress}.py`, `bad_words_failure.md`, `suite_sampling_paths_genconfig_vllm.json` |
| `doc/context_contract.json` | `vllm_integration` block |
| `vllm-tt-plugin` (separate repo) | local patch `platform.py` + `tests/test_chunked_prefill_policy.py` |
