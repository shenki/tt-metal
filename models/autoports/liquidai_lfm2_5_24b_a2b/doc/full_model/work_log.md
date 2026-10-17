# Stage 6 (full model) work log

Base: stage 5 commit bf1d8a2e (SHA log 59366b94), branch `autoport/lfm2_5_24b_a2b`.  Plan:
`/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/stage6_plan.md` (planning subagent).  Goal contract:
`bringup/artifacts/multigoal-runs/20261015T041226Z/06-06-full-model.prompt.txt`.  Hardware: 4 x p150b, checked free
(`qwen3.8-flash-next.service` inactive, no Qwen container, no device PIDs), `tt-smi -ls`, mesh smokes 1x1 / 1x4 clean.

## F-0 reference (CPU)
* `python -m readiness_check.generate --prompt-source aime24 --chat-template ...` failed: transformers 5.12.1's
  `apply_chat_template(tokenize=True)` returns a `BatchEncoding`; `_chat_or_plain_prompt_tokens` iterated its keys
  (`ValueError: invalid literal for int() ... 'input_ids'`).  Package left untouched; `tests/gen_reference.py` patches that
  one helper (unwrap `input_ids`) and writes a metadata sidecar.  HF cache `refs/main` = pinned revision 1769516a.
  41 s on CPU; 157-token prompt, 100 tokens.  `logs/aime24_reference_gen.log`.

## F-1 model.py, reduced probe eager (layers 0, 2, 3)
* Streaming 40-layer build on `MultichipDecoder.from_state_dict`, one shared `FusedRope`, replicated bf16 embedding,
  `ShardedNorm` final norm on `resid_mem`, DS LM head.  Probe build 29 s.
* LM head config: (32, 2, 2) -> `TT_FATAL in1_shard_width_tiles % workers_per_bank` (125 tiles per bank); readers
  must be 1..3 and divide 125 -> only 1; (32, 2, 1) overflows L1 (1.72 MB CBs).  `repro/head_sweep.py`:
  32000 columns: only 16/32/64 cores x in0 1 x 1 reader fit, 393.6 us.  Padding the GLOBAL vocab at its tail to
  4 x 32256 (zero rows; chip d column j = token 32256 d + j; the sampler masks the tail): (64, 1, 3) 175.6 us,
  (32, 2, 3) 180.9, (16, 4, 3) 213.2, (16, 4, 2) 252.7; 8 cores do not fit.  -> `HEAD_CFG = (64, 1, 3)`,
  `HEAD_COL_UNIT = 768`.  `logs/head_sweep_bfp8*.log`.
* Probe eager vs HF (`tests/hf_probe.py`, fp32 CPU): prefill all-logits PCC 0.99835 (65 random tokens), last-row path
  PCC 0.99952 with all other rows exactly 0, 4 eager decode steps PCC 0.992-0.9996 with HF top-1, on-device
  `cur_pos` advance verified (idle rows stay -1).

## F-2 generator.py, probe end to end
* `Lfm2Generator`: TTSampling split sampling, per-bucket (model, sampling) traces, prefill-sampling trace, history
  trace, two-phase `prepare`, counters.  Fixed: `counters` used before init.
* Probe smoke (77-token prompt): prefill_logits PCC 0.99838; token-out == host-compat greedy == HF greedy (24 tokens);
  teacher forcing 24/24 vs HF; slot 17 == slot 0; steady state = replays only.

## F-3 all-layer readiness r1
* The packaged runners import `tt/generator.py` by file path (`spec_from_file_location`): relative imports failed
  (`ImportError: attempted relative import`) and a module-level shared-generator cache would be duplicated -> absolute
  imports, registry `SHARED_GENERATORS` in `tt/model.py`.
* `readiness_driver --runs prefill,tf`: 40-layer build 413 s, prepare 6.6 s (32.5 s on the very first, cold run).
  Prefill top-1 / 5 / 100 = 98 / 100 / 100 %; teacher forcing 98 / 100 / 100 %, TTFT 448 ms, 101.5 t/s/u.
  `logs/readiness_prefill_tf_r1.{log,json}`.

## F-4 perf + autoregressive r1
* `--runs perf,ar`: TTFT 211 ms, token-out 105.7 t/s/u (9.46 ms), TF 101.8, logits-only 9.06 ms, sampling-only
  0.376 ms; b4 13.87 / b8 19.29 / b32 26.27 ms; force-argmax 100.2 t/s/u (sampling 0.903 ms), same tokens -> split.
* AR: TT diverged from the runner's HF output at token 0 (" a" vs " something").  HF bf16 forward at the prompt end:
  " a" 0.5437, " something" 0.3737 (forward with mask / use_cache / `generate`'s raw logits identical; fp32 0.5401 /
  0.3766).  Checkpoint `generation_config.json` has `repetition_penalty: 1.05`, applied by HF `generate` even with
  `do_sample=False`: HF's "greedy" is penalised greedy; TT is the pure argmax the readiness contract requires.
* TT continued past `<|im_end|>`; free-running `generate` now stops at EOS by default (teacher forcing never stops
  early).  (The first edit of that default did not apply — replacement string mismatch — caught in r2: the AR output
  was still 256 tokens; fixed with an exact edit.)

## F-5 qualitative suite
* `tests/qualitative.py --hf` (CPU, pure greedy `repetition_penalty=1.0`) + `--runs qual`: chat template, 6 prompts,
  128 tokens, TT greedy + TT top-k.  First divergence vs HF greedy at 27 / 5 / 102 / 57 / 68 / 21; outputs read: all
  coherent (README verdict).

## F-6 probe tests under the trace allocation tracker (`tests/test_full_model.py`)
Iterations (all test-side unless noted):
1. first-token check vs HF argmax failed (91 vs 7203): the probe's HF logits are flat on real text (top-5 within 0.46);
   checks became HF top-5 + logged top-1 rates; the feedback proof became exact (below).
2. tracker: "7 device buffers still alive before trace replay" = the test's own `ttnn.clone` snapshots kept across a
   replay -> host-side snapshots.  (Confirms the tracker covers lifetime bugs of exactly this kind.)
3. host snapshot copied chip 0's slice to all chips: ShortConv state is channel-sharded -> per-chip concat / shard.
4. the 3-layer probe echoes its input token under greedy (t1 == input), which would hide a stale-token bug -> replay 1
   samples (top-k 32, temperature 20, seed 3), so t1 != input; replay 2 (no host input) must equal bitwise the same step
   with t1 / position 66 written from the host.
5. generator bug found: `set_sampling_params(seed=...)` indexed a uint32 torch tensor (`index_put` unsupported) ->
   seeds kept as int64 on host.
Result: 18 passed with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
(`logs/test_full_model_tracker_r1.log`); acknowledged buffer: per-bucket model-trace logits (README).

## F-7 probe profile, sampler A/B
* `repro/tracy_probe.sh`: needed `--op-support-count=20000 --dump-device-data-mid-run` (build / warm-up overflowed the
  device profiler buffers: "Device data missing: Op ... device 3") and `pytest --timeout=2400` (pytest-timeout 300 s
  killed the profiled run; mesh smoke afterwards clean).
* `repro/sampler_ab.py` (probe): split 375.7 us, split + `pad_logits_to_power_of_2` 339.7 us, force-argmax 868.3 us,
  same tokens -> padded split is the default.  `logs/sampler_ab_probe.log`.
* Final probe profile (`perf/tracy_probe`, `perf/probe_decode_breakdown.txt`): 1070.4 us device / step, decoder layers
  573.2 us (stage 5: 567.7), norm + head 186.0, sampling trace 287.5 (top-k 124.3), embedding + cos/sin 21.9.

## F-8 final all-layer evidence
* `--runs prefill,tf,perf,memory,qual,ar` (`logs/final_all_runs.{log,json}`): readiness unchanged (98 / 100 / 100 both),
  AR stops at EOS (174 tokens), degenerate check pass; memory 23.22 GiB DRAM of 31.58 per chip (73.5 %), persistent L1
  189,696 B per core, trace region 8.9 of 32 MiB per bank with 4 buckets captured.
* The perf run there still polled EOS every 16 steps in the "token-out" window (EOS default); the headline perf is from
  the rerun with `stop_on_eos=False` and the polling variant reported separately: `logs/final_perf.{log,json}`.
* Watcher (`logs/watcher/`, `logs/watcher_probe.log`): needs `TT_METAL_WATCHER_NOINLINE=1` (fabric ERISC kernel too
  large for the kernel-config buffer otherwise); 5 tests passed, 0 error lines.
* Gates: `06-full-model.check.sh` exit 0 (`logs/check_06_full_model.log`); context contract OK.

## F-9 stage review round 1 (Fable subagent, read-only): more-work-needed
* P1 `reset()` compiled `ttnn.fill` on the whole BFP8 KV cache / ShortConv state only on first use; every evidence run
  happened to reset before the first capture.  Fix: `prepare()` zeroes the cache and the ShortConv state once (phase 1);
  new first test `test_reset_after_capture` (first reset / reset_slots after a capture).  Negative control (warm-up
  removed): tracker raises on `program_cache: UnaryDeviceOperation FILL ... BFLOAT8_B`
  (`logs/test_reset_after_capture_negative_control.log`); with the fix 21 passed (`logs/test_full_model_tracker_r2.log`).
* P2 sampler A/B compared two identical configs (pad default True on both): `repro/sampler_ab.py` builds the base
  generator with `pad_logits_to_power_of_2=False`; rerun: 375.4 / 340.2 / 868.5 us (unpadded / padded / force-argmax).
* P2 `set_sampling_params(seed=)` reset every other slot's seeds -> read-modify-write of the device seeds; a short
  `tokens` vector zero-filled other active slots' tokens -> keeps them (readback); `test_serving_row_semantics`.
* Concerns addressed: perf harness builds the force-argmax generator only after the base generator's last replay and
  perf runs last in the driver; README classifies the allocator warning and states tracker coverage is probe-only;
  AR step-0 control recorded (`repro/ar_step0_control.py`, `logs/ar_step0_control.log`: HF generate with
  repetition_penalty=1.0 -> " a tiny" = TT); bucket 2 now captured in the probe tests (slots {0, 1}); `generate()`
  refuses an externally bound cache; free-running stops immediately when the first token is EOS; idle-row `rope_pos`
  invariant commented in `tt/model.py`.
* Final all-layer evidence rerun on the fixed code: `logs/final_r2.{log,json}` (order prefill, tf, memory, qual, ar, perf).

## F-10 stage review round 2 (same Fable reviewer): clean-pass
All round-1 required work and concerns verified fixed; doc-only follow-ups applied (context-contract log pointer ->
tracker r2, README sampler numbers -> final_r2, note that prepare() zeroes a bound cache).
