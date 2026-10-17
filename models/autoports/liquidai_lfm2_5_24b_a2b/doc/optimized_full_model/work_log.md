# Stage 7 (optimized full model) work log

Base: stage 6 6227a175 (+6c3bcd26), branch `autoport/lfm2_5_24b_a2b`.  Plan: `session_state/stage7_plan.md` (Fable planning
subagent, read-only).  Goal: `bringup/artifacts/multigoal-runs/20261015T041226Z/07-07-optimized-full-model.prompt.txt`.
Hardware checked free (qwen service inactive, no Qwen container, no device PIDs).  "Before" = stage-6
`doc/full_model/logs/final_r2.json`: TTFT 221.6 ms, token-out 9.430 ms, TF 9.818, b4/b8/b32 13.83/19.26/26.24.

## O-1 weight cache (W1)
* `tt/weights.py::WeightCache`: one `ttnn.from_torch` hook active only inside the model build routes device-bound
  tensors >= 64 K elements through `ttnn.as_tensor(cache_file_name=...)` (scope = globals / rope / layer; per-scope call
  sequence + shape / mapper / dtype / layout in the name).
* `tests/test_weight_cache.py` (probe, off / cold / warm, every model device tensor compared on all chips + spec):
  passed; build 26.7 -> 6.4 s (`logs/test_weight_cache.log`).
* Key first hashed every `tt/*.py` (generator edits would invalidate); narrowed to `WEIGHT_CODE_FILES` (decoder modules,
  weights.py, model_config.py) + `model.WEIGHT_LAYOUT_VERSION` + construction config.
* All-layer: cold 400.1 s (writes 2610 tensors, 54 GB), warm 80.8 s (`logs/allayer_p1_perf.log` second run).

## O-2 traced prefill (P1)
* `prefill_chunk` split into `prefill_host_inputs` (torch values) + persistent per-bucket inputs (`alloc_prefill_inputs`,
  called first in `prepare()`) + `prefill_chunk_device` (device only; chunk start passed as the program-family
  representative 0 / 2048: the SDPA program depends on the start only through 2048-alignment).  Generator: per-chunk
  host refresh, trace per (S, continuation, out) captured on first use with program-cache misses forbidden.
* Probe tracker suite: 27 passed on the first run incl. new `test_prefill_traced_vs_eager[1, 33, 300, 2049, 4100]`
  (bitwise logits / token / ShortConv state / KV blocks) and `test_prefill_trace_reuse` (`logs/test_full_model_tracker_p1.log`).
* All-layer, first attempt: TTFT 128 / 1024 / 2048 / 4100 = 87.7 / 118.5 / 154.0 / 316.0 ms, then
  `TT_FATAL: Creating trace buffers of size 272778240B ... only 268435456B is allocated for trace region` at the b32
  decode capture -> trace region 768 MiB.  Rerun (`logs/allayer_p1_perf.{log,json}`): 87.5 / 118.4 / 155.2 / 316.4 ms,
  prefill traces 2.1-4.3 MB / bank, capture 0.31-0.55 s each.  The 128-token TTFT inside the 128-step loop read 102 ms:
  the previous request's `reset()` (whole-cache fill) was still queued; the harness now syncs after reset (`_fresh`).
* Negative control (bucket-2048 inputs allocated lazily after captures): tracker did not fire (2 passed,
  `logs/test_prefill_inputs_lifetime_negative_control.log`): the late tensors were captured into a new trace and did
  not overlap earlier traces' footprints -> inconclusive; added `test_prefill_allocation_free` (no device-bound
  `from_torch` during a traced prefill after capture).

## O-3 LM head / vocab mask (S1) — rejected
* `repro/head_bias_sweep.py` (`logs/head_bias_sweep.log`): 32256 (64,1,3) 178.0 us; 32768: (32,2,2) 202.8, (64,1,2)
  214.9, (32,1,2) 214.3, (16,2,2) 253.5, (16,4,2) 256.8, (64,1,1) 402.5 us -> 32768 head loses ~25 us vs the ~45 us of
  sampler pad / mask it removes.  Fused bias in the DS matmul: runs at no cost but the mask check failed;
  `repro/ds_matmul_bias_repro.py` (`logs/ds_matmul_bias_repro.log`): a bias on the last 1024 columns lands on ~1/8 of
  them and on other columns -> bias layout not the natural column order for this config; not used.  Mask-add fallback
  16 us (saves ~12 us net): not worth it.

## O-4 hoisted slices (D1), history in the sampling trace (S2), stable top-k (S3)
* D1: no decoder change needed (`_decode_r3` consumes b-row page table / positions; `_sdpa_decode` does not re-slice in
  the r3 path).  `tests/test_full_model_ab.py`: eager logits bitwise hoisted vs per-layer at buckets 1 / 4 / 8 / 32.
* S2: per-bucket sampling + history trace; `decode_forward(record_history=True)` replays it instead of sampling +
  append.  Counter `history_replays` -> `history_appends`.
* S3: `repro/topk_stable_bench.py`: stable 99.5 vs unstable 124.2 us on [1,1,32,32768] k=32 bf16, lowest tied index
  first in both (`logs/topk_stable_bench.log`) -> `sampler._topk_stable = True`, tie-break kept.  (First edit of that
  line did not apply — formatter had split the constructor call — caught before the measurement; fixed.)
* All-layer (`logs/allayer_d1s2s3_perf.json`): TTFT(128) 65.2 ms (with the reset sync), token-out 9.353 ms, b4 / b8 /
  b32 13.75 / 19.20 / 26.20 ms.  Probe Tracy (`perf/tracy_probe/`): decode b1 1070.4 -> 1049.8 us device (top-k
  124.3 -> 99.1), b8 1625.1 us, traced prefill 4224.6 us device with 225.4 us gaps.
* D2 / D3 not pursued (< 0.1 % each).  Removing the tie-break pass not pursued (cross-chip tie order in `ttnn.sampling`
  unverified).

## O-5 evidence
* Probe tracker suite: 28 passed (`logs/test_full_model_tracker_r1.log`).  Watcher subset: 5 passed, 0 error lines
  (`logs/watcher/`).  Probe Tracy before (`perf/tracy_probe_before/`, stage 6) / after (`perf/tracy_probe/`,
  `ops_windows.csv.xz` = signpost windows only; full dump archived outside the repo).
* Final all-layer run (`logs/final_r1.{log,json}`, order prefill, tf, memory, qual, ar, perf): readiness 98 / 100 / 100
  (both), AR tokens identical to stage 6, greedy qualitative token-identical to stage 6 (top-k prompt 1 differs: stable
  top-k candidate order), perf table in README, memory 23.22 GiB / 31.08 GiB, trace 323 MB used after perf.
* Gates: `07-optimized-full-model.check.sh` exit 0 (`logs/check_07.log`); context contract OK (`optimized_full_model`
  block).  `perf_summary.json` written.

## O-6 stage review round 1 (Fable): more-work-needed
* P1 weight-cache test vacuous: `fingerprint()` filtered on `__module__.startswith("models.")` but pytest imports the
  package as `autoports...` -> 0 tensors compared (log said `tensors 0`).  Fix: match `lfm2_5_24b_a2b` in the module,
  assert > 200 tensors, build at max_seq_len 131072.  Rerun: 222 tensors identical off / cold / warm
  (`logs/test_weight_cache.log`).
* P2 cache hygiene: key now includes a checkpoint fingerprint and model.py; the hook stops before the final-norm config
  / decode state (`WeightCache.stop()`), and all-zero tensors (ShortConv state, zero constants) bypass the cache.
* P2 roofline: `repro/roofline.py` -> 693 MB / chip / token, 1.35 ms at 512 GB/s; perf_summary.json gains
  `decode_ms_per_token_device` / `roofline_ms_per_token_estimate`.
* Concerns: every capture now under the program-cache-miss guard; chunk-start representative invariant asserted;
  probe before/after labelled (history ops); TTFT harness change stated; prefill MoE row cites the stage-5 ledger;
  trace-region budget stated; `LFM2_STABLE_TOPK=0` qualitative A/B (`logs/qual_unstable_topk_ab.json`,
  `qualitative_unstable_topk_ab/`); logs force-added.
* Tracker suite on the fixed code: 28 passed (`logs/test_full_model_tracker_r2.log`).  Final all-layer evidence:
  `logs/final_r2.{log,json}` (cold build of the new cache key first).

## O-7 stage review round 2 (same Fable reviewer): clean-pass
All round-1 findings verified fixed on the live worktree; commit-time check: compact logs force-added, driver logs as .log.xz.
