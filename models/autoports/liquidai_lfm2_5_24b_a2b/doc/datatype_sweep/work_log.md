# Stage 8 datatype sweep — work log

Reconstructed from log timestamps (`logs/`) and `session_state/STAGE8_STATE.md`. All times are local, 2026-10-18.
Hardware: QuietBox 4x Blackhole p150b, 1x4 TP4. Base: stage 7 commit 13e41cd4 (+1185eabe), stage-8 code uncommitted
during the sweep.

## Planning and code (before 00:49)

- Plan (Fable planning subagent): `session_state/stage8_plan.md`. User decision: the accuracy gate applies to the main
  AIME24 ref and to the 400-token extended set (AIME24 idx 0-3), with top-1 >= 90, top-5 >= 98, top-100 = 100.
- Code:
  - `tt/precision.py`: `PrecisionConfig` and `resolve()`. Order: `LFM2_PRECISION_CONFIG` → selected JSON → baseline.
  - `tt/model.py`: per-layer policy from `layer_exceptions`, `weight_signature` cache key, BFP4-head prefill L1 budget.
  - `tt/multichip_decoder.py`: new `MeshPlan.ccl_dtype` / `ccl_moe`. The `_PartialOut` dtype follows it. The MoE
    compact partial goes through `fast_reduce_nc(output=bfp8 buf)`. Persistent `MeshReduce` scratch is one pair per
    dtype, sharing one alternation counter.
  - `tt/optimized_decoder.py`: `prefill_2d_config` takes an `l1_budget`.
  - `tt/generator.py`: the shared-generator key includes the precision.
  - `tests/readiness_driver.py`: new `--precision` flag.
  - New `tests/datatype_sweep.py`, `tests/plot_datatype_sweep.py`, `tests/test_precision_config.py` and
    `repro/check_perf_rows.py`.

## Sweep

| time | step | evidence |
|---|---|---|
| 00:49-00:51 | extended AIME24 references idx 1-3 (100 generated tokens each; e.g. p1 = 146-token prompt) | `logs/ref_gen_p{1,2,3}.log`, `bringup/references/*_p{1,2,3}.refpt` |
| 00:57-01:03 | 3-layer probe smokes of the new knobs under the tracker. `ccl_only` 7 passed. `rest_no_ccl` / `risky_probe` each failed 3 probe-accuracy tests (split-sampling, batch b8/b32); their configs were in a scratchpad that was not kept | `logs/probe_smoke_*.log`, `logs/probe_risky_smoke.log` |
| 01:03-01:48 | P0: B0 (with `--perf`: token-out 106.89, TTFT128 65.5 ms), E1, E2, F1, H1, H2, C1 | `logs/sweep_p0.out` |
| 01:48-02:39 | P1: W_a, F_all, X1, E3, X2, A0 | `logs/sweep_p1.out` |
| 02:39-03:27 | repeats C1_r2, C1_r3, B0_r2; P1b: W_b, W_c, W_d, W_e | `logs/sweep_p1b.out` |
| 03:28-04:38 | interleaved head-to-head rounds 1-3 (B0 / C1 / W_b / W_e); P2: H3, X3, K1 | `logs/sweep_h2h.out` |
| 04:39-04:45 | W_e (fastest single median) confirmation run and HF first-token control (kept as `_W_e`) | `logs/final_selected_W_e.*`, `logs/qual_first_token_control_W_e.log` |
| ~04:45-05:08 | W_e vs W_b are inside the 0.5 % tie band. User decision: **W_b**, because the BFP8 head is simpler and W_e breaks probe bars (prefill PCC 0.989, batch 70/72). Exact decision time is not logged | `selected_precision_config_W_b.json` |
| 04:59-05:08 | probe bars (prefill n=300/4100 + batch) per group: C1, X1, H1, E3, W_b pass; W_e 3 failed | `logs/probe_bars_*.log`, `probe_bars_by_group.log` |
| 05:13-05:32 | W_b confirmation chain. Tracker suite: **4 failed** `test_prefill_lengths[1, 31, 33, 65]` (e.g. n=31 PCC 0.9916 top-1 0.806 < 0.85). Qualitative: first token flips on prompts 3-5. Other steps exit 0 | `logs/confirm_W_b.out`, `*_W_b.*` |
| 05:19 | HF first-token control: the flipped prompts are HF near-ties (0.555/0.600/0.413 vs 0.297/0.364/0.365) | `logs/qual_first_token_control.log` |
| 05:34-05:36 | short-prompt screen by group: B0 and C1 4/4 pass; X1 fails n=65; E3 fails n=1, n=33 | `logs/probe_short_*.log`, `probe_short_by_group.log` |
| ~05:40 | user asked a Fable advisor. Recommendation: option 2 narrowed (drop X1, search the expert group C1_E2 → C1_E2_x → fallback C1). Recorded as `selection_rule_v2` (a) gate, (b) unchanged probe suite, (c) first-token parity | `thresholds.json` |
| 05:48-05:50 | C1_E2 probe screen: 8/8 prefill lengths pass | `logs/probe_screen_C1_E2.log` |
| 05:50-06:22 | C1_E2 sweep row; C1_E2 qualitative: first divergence 26/5/39/**0**/83/**0** → fails (c); head-to-head rounds 4-5 (C1_E2 / W_b / C1) | `logs/stage8_v2.out`, `logs/qual_C1_E2.*` |
| 06:35-06:38 | C1 qualitative: 27/5/102/57/38/21 (stage 7: 27/5/102/57/68/21) → **passes (c)** | `logs/qual_C1.*`, `qualitative_C1/` |
| ~06:40 | **selected C1** (median TF 103.70, n = 8). C1_E2_x skipped: C1_E2 is +0.14 % over C1, inside noise (advisor stop rule). The new `selected_precision_config.json` was written from `configs/C1.json` plus descriptive fields. W_b evidence was copied to `*_W_b` names | `selected_precision_config.json` |
| 06:39-07:00 | C1 confirmation chain (`stage8_confirm.sh`), all exit 0: precision test 2 passed; driver full run: prefill 98/100/100, TF 99/100/100 (103.76 t/s/u), token-out 107.97 t/s/u, TTFT128 65.7 ms, b4/b8/b32 13.63/19.09/26.10 ms, DRAM 23.22 GiB/chip, qual 27/5/102/57/38/21 (reproduces); tracker suite 28 passed; Tracy + `check_perf_rows` all_ok; weight cache 1 passed | `logs/final_selected.*`, `logs/test_*_selected.log`, `perf/perf_rows_check.json`, `qualitative/` |
| ~07:05 | replot (rule-v2 rejections drawn as orange rings), `context_contract.json` `datatype_sweep` block, README/work_log placeholders filled | `top1/top5_perf_pareto.png`, `../context_contract.json` |
| ~07:10 | stage review round 1 (independent Fable subagent): **more-work-needed**, P2 = expert weight_groups described as one shared copy (code keeps decode stacked + prefill dispatch copies); fixed in `selected_precision_config.json`, other concerns added to README Limitations | `session_state/stage8_review_round1.md` (outside repo) |
| ~07:20 | stage review round 2: **clean-pass** | `session_state/stage8_review_round2.md` |

## Notes

- Build-to-build TF noise is about 1 % (C1 over 8 rows: 102.86-104.93). Interleaved head-to-head rows were added to
  separate the configs; the rounds ran in the same fixed order each time.
- Weight caches: 11 keys, ~395 GB under `~/.cache/tt_autoport_weights/lfm2_5_24b_a2b`.
- Every sweep row passes the accuracy gate, so the selection is decided by speed and by rule-v2 (b) and (c).
