# Benchmark: LiquidAI/LFM2.5-24B-A2B

Status: completed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| gpqa_diamond_cot_zeroshot | 128 / 198 | exact_match,flexible-extract | 70.31 | 74.42 | -4.11 | [reference](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| gpqa_diamond_cot_zeroshot | 128 / 198 | exact_match,strict-match | 0.00 | — | — | Unavailable |
| ifeval | 256 / 541 | prompt_level_strict_acc,none | 97.27 | 96.75 | +0.52 | [reference](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| ifeval | 256 / 541 | inst_level_strict_acc,none | 98.25 | — | — | Unavailable |
| ifeval | 256 / 541 | prompt_level_loose_acc,none | 98.83 | — | — | Unavailable |
| ifeval | 256 / 541 | inst_level_loose_acc,none | 99.25 | — | — | Unavailable |
| ifeval | 256 / 541 | ifeval_mean_four | 98.40 | — | — | Unavailable |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 321.52 | 9.34 | 107.08 | 84.87 | 5.20 | 14.42 |
| 32 users | 32 | 32 | 4096 / 128 | 2367.88 | 94.90 | 10.54 | 279.29 | 4.69 | 37.94 |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for both serving profiles; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

Reference protocol notes:
- gpqa_diamond_cot_zeroshot, exact_match,flexible-extract: Model card / blog (https://www.liquid.ai/blog/lfm2-5-24b-a2b): vLLM, temperature 0.6, max output tokens 32768, thinking always on; harness, prompt, answer extraction, top_k/top_p and number of runs not stated; full 198-question Diamond set (this run: frozen 128-question subset, single sample, lm-eval 0.4.13 CoT zero-shot prompt).
- ifeval, prompt_level_strict_acc,none: Model card / blog: vLLM, temperature 0.6, max output tokens 32768; the source does not name the IFEval metric (compared here against prompt-level strict, the most common headline; all four upstream metrics are reported). Full 541 prompts at source; this run: frozen 256-prompt subset.

## Run details

Subset: `3dd6b5fe3878c8138817180626bf5f551ae6a966d11777d3eac935db695fcc38`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [gpqa_diamond_cot_zeroshot](gpqa_diamond_cot_zeroshot/results.json) | 128 | 10 (7.8%) | 10 | 2305.2 |
| [ifeval](ifeval/results.json) | 256 | 0 (0.0%) | 0 | 2305.2 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 12.06 | 0.66 |
| [32](perf-b32.json) | 96 / 96 | 44.00 | 2.18 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 321.52 | 320.74 | 326.10 | 327.50 |
| 1 | TPOT | 9.34 | 9.33 | 9.36 | 9.37 |
| 1 | ITL | 9.34 | 9.37 | 9.46 | 9.57 |
| 1 | E2EL | 1507.55 | 1506.13 | 1513.79 | 1513.81 |
| 32 | TTFT | 2367.88 | 643.56 | 9806.57 | 11250.64 |
| 32 | TPOT | 94.90 | 107.16 | 110.83 | 110.94 |
| 32 | ITL | 94.90 | 26.98 | 215.68 | 224.73 |
| 32 | E2EL | 14420.51 | 14433.86 | 23753.04 | 25290.18 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 31.3872 s. useful prompt FLOPs from executed prefill rows x HF config (2 x active matmul params + causal attention + LM head once per request) Peak: tt-perf-report ArchitectureSpec 'blackhole' (110 worker cores, 4096 FLOP/cycle/core at 1.35 GHz, HiFi2 = /2; DRAM 512 GB/s per chip) x 4 chips. Timing: adapter call records (CLOCK_MONOTONIC) in the measured window; each interval from a call's start to the next call's start charged to that call's phase (non-overlapping, includes host gaps, CCL, sampling and readback waits). [Evidence](roofline-b32-evidence.json).
- Concurrency 32, decode: 12.5683 s. per-step DRAM bytes at C1 stored dtypes over 4 chips: weights (experts: compact bucket x 4, else expected union of active rows' top-4), KV reads/writes at the logged positions, ShortConv state Peak: tt-perf-report ArchitectureSpec 'blackhole' (110 worker cores, 4096 FLOP/cycle/core at 1.35 GHz, HiFi2 = /2; DRAM 512 GB/s per chip) x 4 chips. Timing: adapter call records (CLOCK_MONOTONIC) in the measured window; each interval from a call's start to the next call's start charged to that call's phase (non-overlapping, includes host gaps, CCL, sampling and readback waits). [Evidence](roofline-b32-evidence.json).
- Concurrency 1, prefill: 2.35809 s. useful prompt FLOPs from executed prefill rows x HF config (2 x active matmul params + causal attention + LM head once per request) Peak: tt-perf-report ArchitectureSpec 'blackhole' (110 worker cores, 4096 FLOP/cycle/core at 1.35 GHz, HiFi2 = /2; DRAM 512 GB/s per chip) x 4 chips. Timing: adapter call records (CLOCK_MONOTONIC) in the measured window; each interval from a call's start to the next call's start charged to that call's phase (non-overlapping, includes host gaps, CCL, sampling and readback waits). [Evidence](roofline-b1-evidence.json).
- Concurrency 1, decode: 9.68869 s. per-step DRAM bytes at C1 stored dtypes over 4 chips: weights (experts: compact bucket x 4, else expected union of active rows' top-4), KV reads/writes at the logged positions, ShortConv state Peak: tt-perf-report ArchitectureSpec 'blackhole' (110 worker cores, 4096 FLOP/cycle/core at 1.35 GHz, HiFi2 = /2; DRAM 512 GB/s per chip) x 4 chips. Timing: adapter call records (CLOCK_MONOTONIC) in the measured window; each interval from a call's start to the next call's start charged to that call's phase (non-overlapping, includes host gaps, CCL, sampling and readback waits). [Evidence](roofline-b1-evidence.json).

Total client-stage wall time: 2614.1 seconds.
