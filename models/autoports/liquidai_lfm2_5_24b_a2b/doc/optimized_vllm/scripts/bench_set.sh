#!/bin/bash
# Stage-10 before/after workload set against a running server (port 8000).  usage: bench_set.sh <outdir> <events.jsonl>
# Same vllm bench serve flags as readiness_check.run_vllm_server (single-user 128/128/1 and CI burst 100/100/32), plus
# n=5 sequential repeats and sampling-mode variants (greedy / top_k 20 / top_k 20 + rep 1.05 / checkpoint defaults).
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
OUT=$1; EV=$2; mkdir -p $OUT
B="vllm bench serve --backend vllm --model LiquidAI/LFM2.5-24B-A2B --base-url http://localhost:8000 --endpoint /v1/completions --dataset-name random --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result"
mark() { echo "{\"t\": $(date +%s.%N), \"kind\": \"bench_mark\", \"label\": \"$1\"}" >> $EV; }
run() { local name=$1; shift; mark "start:$name"; $B "$@" --result-filename $OUT/$name.json > $OUT/$name.log 2>&1; echo "$name exit $?"; mark "end:$name"; }
if [ -n "$QUICK" ]; then  # matrix subset: single-user greedy + penalty, CI burst, mixed shape, prefix
run warm_greedy_128_128_1      --random-input-len 128 --random-output-len 128 --num-prompts 1 --max-concurrency 1 --temperature 0.0 --repetition-penalty 1.0
run greedy_128_128_n5          --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0.0 --repetition-penalty 1.0
run k20_rep105_128_128_n5      --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0.1 --top-k 20 --repetition-penalty 1.05
run ci_100_100_32              --random-input-len 100 --random-output-len 100 --num-prompts 32 --temperature 0.0 --repetition-penalty 1.0
else
run warm_greedy_128_128_1      --random-input-len 128 --random-output-len 128 --num-prompts 1 --max-concurrency 1 --temperature 0.0 --repetition-penalty 1.0
run greedy_128_128_1           --random-input-len 128 --random-output-len 128 --num-prompts 1 --max-concurrency 1 --temperature 0.0 --repetition-penalty 1.0
run greedy_128_128_n5          --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0.0 --repetition-penalty 1.0
run k20_128_128_n5             --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0.1 --top-k 20 --repetition-penalty 1.0
run k20_rep105_128_128_n5      --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0.1 --top-k 20 --repetition-penalty 1.05
run genconfig_128_128_n5       --random-input-len 128 --random-output-len 128 --num-prompts 5 --max-concurrency 1
run ci_100_100_32              --random-input-len 100 --random-output-len 100 --num-prompts 32 --temperature 0.0 --repetition-penalty 1.0
run ci_k20_rep105_100_100_32   --random-input-len 100 --random-output-len 100 --num-prompts 32 --temperature 0.1 --top-k 20 --repetition-penalty 1.05
run ci_genconfig_100_100_32    --random-input-len 100 --random-output-len 100 --num-prompts 32
fi
mark "start:mixed_shape"
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_mixed_shape_check --events $EV --out $OUT/mixed_shape > $OUT/mixed_shape.log 2>&1; echo "mixed_shape exit $?"
mark "start:prefix"
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_prefix_ab --label $(basename $OUT) --tag "$(basename $OUT)-$(date +%s)" --concurrent --out $OUT/prefix_concurrent.json > $OUT/prefix_concurrent.log 2>&1; echo "prefix exit $?"
mark "end:all"
