#!/bin/bash
# Stage-10 review round 1 fixes (vLLM-ordered device penalties, neutral released slots): rerun the affected evidence.
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
T=/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/scratchpad_tools
D=$PWD/$M/doc/optimized_vllm
LFM2_VLLM_LAYERS=all timeout 3600 pytest -q --timeout=0 $M/tests/test_generator_vllm.py > $D/logs/adapter_tests_full_r6.log 2>&1; echo "full tests exit $? $(grep -E 'passed|failed' $D/logs/adapter_tests_full_r6.log | tail -1)"
mv $D/tracked $D/superseded_r1_tracked 2>/dev/null; mv $D/penalty_parity $D/superseded_r1_penalty_parity 2>/dev/null
QUICK=1 INTERLEAVE=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $T/run_server_bench.sh $D/tracked > $D/tracked_driver.log 2>&1; echo "tracked done: $(tail -1 $D/tracked_driver.log)"
mkdir -p $D/penalty_parity; export LFM2_VLLM_EVENTS=$D/penalty_parity/events.jsonl
INTERLEAVE=1 $T/serve_lfm2.sh $D/penalty_parity/server.log & SP=$!
for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_penalty_parity --out $D/penalty_parity > $D/penalty_parity/penalty_parity.log 2>&1; echo "parity exit $?"
kill -INT $SP; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done; kill -0 $SP 2>/dev/null && kill -TERM $SP; sleep 3
unset LFM2_VLLM_EVENTS
mkdir -p $PWD/$M/readiness_vllm/superseded_s10r1; for f in sampling_tests.log server.log vllm_benchmark.json vllm_benchmark.log vllm_ci_serving_benchmark.json vllm_ci_serving_benchmark.log vllm_ci_serving_result.json vllm_qualitative_outputs.json vllm_result.json; do mv $PWD/$M/readiness_vllm/$f $PWD/$M/readiness_vllm/superseded_s10r1/ 2>/dev/null; done
$T/runner_final.sh runner_final_r2 > $D/logs/runner_final_r2_driver.log 2>&1; cat $D/logs/runner_final_r2_driver.log
pgrep -af "EngineCore|vllm.entrypoints|run_vllm_server" | grep -v pgrep | awk '{print $1}' | while read p; do kill -INT $p; done; sleep 10
pgrep -af "EngineCore|vllm.entrypoints|run_vllm_server" | grep -v pgrep || echo "no vllm left"
echo ALL DONE
