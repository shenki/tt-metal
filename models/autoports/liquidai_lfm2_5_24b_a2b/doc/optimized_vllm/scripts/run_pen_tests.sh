#!/bin/bash
# usage: run_pen_tests.sh <outdir>   (env LFM2_DEVICE_PENALTIES etc. passed through) -- serve, run 2 plugin tests x2, stop
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
OUT=$1; mkdir -p $OUT; export LFM2_VLLM_EVENTS=$OUT/events.jsonl; rm -f $LFM2_VLLM_EVENTS
INTERLEAVE=1 /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/scratchpad_tools/serve_lfm2.sh $OUT/server.log & SP=$!
for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $SP 2>/dev/null || exit 2; sleep 5; done
P=/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin
for r in 1 2; do
  (cd $P && python -m pytest -q tests/tt/test_tt_penalties.py::TestRepetitionPenalty::test_repetition_penalty_mixed_batch "tests/tt/test_seeding_and_variety.py::TestSeedingAndVariety::test_topk" \
     --tt-server-url=http://localhost:8000 --tt-model-name=LiquidAI/LFM2.5-24B-A2B --tt-max-num-seqs=32 -p no:cacheprovider) > $OUT/tests_r$r.log 2>&1
  echo "round $r: $(grep -E 'passed|failed' $OUT/tests_r$r.log | tail -1)"
done
kill -INT $SP; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done; kill -0 $SP 2>/dev/null && kill -TERM $SP
sleep 3; pgrep -af "EngineCore|vllm.entrypoints" | grep -v pgrep || echo "no vllm left"
