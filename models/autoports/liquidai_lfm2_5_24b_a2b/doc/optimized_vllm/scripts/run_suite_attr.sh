#!/bin/bash
# usage: run_suite_attr.sh <outdir>  -- serve (production generation config, INTERLEAVE=1, events), per-test suite attribution, stop
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
OUT=$1; mkdir -p $OUT; export LFM2_VLLM_EVENTS=$OUT/events.jsonl VLLM_TT_PLUGIN_ROOT=/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin; rm -f $LFM2_VLLM_EVENTS
INTERLEAVE=1 /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/scratchpad_tools/serve_lfm2.sh $OUT/server.log & SP=$!
for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $SP 2>/dev/null || exit 2; sleep 5; done
python /tmp/claude-1000/-home-ttuser-liquid-LFM2-5-24B-A2B/6c482fbf-6561-4b4d-8dc8-6b24c39ee2f2/scratchpad/suite_paths.py --events $OUT/events.jsonl --out $OUT/suite_sampling_paths.json --fail-dir $OUT/failures > $OUT/suite.log 2>&1; echo "suite exit $?"
kill -INT $SP; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done; kill -0 $SP 2>/dev/null && kill -TERM $SP
sleep 3; pgrep -af "EngineCore|vllm.entrypoints" | grep -v pgrep || echo "no vllm left"
