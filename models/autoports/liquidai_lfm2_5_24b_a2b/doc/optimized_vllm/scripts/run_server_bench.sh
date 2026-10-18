#!/bin/bash
# Launch serve_lfm2.sh, wait for /health, run bench_set.sh, stop the server (SIGINT to the exact api_server PID).
# usage: run_server_bench.sh <outdir> [extra vllm args...]   (env passed through: ALIGN, INTERLEAVE, LFM2_*, TT_METAL_*)
T=/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/scratchpad_tools
OUT=$1; shift; mkdir -p $OUT
export LFM2_VLLM_EVENTS=$OUT/events.jsonl; rm -f $LFM2_VLLM_EVENTS
$T/serve_lfm2.sh $OUT/server.log "$@" &
SP=$!
for i in $(seq 1 240); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 $OUT/server.log; exit 2; }
  sleep 5
done
curl -sf localhost:8000/health >/dev/null || { echo "server not healthy"; kill -INT $SP; exit 2; }
echo "server up after ~$((i*5)) s"
$T/bench_set.sh $OUT $LFM2_VLLM_EVENTS
kill -INT $SP; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 2; done
kill -0 $SP 2>/dev/null && { echo "server still alive, SIGTERM"; kill -TERM $SP; sleep 10; }
pgrep -af "EngineCore|vllm.entrypoints" | grep -v pgrep || echo "no vllm processes left"
