#!/bin/bash
# Stage-10 final runner evidence: one runner-launched server (--stages serve, held) + attached sampling (full profile)
# and qualitative,benchmark stages; then SIGINT to the exact runner PID.  usage: runner_final.sh <tag>
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
export TT_MODEL_CLASS_OVERRIDES="TTLfm2MoeForCausalLM=models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm:TTLfm2MoeForCausalLM"
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin HF_HUB_OFFLINE=1
TAG=$1; L=$PWD/$M/doc/optimized_vllm/logs
TTC='{"trace_region_size": 805306368, "l1_small_size": 24576, "prefill_chunk_alignment": 2048, "decode_interleave_prefill_steps": 1}'
S=(python -m readiness_check.run_vllm_server --stages serve --model-dir $M --hf-model LiquidAI/LFM2.5-24B-A2B --mesh-device P300x2 --max-num-seqs 32 --max-model-len 131072 --tt-config "$TTC")
printf '%q ' "${S[@]}" > $L/${TAG}_serve_cmd.txt
"${S[@]}" > $L/${TAG}_serve.log 2>&1 &
RP=$!
for i in $(seq 1 300); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $RP 2>/dev/null || { echo "runner died"; exit 2; }; sleep 5; done
echo "server up ~$((i*5)) s"
A=(python -m readiness_check.run_vllm_server --stages sampling --server-url http://localhost:8000 --model-dir $M --hf-model LiquidAI/LFM2.5-24B-A2B --max-num-seqs 32 --sampling-profile full)
printf '%q ' "${A[@]}" > $L/${TAG}_sampling_cmd.txt
"${A[@]}" > $L/${TAG}_sampling.log 2>&1; echo "sampling exit $?"
B=(python -m readiness_check.run_vllm_server --stages qualitative,benchmark --server-url http://localhost:8000 --model-dir $M --hf-model LiquidAI/LFM2.5-24B-A2B --max-num-seqs 32 --max-model-len 131072 --additional-benchmark-args "--repetition-penalty 1.0")
printf '%q ' "${B[@]}" > $L/${TAG}_qual_bench_cmd.txt
"${B[@]}" > $L/${TAG}_qual_bench.log 2>&1; echo "qual_bench exit $?"
kill -INT $RP; for i in $(seq 1 90); do kill -0 $RP 2>/dev/null || break; sleep 2; done
kill -0 $RP 2>/dev/null && echo "runner still alive"
pgrep -af "EngineCore|vllm.entrypoints|run_vllm_server" | grep -v pgrep || echo "no vllm processes left"
