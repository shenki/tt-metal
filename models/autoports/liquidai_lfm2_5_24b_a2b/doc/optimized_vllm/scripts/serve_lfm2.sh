#!/bin/bash
# Launch the LFM2 vLLM server exactly like readiness_check.run_vllm_server (_launch_server) for iteration.
# usage: serve_lfm2.sh <log> [extra vllm args...]   env: LFM2_VLLM_LAYERS (reduced target), LFM2_VLLM_EVENTS, MAXLEN,
#   MAXSEQS, PORT, ALIGN (tt prefill_chunk_alignment, 2048), INTERLEAVE (tt decode_interleave_prefill_steps, plugin default)
source /home/ttuser/liquid-LFM2.5-24B-A2B/session_state/env.sh >/dev/null
export TT_MODEL_CLASS_OVERRIDES="TTLfm2MoeForCausalLM=models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm:TTLfm2MoeForCausalLM"
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/liquid-LFM2.5-24B-A2B/vllm-tt-plugin HF_HUB_OFFLINE=1
export MESH_DEVICE=P300x2 HF_MODEL=LiquidAI/LFM2.5-24B-A2B VLLM_RPC_TIMEOUT=300000
LOG=$1; shift
TT_CFG="\"sample_on_device_mode\": \"all\", \"trace_region_size\": 805306368, \"l1_small_size\": 24576, \"prefill_chunk_alignment\": ${ALIGN:-2048}"
[ -n "$INTERLEAVE" ] && TT_CFG="$TT_CFG, \"decode_interleave_prefill_steps\": $INTERLEAVE"
exec python -m vllm.entrypoints.openai.api_server --model LiquidAI/LFM2.5-24B-A2B --block_size 64 --max_num_seqs ${MAXSEQS:-32} \
  --port ${PORT:-8000} --max_model_len ${MAXLEN:-131072} \
  --additional-config "{\"tt\": {$TT_CFG}}" \
  "$@" > "$LOG" 2>&1
