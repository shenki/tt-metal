#!/bin/bash
# One-change-at-a-time ablation over DEFAULT at a few lengths (traced replay), attn_moe + conv_moe.
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
D=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder
export LFM2_SWEEP_OUT=$D/sweeps/r2_prefill_ablate.jsonl
declare -A V=(
 [scan]='{"moe_scan_multicore":true}'
 [scatter32]='{"moe_scatter_rows":32}'
 [untilize]='{"moe_out_direct_untilize":true}'
 [mmv2]='{"prefill_mm_v2":true}'
 [sdpa_q64k512_hifi2]='{"sdpa_prefill_chunk":64,"sdpa_prefill_k_chunk":512,"sdpa_prefill_fidelity":"hifi2"}'
 [sdpa_q64k512_hifi4]='{"sdpa_prefill_chunk":64,"sdpa_prefill_k_chunk":512}'
 [sdpa_q128k512_hifi2]='{"sdpa_prefill_chunk":128,"sdpa_prefill_k_chunk":512,"sdpa_prefill_fidelity":"hifi2"}'
)
for kind in ${KINDS:-attn_moe conv_moe}; do for name in ${NAMES:-${!V[@]}}; do
  case $kind:$name in conv_moe:sdpa*) continue;; esac
  echo "== $kind $name"; LFM2_CAND="${V[$name]}" timeout 1800 python $D/repro/r2_prefill_len_sweep.py $kind $name ${LENS:-1 65 97 2048} > $D/sweeps/r2_prefill_ablate_${kind}_${name}.log 2>&1; echo rc=$?
done; done
