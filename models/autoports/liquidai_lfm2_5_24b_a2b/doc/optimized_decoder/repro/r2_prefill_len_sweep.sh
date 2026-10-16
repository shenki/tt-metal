#!/bin/bash
# BASELINE vs DEFAULT vs candidate single-chunk prefill at short/boundary lengths (conv_moe, attn_moe).
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
D=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder
export LFM2_SWEEP_OUT=${LFM2_SWEEP_OUT:-$D/sweeps/r2_prefill_len_sweep.jsonl}
export LFM2_CAND=${LFM2_CAND:-'{"sdpa_prefill_chunk":64,"sdpa_prefill_k_chunk":512,"sdpa_prefill_fidelity":"hifi2","moe_scan_multicore":true,"moe_scatter_rows":32,"moe_out_direct_untilize":true,"prefill_mm_v2":true}'}
for kind in ${KINDS:-conv_moe attn_moe}; do for pol in ${POLS:-baseline default candidate}; do
  echo "== $kind $pol"; timeout 2400 python $D/repro/r2_prefill_len_sweep.py $kind $pol $LENS > $D/sweeps/r2_prefill_len_${kind}_${pol}.log 2>&1; echo rc=$?
done; done
