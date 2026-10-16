#!/bin/bash
# Tracy matrix for the real-input passing candidates (r3_bfp4_tracy_pol.sh per kind x policy x batch)
cd $(dirname $0)
declare -A P=(
 [default]='{}'
 [e_gu4_dec_hifi2]='{"expert_gate_up_decode": "bfp4"}'
 [e_gu4_dec_lofi]='{"expert_gate_up_decode": "bfp4", "expert_fidelity": "lofi"}'
 [a_bfp8_lofi]='{"attn_fidelity": "lofi"}'
 [combo_e_a]='{"expert_gate_up_decode": "bfp4", "expert_fidelity": "lofi", "attn_fidelity": "lofi"}'
 [c_bfp8_lofi]='{"conv_fidelity": "lofi"}'
 [c_in4]='{"conv_in": "bfp4"}'
 [d_bfp8_lofi]='{"dense_fidelity": "lofi"}'
 [combo_e_a_convlofi]='{"expert_gate_up_decode": "bfp4", "expert_fidelity": "lofi", "attn_fidelity": "lofi", "conv_fidelity": "lofi"}'
)
declare -A K=(
 [attn_moe]="default e_gu4_dec_hifi2 e_gu4_dec_lofi a_bfp8_lofi combo_e_a"
 [conv_moe]="default e_gu4_dec_hifi2 e_gu4_dec_lofi c_bfp8_lofi c_in4 combo_e_a_convlofi"
 [conv_dense]="default c_bfp8_lofi d_bfp8_lofi c_in4"
)
for kind in attn_moe conv_moe conv_dense; do
  for tag in ${K[$kind]}; do
    for bb in "1 1" "4 1" "32 0"; do set -- $bb
      ./r3_bfp4_tracy_pol.sh $kind $1 $tag "${P[$tag]}" $2
    done
  done
done
echo ALLDONE
