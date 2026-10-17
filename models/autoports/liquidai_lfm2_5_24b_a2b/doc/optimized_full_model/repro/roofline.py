# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM roofline of one batch-1 decode token per chip (TP = 4): bytes of every weight / cache tensor the step reads,
from the HF config and MC_POLICY dtypes, / 512 GB/s (tt-perf-report's Blackhole DRAM bandwidth).  Decode at b1 uses the
compact MoE path: num_experts_per_tok experts x 1/4 of each (TP over the intermediate dim).  KV read at position P.
Usage: python roofline.py [P=192] -> JSON on stdout."""
import json
import sys

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import Lfm2MoeArgs, load_hf_config
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import MC_POLICY

P = int(sys.argv[1]) if len(sys.argv) > 1 else 192
B = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
a = Lfm2MoeArgs.from_hf_config(load_hf_config())
tp, H, I, K = 4, a.hidden_size, a.moe_intermediate_size // 4, a.num_experts_per_tok
pol = MC_POLICY
hd, nq, nkv = a.head_dim, a.num_heads // tp, a.num_kv_heads // tp
dense_i = 3072  # per-chip dense intermediate padded (ds_dense_w13 / ds_dense_w2_decode)
attn = (
    (H * (nq + 2 * nkv) * hd) * B[pol.attn_qkv] + (nq * hd * H) * B[pol.attn_out] + 2 * nkv * P * hd * B[pol.kv_cache]
)
conv = H * 3 * (H // tp) * B[pol.conv_in] + (H // tp) * H * B[pol.conv_out]
moe = H * 256 * 2.0 + K * (
    H * 2 * I * B[pol.expert_gate_up] + I * H * B[pol.expert_down]
)  # router (bf16, N padded) + experts
dense = H * 2 * dense_i * B[pol.dense_w13] + dense_i * H * B[pol.dense_w2]
head = H * 32256 * B["bfp8"]
n_attn = sum(a.is_attention_layer(i) for i in range(a.num_layers))
n_dense = a.num_dense_layers
n_conv_moe = a.num_layers - n_attn - n_dense
total = n_attn * (attn + moe) + n_conv_moe * (conv + moe) + n_dense * (conv + dense) + head
out = {
    "position": P,
    "bytes_per_chip": {
        "attn_moe_layer": attn + moe,
        "conv_moe_layer": conv + moe,
        "conv_dense_layer": conv + dense,
        "lm_head": head,
        "total": total,
    },
    "layer_counts": {"attn_moe": n_attn, "conv_moe": n_conv_moe, "conv_dense": n_dense},
    "dram_gb_s": 512,
    "roofline_ms_per_token": total / 512e9 * 1e3,
}
print(json.dumps(out, indent=1))
