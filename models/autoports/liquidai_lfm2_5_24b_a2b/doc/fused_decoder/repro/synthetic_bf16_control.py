"""CPU control: HF Lfm2MoeDecoderLayer in bf16 vs fp32 on the synthetic batch32 conv_moe inputs.

    LFM2_REAL_WEIGHTS=0 python models/autoports/liquidai_lfm2_5_24b_a2b/doc/fused_decoder/repro/synthetic_bf16_control.py
"""
import torch

from models.autoports.liquidai_lfm2_5_24b_a2b.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import (
    hf_config,
    layer_inputs,
    pcc,
    synthetic_layer_state_dict,
)

cfg = hf_config()
li = 3
sd = synthetic_layer_state_dict(li)
f32, rope = build_hf_layer(cfg, li, sd, dtype=torch.float32)
b16, _ = build_hf_layer(cfg, li, sd, dtype=torch.bfloat16)
lens = [1 + (37 * i) % 200 for i in range(32)]
worst = []
for i in range(32):
    n = lens[i]
    x = layer_inputs(n + 3, 2048, 100 + i, None).to(torch.bfloat16).float()
    r = hf_layer_forward(f32, rope, x[None])[0]
    with torch.no_grad():
        pos = torch.arange(n + 3)[None]
        y = b16(
            x[None].bfloat16(), position_embeddings=rope(x[None].bfloat16(), pos), attention_mask=None, position_ids=pos
        )
        y = y[0].float()
    worst.append((pcc(y[n:] - x[n:], r[n:] - x[n:]), pcc(y[:n] - x[:n], r[:n] - x[:n]), i))
worst.sort()
print(
    "HF-bf16 vs HF-fp32 update PCC (decode, prefill, slot), worst 4:",
    [(round(a, 4), round(b, 4), s) for a, b, s in worst[:4]],
)
