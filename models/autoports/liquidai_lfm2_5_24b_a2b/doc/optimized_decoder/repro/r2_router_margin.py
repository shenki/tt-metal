# CPU: router top-k margins for a user's decode tokens (HF fp32 reference vs bf16-rounded router input).
import sys

import torch

from models.autoports.liquidai_lfm2_5_24b_a2b.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_inputs, layer_state_dict
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.test_functional_decoder import _real_embed

layer_idx, n, steps, seed = (int(a) for a in sys.argv[1:5])
cfg = hf_config()
sd = layer_state_dict(layer_idx, True)
embed = _real_embed()
layer, rope = build_hf_layer(cfg, layer_idx, sd)
x = layer_inputs(n + steps, cfg.hidden_size, seed, embed).to(torch.bfloat16).float()
cap = {}
layer.feed_forward.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach().clone()))
out = hf_layer_forward(layer, rope, x.unsqueeze(0))[0]
ff = layer.feed_forward
h = cap["h"].reshape(-1, cfg.hidden_size)
for name, hh, w in (("fp32", h, ff.gate.weight), ("bf16 in", h.bfloat16().float(), ff.gate.weight.bfloat16().float())):
    s = torch.sigmoid(hh @ w.T) + ff.expert_bias
    top = torch.topk(s, 5, dim=-1)
    for t in range(n - 2, n + steps):
        v, i = top.values[t], top.indices[t]
        print(f"{name:7s} pos {t}: top5 {i.tolist()} margin(4th-5th) {(v[3]-v[4]).item():.2e}")
upd = out - x
print("update norm per decode pos:", [f"{upd[t].norm().item():.2f}" for t in range(n, n + steps)])
