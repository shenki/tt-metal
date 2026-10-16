# CPU: effect on decode update PCC of swapping the 4th-ranked expert for the 5th at one position.
import sys

import torch

from models.autoports.liquidai_lfm2_5_24b_a2b.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_inputs, layer_state_dict
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.test_functional_decoder import _real_embed

layer_idx, n, steps, seed, flip_pos = (int(a) for a in sys.argv[1:6])
cfg = hf_config()
sd = layer_state_dict(layer_idx, True)
embed = _real_embed()
layer, rope = build_hf_layer(cfg, layer_idx, sd)
x = layer_inputs(n + steps, cfg.hidden_size, seed, embed).to(torch.bfloat16).float()
ref = hf_layer_forward(layer, rope, x.unsqueeze(0))[0]
ff = layer.feed_forward
orig = ff.route_tokens_to_experts


def flipped(logits):
    sel, w = orig(logits)
    s = logits.sigmoid() + ff.expert_bias
    top5 = torch.topk(s, 5, dim=-1).indices
    sel = sel.clone()
    sel[flip_pos, 3] = top5[flip_pos, 4]
    rw = torch.gather(logits.sigmoid(), 1, sel)
    rw = rw / (rw.sum(-1, keepdim=True) + 1e-6)
    return sel, rw


ff.route_tokens_to_experts = flipped
alt = hf_layer_forward(layer, rope, x.unsqueeze(0))[0]
pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
d = slice(n, n + steps)
print(
    f"flip at {flip_pos}: decode output PCC {pcc(alt[d], ref[d]):.6f} update PCC {pcc((alt - x)[d], (ref - x)[d]):.6f}"
)
