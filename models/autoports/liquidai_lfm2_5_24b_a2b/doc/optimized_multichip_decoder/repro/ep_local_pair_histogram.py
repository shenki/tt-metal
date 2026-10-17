# Stage 5 review P2 (plan 5.1.3): share of a 2048-token chunk's top-4 pairs that land on each chip's 16 experts
# (EP prefill dispatch), from the HF router on real hidden states (tests/gen_real_hidden.py output).  A capped
# local buffer is exact only if the max share never exceeds the cap.
#   python ep_local_pair_histogram.py
import torch

from models.autoports.liquidai_lfm2_5_24b_a2b.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.liquidai_lfm2_5_24b_a2b.tests import gen_real_hidden
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_state_dict

hidden_all = torch.load(str(gen_real_hidden.OUT))["hidden"]
cfg = hf_config()
for layer in (3, 15, 31, 38):
    hl, rope = build_hf_layer(cfg, layer, layer_state_dict(layer, True))
    ff = hl.feed_forward
    shares = []
    for p in range(hidden_all[layer].shape[0]):
        x = hidden_all[layer][p].float()
        cap = {}
        hk = ff.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach().reshape(-1, a[0].shape[-1])))
        with torch.no_grad():
            hf_layer_forward(hl, rope, x.unsqueeze(0))
        hk.remove()
        idx = torch.topk(torch.sigmoid(cap["h"] @ ff.gate.weight.T) + ff.expert_bias, 4, dim=-1).indices  # [S, 4]
        per_chip = torch.bincount(idx.flatten() // 16, minlength=4).float() / idx.numel()
        shares.append(per_chip)
    s = torch.stack(shares)
    print(
        f"EPSHARE layer {layer}: per-prompt per-chip pair share (ideal 0.25): max {s.max():.3f} min {s.min():.3f} "
        f"mean-of-max {s.max(1).values.mean():.3f}; tokens/prompt {hidden_all[layer].shape[1]}",
        flush=True,
    )
