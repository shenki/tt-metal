# CPU: real MoE router inputs (the ffn_norm output, captured with a forward pre-hook on the HF
# feed_forward) for several MoE layers, from generated/lfm2_real_hidden.pt (book text, 4 x 400 tokens).
import torch

from models.autoports.lfm2_r3_route.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.lfm2_r3_route.tests.harness import hf_config, layer_state_dict

LAYERS = [2, 3, 7, 15, 23, 31, 39]
hidden = torch.load("generated/lfm2_real_hidden.pt")["hidden"]
cfg = hf_config()
out = {}
for li in LAYERS:
    layer, rope = build_hf_layer(cfg, li, layer_state_dict(li, True))
    cap = []
    h = layer.feed_forward.register_forward_pre_hook(
        lambda m, a: cap.append(a[0].detach().reshape(-1, a[0].shape[-1]).clone())
    )
    with torch.no_grad():
        hf_layer_forward(layer, rope, hidden[li].float())
    h.remove()
    out[li] = cap[0].to(torch.bfloat16)
    print(li, out[li].shape, flush=True)
    del layer
torch.save(out, "generated/r3_route_hn.pt")
