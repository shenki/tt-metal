import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_inputs, local_model_path
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import MoE
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import FusedMoE
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import Lfm2MoeArgs
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.weights import load_layer_state_dict, load_tensors

p = local_model_path()
args = Lfm2MoeArgs.from_hf_config(hf_config())
emb = load_tensors(p, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
for li in (3, 20):
    sd = load_layer_state_dict(p, li)
    x = layer_inputs(512, 2048, 7, emb).float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * sd["ffn_norm.weight"].float()).bfloat16().float()
    s = torch.sigmoid(x @ sd["feed_forward.gate.weight"].float().T)
    idx = (s + sd["feed_forward.expert_bias"].float()).topk(4).indices
    w = s.gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-6)
    ref = torch.zeros_like(s).scatter(-1, idx, w)
    tx = ttnn.from_torch(x.reshape(1, 1, 512, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    for name, mod in [
        ("router_fold", FusedMoE(md, args, sd, ttnn.bfloat16, frozenset({"router_fold"}))),
        ("functional", MoE(md, args, sd, ttnn.bfloat16)),
        ("moe_grouped_topk", FusedMoE(md, args, sd, ttnn.bfloat16, frozenset({"moe_grouped_topk"}))),
        ("router_tile_mask", FusedMoE(md, args, sd, ttnn.bfloat16, frozenset({"router_tile_mask", "router_fold"}))),
    ]:
        r = ttnn.to_torch(mod.route(tx)).float().reshape(512, 64)
        sel_diff = ((r > 0) != (ref > 0)).any(-1).sum().item()
        same = (r > 0) == (ref > 0)
        werr = (r - ref)[(ref > 0) & (r > 0)].abs().max().item()
        print(
            f"PROBE layer {li} {name:18s}: rows with different expert set {sel_diff}/512, max |weight err| on agreed experts {werr:.5f}"
        )
ttnn.close_mesh_device(md)
