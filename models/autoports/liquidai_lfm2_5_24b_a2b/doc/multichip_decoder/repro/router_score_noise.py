# Stage 4 router-score noise: decode router scores (sigmoid, fp32) of the multichip decoder vs the stage-3 single-chip
# decoder (replicated on the same mesh, unified dispatch FFN like the multichip one) vs HF fp32, same inputs.
# Both TT decoders run eager decode; the scores are captured from MoE._router_scores (host copy, diagnostic only).
# Reports max / p99 |score - HF score| over every active decode row, and the selection disagreements vs HF.
#   python router_score_noise.py [layer]
import json
import os
import sys

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import (
    DecodeInputs,
    PagedLayout,
    User,
    decode_step,
    hf_config,
    layer_inputs,
    layer_state_dict,
    run_prefill,
)
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.test_functional_decoder import _real_embed
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import TILE, Lfm2MoeArgs
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import (
    DEFAULT_MESH_PLAN,
    MC_POLICY,
    make_multichip,
    mesh_device_params,
)
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    OptimizedMoE,
    make_optimized,
    replace,
)

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# LFM2_PLAN_OVERRIDE (JSON) for the multichip plan, e.g. '{"partial_fp32": ["mixer"]}'
PLAN = replace(
    DEFAULT_MESH_PLAN,
    **{
        k: tuple(v) if isinstance(v, list) else v
        for k, v in json.loads(os.environ.get("LFM2_PLAN_OVERRIDE", "{}")).items()
    },
)
cfg = hf_config()
args = Lfm2MoeArgs.from_hf_config(cfg)
H = args.hidden_size
sd = layer_state_dict(layer, True)
embed = _real_embed()
users = [User(slot=s, prompt_len=n, decode_steps=12, seed=900 + s) for s, n in [(1, 33), (5, 63), (9, 300), (20, 777)]]
users.append(User(slot=12, prompt_len=33, decode_steps=2, seed=533))  # the increasing-lengths slot-9 user
for u in users:
    u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
hf, rope_hf = build_hf_layer(cfg, layer, sd)
ff = hf.feed_forward
hf_scores = {}
for u in users:
    cap = {}
    h = ff.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach().reshape(-1, a[0].shape[-1])))
    with torch.no_grad():
        hf_layer_forward(hf, rope_hf, u.x.unsqueeze(0))
    h.remove()
    hf_scores[u.slot] = torch.sigmoid(cap["h"] @ ff.gate.weight.T)  # [S, E] fp32
bias = ff.expert_bias.float()

captured = []
orig = OptimizedMoE._router_scores


def spy(self, x):
    s = orig(self, x)
    if x.shape[-2] == TILE:  # decode
        captured.append(ttnn.to_torch(ttnn.get_device_tensors(s)[0]).float().reshape(TILE, -1)[:, : self.E])
    return s


OptimizedMoE._router_scores = spy
p = mesh_device_params()
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
md = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=p["trace_region_size"], l1_small_size=p["l1_small_size"]
)
max_seq = max(u.prompt_len + u.decode_steps for u in users)
sc_policy = replace(DEFAULT_POLICY, dispatch_ffn="unified", dispatch_fidelity="hifi2")
res = {}
try:
    for name, make in [
        ("single", lambda: make_optimized(sd, cfg, args, layer, md, max_seq, policy=sc_policy)),
        ("multi", lambda: make_multichip(sd, cfg, args, layer, md, max_seq, policy=MC_POLICY, plan=PLAN)),
    ]:
        dec, rope = make()
        layout = PagedLayout(users, max_seq)
        if dec.is_attention:
            dec.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
        for u in users:
            run_prefill(dec, rope, md, layout, u, 2048)
        inp = DecodeInputs(md, H, layout.max_blocks_per_seq, slot_mask_width=getattr(dec, "slot_mask_width", None))
        table = layout.decode_table(users)
        captured.clear()
        rows = []
        for t in range(max(u.decode_steps for u in users)):
            x = torch.zeros(1, 1, TILE, H)
            cur = torch.full((TILE,), -1, dtype=torch.int32)
            for u in users:
                if t < u.decode_steps:
                    x[0, 0, u.slot] = u.x[u.prompt_len + t]
                    cur[u.slot] = u.prompt_len + t
            inp.update(x, cur, table)
            ttnn.deallocate(decode_step(dec, rope, inp))
            for u in users:
                if t < u.decode_steps:
                    rows.append((u.slot, u.prompt_len + t, captured[-1][u.slot]))
        res[name] = rows
        del dec, rope, inp
finally:
    ttnn.close_mesh_device(md)

for name, rows in res.items():
    err = torch.stack([(s - hf_scores[slot][pos]).abs().max() for slot, pos, s in rows])
    flips, near = 0, []
    for slot, pos, s in rows:
        ref = hf_scores[slot][pos] + bias
        sel_tt = set(torch.topk(s + bias, 4).indices.tolist())
        sel_hf = set(torch.topk(ref, 4).indices.tolist())
        top5 = torch.topk(ref, 5).values
        if sel_tt != sel_hf:
            flips += 1
            near.append(f"slot {slot} pos {pos} HF gap {top5[3] - top5[4]:.2e}")
    print(
        f"NOISE layer {layer} {name:6s}: {len(rows)} decode rows, |score - HF| max {err.max():.2e} "
        f"p99 {err.quantile(0.99):.2e} median {err.median():.2e}; top-4 set != HF: {flips} {near}",
        flush=True,
    )
