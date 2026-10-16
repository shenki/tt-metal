# Exactness of Policy.moe_route_v3 vs the current decode routing on REAL router inputs
# (generated/r3_route_hn.pt: HF ffn_norm outputs, 1600 book-text tokens per layer; repro/r3_route_capture_hn.py).
# 64-wide path: routing bf16 [32, E] bitwise; compact buckets 1/2/4: per active row the induced dense
# weights D[r, e] = sum_s wc[r, s] [ids[s] == e] bitwise (same expert set + same bf16 weights).
import json
import sys

import torch

import ttnn
from models.autoports.lfm2_r3_route.tests.harness import hf_config, layer_state_dict
from models.autoports.lfm2_r3_route.tt.fused_decoder import DEFAULT_FUSIONS
from models.autoports.lfm2_r3_route.tt.model_config import Lfm2MoeArgs
from models.autoports.lfm2_r3_route.tt.optimized_decoder import DEFAULT_POLICY, OptimizedMoE, ShardedNorm, _ckc, replace

L1 = ttnn.L1_MEMORY_CONFIG
layers = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 3, 7, 15, 23, 31, 39]
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
hn_all = torch.load("generated/r3_route_hn.pt")
cfg = hf_config()
args = Lfm2MoeArgs.from_hf_config(cfg)
pol = replace(DEFAULT_POLICY, moe_route_v3=True, **extra)
pol_cur = replace(pol, router_ds_fused=False)  # "current" = DEFAULT router + current routing glue
T = 32
dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=md, memory_config=L1)
st = lambda t: ttnn.to_torch(t).float()


def induced(ids, wc, E):
    ids = ttnn.to_torch(ids).long().reshape(-1)
    wc = st(wc)
    wc = wc[0, :, :, 0].T if wc.shape[-1] == 1 else wc.reshape(T, -1)[:, : ids.numel()]  # [1,n,T,1] -> [T,n]
    D = torch.zeros(T, E)
    for s in range(ids.numel()):
        D[:, ids[s]] += wc[:, s]
    return D


tot = {}
for li in layers:
    moe = OptimizedMoE(md, args, layer_state_dict(li, True), pol, DEFAULT_FUSIONS)
    norm = ShardedNorm(md, moe.H, 8, 1e-5, _ckc(md, "hifi4"))
    hn = hn_all[li]
    stats = {
        "scores": [0, 0],
        "wide": [0, 0],
        "wide_idle": [0, 0],
        "c1": [0, 0],
        "c2": [0, 0],
        "c4": [0, 0],
        "c4_gap": [0, 0],
    }
    for bi in range(hn.shape[0] // T):
        x = dev(hn[bi * T : (bi + 1) * T].reshape(1, 1, T, -1))
        moe._x_sharded = ttnn.to_memory_config(x, norm.mem)
        moe.policy = pol_cur
        s0 = st(moe._router_scores(x))
        moe.policy = pol
        s1 = st(moe._router_scores(x))
        stats["scores"][0] += 1
        stats["scores"][1] += int(torch.equal(s0, s1))
        if not torch.equal(s0, s1) and bi < 3:
            print(
                f"L{li} batch {bi} scores differ: max {(s0-s1).abs().max().item():.3g} frac {(s0 != s1).float().mean().item():.3f}",
                flush=True,
            )

        def act_t(rows):
            a = torch.zeros(1, 1, T, 1)
            a[0, 0, rows] = 1
            return dev(a), rows

        for name, rows in [("wide", list(range(T))), ("wide_idle", [r for r in range(T) if r % 3 != 1])]:
            sa, rows = act_t(rows)
            moe.policy = pol_cur
            cur = st(ttnn.where(sa, moe.route(x), 0.0))
            moe.policy = pol
            new = st(moe._route_v3_wide(x, sa))
            stats[name][0] += 1
            stats[name][1] += int(torch.equal(cur, new))
            if not torch.equal(cur, new):
                bad = (cur != new).any(-1).nonzero().flatten().tolist()
                print(
                    f"L{li} batch {bi} {name}: MISMATCH rows {bad[:8]} sel-equal {torch.equal(cur > 0, new > 0)} max|dw| {(cur-new).abs().max().item():.3g}",
                    flush=True,
                )
        for name, b, rows in [("c1", 1, [0]), ("c2", 2, [0, 1]), ("c4", 4, [0, 1, 2, 3]), ("c4_gap", 4, [0, 1, 3])]:
            sa, rows = act_t(rows)
            moe.policy = pol_cur
            ids0, wc0 = moe._compact_route(x, sa, b)
            moe.policy = pol
            ids1, wc1 = moe._compact_route_v3(x, b)
            D0, D1 = induced(ids0, wc0, moe.E)[rows], induced(ids1, wc1, moe.E)[rows]
            ok = torch.equal(D0, D1)
            stats[name][0] += 1
            stats[name][1] += int(ok)
            if not ok:
                print(
                    f"L{li} batch {bi} {name}: MISMATCH sel-equal {torch.equal(D0 > 0, D1 > 0)} max|dw| {(D0-D1).abs().max().item():.3g}",
                    flush=True,
                )
        ttnn.deallocate(moe._x_sharded)
        ttnn.deallocate(x)
    print(f"LAYER {li}: " + "  ".join(f"{k} {v[1]}/{v[0]} bitwise" for k, v in stats.items()), flush=True)
    for k, v in stats.items():
        tot.setdefault(k, [0, 0])
        tot[k][0] += v[0]
        tot[k][1] += v[1]
    del moe
print("TOTAL: " + "  ".join(f"{k} {v[1]}/{v[0]} batches of 32 tokens bitwise" for k, v in tot.items()))
ttnn.close_mesh_device(md)
