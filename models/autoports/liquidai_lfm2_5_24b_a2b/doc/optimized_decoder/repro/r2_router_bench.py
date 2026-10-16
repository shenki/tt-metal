# Decode router variants (T=32): exactness vs the current route + traced device time.
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_inputs, layer_state_dict
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.test_functional_decoder import _real_embed
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import DEFAULT_FUSIONS
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import Lfm2MoeArgs
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    DramShardedLinear,
    OptimizedMoE,
    ShardedNorm,
    _ckc,
)

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
cfg = hf_config()
args = Lfm2MoeArgs.from_hf_config(cfg)
sd = layer_state_dict(3, True)
moe = OptimizedMoE(md, args, sd, DEFAULT_POLICY, DEFAULT_FUSIONS)
H, E, K = moe.H, moe.E, moe.K
x = layer_inputs(32, H, 5, _real_embed()).reshape(1, 1, 32, H)
x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=L1)
norm = ShardedNorm(md, H, 8, 1e-5, _ckc(md, "hifi4"))
xs = ttnn.to_memory_config(xt, norm.mem)  # the ffn_norm output layout (8-core width shard)
ckc = moe.ckc_router
wr = sd["feed_forward.gate.weight"].T.float()  # [H, E]
ds = DramShardedLinear(md, torch.nn.functional.pad(wr, (0, 256 - E)), ttnn.bfloat16, (8, 8, 1))
assert ds.in_mem == norm.mem


def tile_mask(scores, mem):
    biased = ttnn.add(scores, moe.bias, memory_config=mem)
    vals, idx = ttnn.topk(biased, k=K, dim=-1, memory_config=mem)
    ttnn.deallocate(idx)
    kth = ttnn.slice(vals, (0, 0, 0, K - 1), (1, 1, vals.shape[-2], K), memory_config=mem)  # topk is sorted
    ttnn.deallocate(vals)
    sel = ttnn.multiply(scores, ttnn.ge(biased, kth, memory_config=mem), memory_config=mem)
    ttnn.deallocate(biased)
    ttnn.deallocate(kth)
    ttnn.deallocate(scores)
    denom = ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True, memory_config=mem), 1e-6, memory_config=mem)
    w = ttnn.div(sel, denom, memory_config=mem)
    ttnn.deallocate(sel)
    return ttnn.typecast(w, ttnn.bfloat16, memory_config=mem)


def scores_ksplit_l1(xi):
    T, ks = 32, moe.policy.router_ksplit
    xr = ttnn.permute(ttnn.reshape(xi, (T, ks, H // ks)), (1, 0, 2), memory_config=L1)
    part = ttnn.matmul(
        ttnn.reshape(xr, (1, ks, T, H // ks)),
        moe.router_split,
        compute_kernel_config=ckc,
        dtype=ttnn.float32,
        memory_config=L1,
    )
    ttnn.deallocate(xr)
    logits = ttnn.sum(part, dim=1, keepdim=True, memory_config=L1)
    ttnn.deallocate(part)
    return ttnn.sigmoid(ttnn.reshape(logits, (1, 1, T, E)), memory_config=L1)


def scores_plain(xi):
    return ttnn.linear(
        xi, moe.router, compute_kernel_config=ckc, dtype=ttnn.float32, activation="sigmoid", memory_config=L1
    )


def scores_ds(xsh):
    o = ttnn.linear(
        xsh,
        ds.weight,
        program_config=ds.decode_cfg,
        compute_kernel_config=ckc,
        memory_config=ds.out_mem,
        dtype=ttnn.float32,
    )
    oi = ttnn.to_memory_config(o, L1)
    ttnn.deallocate(o)
    lg = ttnn.slice(oi, (0, 0, 0, 0), (1, 1, 32, E), memory_config=L1)
    ttnn.deallocate(oi)
    s = ttnn.sigmoid(lg, memory_config=L1)
    ttnn.deallocate(lg)
    return s


V = {
    "current": lambda: moe.route(xt),
    "tile_mask_l1(kth slice) + ksplit_l1": lambda: tile_mask(scores_ksplit_l1(xt), L1),
    "tile_mask_l1 + plain bf16 linear": lambda: tile_mask(scores_plain(xt), L1),
    "tile_mask_l1 + DS router (sharded in)": lambda: tile_mask(scores_ds(xs), L1),
}
ref = ttnn.to_torch(V["current"]()).float()


def timed(f):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        o = f()
        ttnn.deallocate(o)
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(100):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    return (time.perf_counter() - t0) / 100 * 1e6


for name, f in V.items():
    try:
        out = ttnn.to_torch(f()).float()
        same_sel = torch.equal(out > 0, ref > 0)
        print(
            f"ROUTER {name:40s}: {timed(f):7.1f} us  same selection {same_sel}  max|dw| {(out - ref).abs().max().item():.2e}",
            flush=True,
        )
    except Exception as e:
        print(f"ROUTER {name:40s}: ERR {str(e).splitlines()[0][:150]}", flush=True)
ttnn.close_mesh_device(md)
