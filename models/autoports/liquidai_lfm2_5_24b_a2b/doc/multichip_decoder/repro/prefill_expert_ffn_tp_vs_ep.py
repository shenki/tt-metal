# Stage 4: prefill expert FFN (deepseek_prefill.unified_routed_expert_moe) per chip, 2048-token chunk, top-4:
#   single-chip baseline  E=64 experts, I=1536, 8192 pairs (128 per expert)
#   TP=4 (current MC)     E=64 experts, I=384,  8192 pairs (every chip runs every pair on a quarter of I)
#   EP=4 (candidate)      E=16 experts, I=1536, 2048 pairs (each chip runs its 16 experts' pairs)
# Traced replay timing (device work + launch), BFP8 weights, HiFi2 (MC_POLICY dispatch_fidelity).
#   TT_VISIBLE_DEVICES=0 python prefill_expert_ffn_tp_vs_ep.py
import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
H, TILE, T, K = 2048, 32, 2048, 4
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


def as_tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=md)


def timeit(f, n=10):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(n):
            ttnn.deallocate(f())
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / (3 * n) * 1e6
    finally:
        ttnn.release_trace(md, tid)


g = torch.Generator().manual_seed(0)
for name, E, I, per_expert in [
    ("1chip E64 I1536", 64, 1536, 128),
    ("TP4 E64 I384", 64, 384, 128),
    ("EP4 E16 I1536", 16, 1536, 128),
]:
    counts = torch.full((E,), per_expert, dtype=torch.int32)
    tiles = (counts + TILE - 1) // TILE
    region = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(tiles, 0)[:-1].to(torch.int32)]) * TILE
    maxbuf = ((T * K + 64 * 31) + TILE - 1) // TILE * TILE  # MC _max_buf (64-expert worst case padding)
    buf = as_tt(
        torch.randn(maxbuf, H, generator=g) * 0.5, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # as the decoder (embedding output)
    gates = [as_tt(torch.randn(H, I, generator=g) * 0.02, ttnn.bfloat8_b) for _ in range(E)]
    ups = [as_tt(torch.randn(H, I, generator=g) * 0.02, ttnn.bfloat8_b) for _ in range(E)]
    downs = [as_tt(torch.randn(I, H, generator=g) * 0.02, ttnn.bfloat8_b) for _ in range(E)]
    rm = ttnn.ROW_MAJOR_LAYOUT
    reg_t, cnt_t = as_tt(region, ttnn.uint32, rm), as_tt(counts, ttnn.uint32, rm)
    gidx = as_tt(torch.arange(E, dtype=torch.int32), ttnn.uint32, rm)
    f = lambda: ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        buf, reg_t, cnt_t, gidx, gates, ups, downs, T, compute_kernel_config=ckc
    )
    us = timeit(f)
    flops = 2 * 3 * E * per_expert * H * I
    print(f"FFN {name:18s} pairs {E * per_expert:5d}: {us:8.1f} us  {flops / us / 1e6:6.1f} TFLOP/s", flush=True)
    for t in gates + ups + downs + [buf]:
        ttnn.deallocate(t)
ttnn.close_mesh_device(md)
