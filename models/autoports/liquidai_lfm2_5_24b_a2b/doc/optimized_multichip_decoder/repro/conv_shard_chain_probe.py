# Stage 5 review P1: ShortConv decode chain on the 8-core shard grid (per chip: 512 channels).
#   current: DS in_proj [2048,1536] -> S2I -> 3 slices (L1 interleaved) -> bx = b*x, conv = s0*t0 + s1*t1 + bx*t2 (mul, 2 mac),
#            state update 2 x where(slot_mask), y = c*conv (110-core eltwise on 16 tiles) -> I2S into out_proj in_mem
#   sharded: DS in_proj -> 3 slices straight into the out_proj input layout (8 cores x [32, 64]) -> the same chain on 8-core
#            shards (state, taps expanded to [32, 512], slot mask: all on the same shards) -> y already in out_proj in_mem
# Single chip (every chip runs the same local chain).  Eager check vs torch, then traced timing (wall / REPS calls).
#   TT_VISIBLE_DEVICES=0 python conv_shard_chain_probe.py
import math
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, _ckc, width_shard_layout

T, H, C = 32, 2048, 512
REPS = 16
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ckc = _ckc(md, "hifi2")
g = torch.Generator().manual_seed(0)
w_in = torch.randn(H, 3 * C, generator=g) / math.sqrt(H)
taps_t = torch.randn(3, C, generator=g) * 0.3
x_t = torch.randn(1, 1, T, H, generator=g)
s0_t, s1_t = torch.randn(1, 1, T, C, generator=g), torch.randn(1, 1, T, C, generator=g)
act = torch.zeros(1, 1, T, 1)
act[0, 0, 5] = 1.0
act[0, 0, 9] = 1.0
mask_t = act.expand(1, 1, T, C).contiguous()
lin = DramShardedLinear(md, w_in, ttnn.bfloat8_b, (8, 8, 2), True)
ds_out_in = width_shard_layout(md, 8, C // 8)  # conv out_proj (8, 2, 2) input layout: [32, 64] per core
xs = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=lin.in_mem)
# reference (bf16-ish via torch fp32)
bcx = x_t[0, 0] @ w_in
b_, c_, xx_ = bcx[:, :C], bcx[:, C : 2 * C], bcx[:, 2 * C :]
bx_ = b_ * xx_
conv_ = s0_t[0, 0] * taps_t[0] + s1_t[0, 0] * taps_t[1] + bx_ * taps_t[2]
y_ref = c_ * conv_
s0_ref = torch.where(mask_t[0, 0] > 0, s1_t[0, 0], s0_t[0, 0])
s1_ref = torch.where(mask_t[0, 0] > 0, bx_, s1_t[0, 0])


def dev(t, mem, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=md, memory_config=mem)


def make(variant):
    if variant == "current":
        st = [dev(s0_t, DRAM), dev(s1_t, DRAM)]
        taps = [dev(taps_t[k].reshape(1, 1, 1, C), DRAM) for k in range(3)]
        mask = dev(mask_t, DRAM)
    else:
        # persistent state / taps placement: "sharded" = all L1 shards; "sharded_dram_taps" = taps (expanded) in DRAM
        # interleaved; "sharded_dram_state_taps" = state and taps in DRAM interleaved (chain intermediates still sharded)
        st_mem = DRAM if variant == "sharded_dram_state_taps" else ds_out_in
        tap_mem = DRAM if variant in ("sharded_dram_taps", "sharded_dram_state_taps") else ds_out_in
        st = [dev(s0_t, st_mem), dev(s1_t, st_mem)]
        taps = [dev(taps_t[k].reshape(1, 1, 1, C).expand(1, 1, T, C).contiguous(), tap_mem) for k in range(3)]
        mask = dev(mask_t, ds_out_in)
    return st, taps, mask


def chain(variant, st, taps, mask):
    mem = L1 if variant == "current" else ds_out_in
    if variant == "current":
        bcx_t = lin.decode(xs, ckc, L1)  # DS + S2I
    else:
        bcx_t = lin.decode_sharded(xs, ckc)  # 8 cores x [32, 192]
    b = ttnn.slice(bcx_t, (0, 0, 0, 0), (1, 1, T, C), memory_config=mem)
    c = ttnn.slice(bcx_t, (0, 0, 0, C), (1, 1, T, 2 * C), memory_config=mem)
    xx = ttnn.slice(bcx_t, (0, 0, 0, 2 * C), (1, 1, T, 3 * C), memory_config=mem)
    ttnn.deallocate(bcx_t)
    bx = ttnn.multiply(b, xx, memory_config=mem)
    ttnn.deallocate(b)
    ttnn.deallocate(xx)
    conv = ttnn.multiply(st[0], taps[0], memory_config=mem)
    conv = ttnn.mac(st[1], taps[1], conv)
    conv = ttnn.mac(bx, taps[2], conv)
    ttnn.where(mask, st[1], st[0], output_tensor=st[0])
    ttnn.where(mask, bx, st[1], output_tensor=st[1])
    ttnn.deallocate(bx)
    y = ttnn.multiply(c, conv, memory_config=mem)
    ttnn.deallocate(c)
    ttnn.deallocate(conv)
    if variant == "current":
        ys = ttnn.to_memory_config(y, ds_out_in)  # I2S into the out_proj input layout
        ttnn.deallocate(y)
        y = ys
    return y


for variant in ("current", "sharded", "sharded_dram_taps", "sharded_dram_state_taps"):
    try:
        st, taps, mask = make(variant)
        y = chain(variant, st, taps, mask)
        p = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        ok = (
            p(ttnn.to_torch(y).float()[0, 0], y_ref),
            p(ttnn.to_torch(st[0]).float()[0, 0], s0_ref),
            p(ttnn.to_torch(st[1]).float()[0, 0], s1_ref),
            y.memory_config() == ds_out_in,
        )
        ttnn.deallocate(y)
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        try:
            for _ in range(REPS):
                ttnn.deallocate(chain(variant, st, taps, mask))
        finally:
            ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        us = (time.perf_counter() - t0) / (5 * REPS) * 1e6
        ttnn.release_trace(md, tid)
        print(
            f"CONVCHAIN {variant:8s}: {us:6.2f} us per chain (DS in_proj included)  pcc(y, s0, s1)/on_out_proj_layout {ok}",
            flush=True,
        )
    except Exception as e:
        print(f"CONVCHAIN {variant:8s}: ERR {str(e).splitlines()[0][:220]}", flush=True)
        import traceback

        traceback.print_exc()
ttnn.close_mesh_device(md)
