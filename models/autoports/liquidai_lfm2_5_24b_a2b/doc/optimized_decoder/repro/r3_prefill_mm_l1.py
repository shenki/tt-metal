# Review round 3 P2-2(a): tt-perf-report SLOW prefill matmul rows (2048-row chunk).
#  1. "place input 0 in L1": live DramShardedLinear.prefill (qkv / wo / conv_in / conv_out) and the dense w1
#     with the activation in DRAM (live) vs L1 interleaved.
#  2. dense w1/w3 (2048 x 2048 x 11776, interleaved BFP8) 1x1 output subblock: alternative 2D grids and the
#     N-padded 12288 geometry (= the decode-padded DS weight), all with >= 1x2 subblocks.
# Timing = traced replay wall / N (device time + small launch), PCC vs torch for every variant.
#   TT_VISIBLE_DEVICES=0 python models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/repro/r3_prefill_mm_l1.py
import math
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DEFAULT_POLICY as P
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, _ckc, prefill_2d_config

TILE, T, H = 32, 2048, 2048
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
grid = md.compute_with_storage_grid_size()
print(f"R3MM grid {grid.x}x{grid.y}", flush=True)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
ckc = _ckc(md, "hifi2")


def timed(f, n=20):
    out = f()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    o2 = f()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        us = (time.perf_counter() - t0) / n * 1e6
    finally:
        ttnn.release_trace(md, tid)
    ttnn.deallocate(o2)
    return us, out


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def show(name, us, p, cfg=""):
    print(f"R3MM {name:58s} {us:8.1f} us  pcc {p:.6f}  {cfg}", flush=True)


def cfg_str(c):
    return (
        f"grid {c.compute_with_storage_grid_size.x}x{c.compute_with_storage_grid_size.y} bw {c.in0_block_w} "
        f"blk {c.out_block_h}x{c.out_block_w} sub {c.out_subblock_h}x{c.out_subblock_w} pc {c.per_core_M}x{c.per_core_N}"
    )


g = torch.Generator().manual_seed(0)
x_t = torch.randn(1, 1, T, H, generator=g)
x_dram = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=DRAM)
x_l1 = ttnn.to_memory_config(x_dram, L1)

# ---- 1. input 0 DRAM vs L1 on the live DS prefill matmuls ----
for name, n, cfg in [
    ("qkv", 3072, P.ds_wqkv),
    ("wo", H, P.ds_wo),
    ("conv_in", 6144, P.ds_conv_in),
    ("conv_out", H, P.ds_conv_out),
]:
    w = torch.randn(H, n, generator=g) / math.sqrt(H)
    lin = DramShardedLinear(md, w, ttnn.bfloat8_b, cfg, P.prefill_mm_v2)
    ref = x_t[0, 0] @ w
    for where, x in (("DRAM", x_dram), ("L1", x_l1)):
        us, o = timed(lambda: lin.prefill(x, ckc, DRAM))
        show(
            f"{name} 2048x{H}x{n} in0 {where}",
            us,
            pcc(ttnn.to_torch(o)[0, 0].float(), ref),
            cfg_str(lin._prefill_cfgs[T]),
        )
        ttnn.deallocate(o)
    ttnn.deallocate(lin.weight)


# ---- 2. dense w1 geometry ----
def cfg2d(gx, gy, n, bw, ob_h, ob_w, sh, sw):
    mt, nt = T // TILE, n // TILE
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=ob_h,
        out_block_w=ob_w,
        per_core_M=math.ceil(mt / gy),
        per_core_N=math.ceil(nt / gx),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


I0 = 11776
w1_t = torch.randn(H, I0, generator=g) / math.sqrt(H)
ref = x_t[0, 0] @ w1_t
for n in (I0, 12288):
    wp = torch.zeros(H, n)
    wp[:, :I0] = w1_t
    w1 = ttnn.from_torch(wp, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=DRAM)
    live = prefill_2d_config(md, T, H, n, P.prefill_mm_v2, 1088)
    variants = [("live", live)]
    nt = n // TILE
    # per_core_M 8 (gy 8) makes 2x subblock heights legal; per-core N chosen to have a factor 2/3/4
    for gx, gy in [(grid.x, 8), (8, 8), (grid.x, grid.y), (8, grid.y)]:
        if gx > grid.x or gy > grid.y:
            continue
        pm, pn = math.ceil(T // TILE / gy), math.ceil(nt / gx)
        for ob_h, ob_w in [(pm, pn), (pm // 2, pn), (pm, pn // 2), (pm // 2, pn // 2)]:
            if ob_h < 1 or ob_w < 1 or pm % ob_h or pn % ob_w:
                continue
            for sh, sw in [(2, 2), (1, 4), (1, 3), (2, 1), (1, 2)]:
                if ob_h % sh or ob_w % sw:
                    continue
                for bw in (8, 4, 2):
                    used = 2 * pm * bw * 2048 + 2 * bw * pn * 1088 + ob_h * ob_w * (2048 + 4096)
                    if used <= 1_300_000:
                        variants.append((f"g{gx}x{gy}", cfg2d(gx, gy, n, bw, ob_h, ob_w, sh, sw)))
                        break
                break  # largest legal subblock only
    seen = set()
    for vname, c in variants:
        key = cfg_str(c)
        if key in seen:
            continue
        seen.add(key)
        for where, x in (("DRAM", x_dram),) + ((("L1", x_l1),) if vname == "live" else ()):
            try:
                us, o = timed(
                    lambda: ttnn.linear(x, w1, program_config=c, compute_kernel_config=ckc, memory_config=DRAM)
                )
            except Exception as e:
                print(f"R3MM w1 N{n} {vname} ERR {str(e).splitlines()[0][:150]} {key}", flush=True)
                continue
            got = ttnn.to_torch(o)[0, 0].float()[:, :I0]
            show(f"w1 2048x{H}x{n} {vname} in0 {where}", us, pcc(got, ref), key)
            ttnn.deallocate(o)
    ttnn.deallocate(w1)

ttnn.close_mesh_device(md)
