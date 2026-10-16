# Dense MLP w1/w3 decode (M=32, K=2048, N=11776): TTNN default vs explicit 1D in0-mcast program
# configs (per_core_N -> cores, in0_block_w over divisors of K-tiles 64, out_subblock_w) for one
# weight dtype / math fidelity.  Interleaved DRAM weight, x in L1 or DRAM, output DRAM (as tt/ code).
#   python r2_bfp4_w13_1d_sweep.py <bfp8|bfp4> <hifi2|lofi> [x_mem=dram|l1]
# Timing: device-side traced replay (trace of 20 calls, replayed 5x, synchronize).
import math
import sys
import time

import torch

import ttnn

DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FI = {"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}
dt_name, fi_name = sys.argv[1], sys.argv[2]
xmem = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[sys.argv[3] if len(sys.argv) > 3 else "dram"]
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
grid = md.compute_with_storage_grid_size()
ncores = grid.x * grid.y
print("grid", grid.x, grid.y, dt_name, fi_name, sys.argv[3:] or "dram")
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=FI[fi_name], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def timeit(f, n=20, reps=5):
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
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / (n * reps) * 1e6
    finally:
        ttnn.release_trace(md, tid)


K, N, T = 2048, 11776, 32
g = torch.Generator().manual_seed(0)
w_t = torch.randn(K, N, generator=g) * 0.02
x_t = torch.randn(1, 1, T, K, generator=g)
ref = x_t[0, 0] @ w_t
x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=xmem)
w = ttnn.from_torch(w_t, dtype=DT[dt_name], layout=ttnn.TILE_LAYOUT, device=md)


def pcc(o):
    return torch.corrcoef(torch.stack([ttnn.to_torch(o).float()[0, 0].flatten(), ref.flatten()]))[0, 1].item()


f0 = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
o = f0()
p0 = pcc(o)
ttnn.deallocate(o)
base = timeit(f0)
print(f"DEFAULT w13 {dt_name}/{fi_name}: {base:.1f} us pcc={p0:.5f}", flush=True)
nt, kt = N // 32, K // 32
res = []
for pcn in (2, 3, 4, 5, 6, 8, 12, 16, 23):
    cores = math.ceil(nt / pcn)
    if cores > ncores:
        continue
    for bw in (1, 2, 4, 8, 16, 32):
        for sw in [s for s in (4, 2, 1) if pcn % s == 0][:2]:
            try:
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    in0_block_w=bw,
                    out_subblock_h=1,
                    out_subblock_w=sw,
                    per_core_M=1,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                f = lambda: ttnn.linear(
                    x, w, program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                o = f()
                p = pcc(o)
                ttnn.deallocate(o)
                us = timeit(f)
                res.append((us, pcn, cores, bw, sw, p))
                print(f"    try per_core_N={pcn} cores={cores} bw={bw} sw={sw}: {us:.1f} us pcc={p:.5f}", flush=True)
            except Exception as e:
                print(f"    FAIL per_core_N={pcn} bw={bw} sw={sw}: {str(e).splitlines()[0][:150]}", flush=True)
res.sort()
for r in res[:8]:
    print(
        f"  1D w13 {dt_name}/{fi_name}: {r[0]:7.1f} us per_core_N={r[1]} cores={r[2]} in0_bw={r[3]} sub_w={r[4]} pcc={r[5]:.5f}"
    )
ttnn.close_mesh_device(md)
