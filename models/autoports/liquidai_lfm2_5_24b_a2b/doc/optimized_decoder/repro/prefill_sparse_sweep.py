import sys
import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
E, H, I, R, G = 64, 2048, 1536, 32, 16
g = torch.Generator().manual_seed(0)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def cfg(grid, in0_bw, sub_w, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def timeit(f, n=3, reps=2):
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


# per-tile masks: ~55 of 64 experts active per 32-token tile (as with real routing, 32 tokens x top-4)
mask = torch.zeros(1, 1, G, E)
for t in range(G):
    sel = torch.stack([torch.randperm(E, generator=g)[:4] for _ in range(R)]).flatten()
    mask[0, 0, t, sel.unique()] = 1.0
print("avg active experts per tile", mask.sum(-1).mean().item(), flush=True)
sp = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
role = sys.argv[1]
if role == "gate_up":
    K, N = H, 2 * I
    a = ttnn.from_torch(torch.randn(1, G, R, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    extra = {}
else:
    K, N = I, H
    a = ttnn.from_torch(torch.randn(G, E, R, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    extra = dict(is_input_a_sparse=True, is_input_b_sparse=False)
w = ttnn.from_torch(
    torch.randn(1, E, K, N, generator=g) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md
)
nt, kt = N // 32, K // 32
res = []
for grid in [(8, 6), (8, 8), (8, 4), (12, 8), (6, 8)]:
    per_core_n = -(-nt // (grid[0] * grid[1]))
    for bw in [b for b in (1, 2, 4, 8, 16) if kt % b == 0]:
        for sub_w in [s for s in (1, 2, 4) if per_core_n % s == 0]:
            try:
                kw = dict(
                    sparsity=sp,
                    nnz=None,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=ttnn.Tile([32, 32]),
                    program_config=cfg(grid, bw, sub_w, per_core_n),
                    compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16,
                    **extra,
                )
                us = timeit(lambda: ttnn.sparse_matmul(a, w, **kw))
                res.append((us, grid, bw, sub_w, per_core_n))
                print(f"  try {role} grid={grid} bw={bw} sub={sub_w}: {us:9.1f} us", flush=True)
            except Exception as e:
                print(f"  FAIL {role} grid={grid} bw={bw} sub={sub_w}: {str(e).splitlines()[0][:150]}", flush=True)
res.sort()
for r in res[:4]:
    print(
        f"BEST {role}: {r[0]:9.1f} us (per 512 tokens) grid={r[1]} in0_bw={r[2]} sub_w={r[3]} per_core_N={r[4]}",
        flush=True,
    )
ttnn.close_mesh_device(md)
