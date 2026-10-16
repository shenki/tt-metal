import sys
import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
E, H, I, M = 64, 2048, 1536, 32
g = torch.Generator().manual_seed(0)
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def ckc(fid):
    return ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity={"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[fid],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
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


role = sys.argv[1]  # gate_up | down
n_active = int(sys.argv[2])
K, N = (H, 2 * I) if role == "gate_up" else (I, H)
w_t = torch.randn(1, E, K, N, generator=g) * 0.02
sp_t = torch.zeros(1, 1, 1, E)
sp_t[..., torch.randperm(E, generator=g)[:n_active]] = 1.0
sp = ttnn.from_torch(sp_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
if role == "gate_up":
    a = ttnn.from_torch(torch.randn(1, 1, M, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
else:
    a = ttnn.from_torch(torch.randn(1, E, M, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
nt = N // 32
kt = K // 32
grids = [(8, 6), (8, 8), (11, 8), (8, 4), (12, 8)]
for dname in sys.argv[3].split(","):
    w = ttnn.from_torch(w_t, dtype=DT[dname], layout=ttnn.TILE_LAYOUT, device=md)
    for fid in sys.argv[4].split(","):
        best = None
        for grid in grids:
            cores = grid[0] * grid[1]
            if nt % cores and nt > cores:
                pass
            per_core_n = -(-nt // cores)
            if per_core_n * cores < nt:
                continue
            for in0_bw in [b for b in (1, 2, 4, 8, 16, 32, 64) if kt % b == 0]:
                for sub_w in [s for s in (1, 2, 4, 8) if per_core_n % s == 0 and s <= 8]:
                    try:
                        kw = dict(
                            sparsity=sp,
                            nnz=None,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            output_tile=ttnn.Tile([32, 32]),
                            program_config=cfg(grid, in0_bw, sub_w, per_core_n),
                            compute_kernel_config=ckc(fid),
                            dtype=ttnn.bfloat16,
                        )
                        if role == "down":
                            kw["is_input_a_sparse"] = True
                        f = lambda: ttnn.sparse_matmul(a, w, **kw)
                        f()
                        ttnn.synchronize_device(md)
                        t0 = time.perf_counter()
                        for _ in range(10):
                            o = f()
                        ttnn.synchronize_device(md)
                        us = (time.perf_counter() - t0) / 10 * 1e6
                        if best is None or us < best[0]:
                            best = (us, grid, in0_bw, sub_w, per_core_n)
                        if in0_bw == 1 and sub_w == 1 and grid == (8, 6):
                            print(
                                f"REF {role} active={n_active} {dname} {fid} grid={grid} in0_bw=1 sub=1 pcn={per_core_n}: {us:8.1f} us",
                                flush=True,
                            )
                    except Exception as e:
                        pass
        print(
            f"BEST {role} active={n_active} {dname} {fid}: {best[0]:8.1f} us grid={best[1]} in0_bw={best[2]} sub_w={best[3]} per_core_N={best[4]}",
            flush=True,
        )
    ttnn.deallocate(w)
ttnn.close_mesh_device(md)
