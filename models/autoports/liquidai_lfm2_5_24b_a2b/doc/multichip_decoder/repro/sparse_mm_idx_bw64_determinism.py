# Op issue: sparse_matmul indexed (gather) mode, x [1,1,32,2048] @ w [1,64,2048,768] BFP8, 1D mcast config
# (8,3,in0_block_w,1) per_core_N 1, eager, nothing else in L1: bitwise stable for in0_block_w 8 / 32 / 64 and
# 4 / 16 slots (sweeps/sparse_mm_idx_bw64_determinism.log).  Inside the traced multichip conv_moe bucket-4 decode
# in0_block_w 64 was nondeterministic (logs/det_bisect.log) -> not an isolated-op bug; 32 adopted.
#   TT_VISIBLE_DEVICES=0 python sparse_mm_idx_bw64_determinism.py
import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
E, H, N = 64, 2048, 768
g = torch.Generator().manual_seed(0)
w = ttnn.from_torch(
    torch.randn(1, E, H, N, generator=g) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md
)
x = ttnn.from_torch(torch.randn(1, 1, 32, H, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
sp = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
for slots in (4, 16):
    ids = torch.randperm(E, generator=g)[:slots].to(torch.int32).reshape(1, 1, 1, -1)
    idx = ttnn.from_torch(ids, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
    for bw in (8, 32, 64):
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        outs = []
        for _ in range(10):
            o = ttnn.sparse_matmul(
                x,
                w,
                sparsity=sp,
                indices=idx,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=cfg,
                compute_kernel_config=ckc,
                dtype=ttnn.bfloat16,
            )
            outs.append(ttnn.to_torch(o).float())
            ttnn.deallocate(o)
        d = max((o - outs[0]).abs().max().item() for o in outs[1:])
        print(f"SPMM_DET slots {slots:2d} in0_bw {bw:2d}: max |diff| over 10 calls {d:.6f}", flush=True)
ttnn.close_mesh_device(md)
