import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
K, N, M = 2048, 3072, 2048
banks = md.dram_grid_size().x
g = torch.Generator().manual_seed(0)
w_t = torch.randn(K, N, generator=g) * 0.02
x_t = torch.randn(1, 1, M, K, generator=g)
crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
wmem = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.DRAM,
    ttnn.ShardSpec(crs, (K, N // banks), ttnn.ShardOrientation.ROW_MAJOR),
)
wd = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=wmem)
wi = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md)
x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
ref = x_t[0, 0] @ w_t
grid = md.compute_with_storage_grid_size()
pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=4,
    out_block_h=M // 32 // 8,
    out_block_w=N // 32 // 8,
    per_core_M=M // 32 // 8,
    per_core_N=N // 32 // 8,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)
for name, w in [("interleaved", wi), ("dram-sharded", wd), ("dram-sharded+2d", wd)]:
    kw = {"program_config": pc} if name.endswith("2d") else {}
    try:
        o = ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        p = torch.corrcoef(torch.stack([ttnn.to_torch(o).float()[0, 0].flatten(), ref.flatten()]))[0, 1].item()
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw))
        ttnn.synchronize_device(md)
        print(f"PROBE prefill M={M} weight {name}: ok pcc {p:.5f} {(time.perf_counter()-t0)/10*1e6:.0f} us")
    except Exception as e:
        print(f"PROBE prefill weight {name}: FAIL {str(e).splitlines()[0][:200]}")
ttnn.close_mesh_device(md)
