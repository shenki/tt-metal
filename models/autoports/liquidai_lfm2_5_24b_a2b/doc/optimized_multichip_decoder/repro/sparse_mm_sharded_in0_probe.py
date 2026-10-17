# Stage 5 audit row A4: can the compact expert gate_up sparse_matmul (indexed mode) read the ffn-norm output in its
# width-sharded L1 layout (8 cores x [32, 256]) instead of the interleaved copy (saves one S2I per MoE layer)?
#   TT_VISIBLE_DEVICES=0 python sparse_mm_sharded_in0_probe.py
import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import _ckc, _mcast_1d_config, width_shard_layout

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
E, H, N = 64, 2048, 768
g = torch.Generator().manual_seed(0)
w_t = torch.randn(1, E, H, N, generator=g) * 0.02
x_t = torch.randn(1, 1, 32, H, generator=g)
w = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md)
ids = torch.tensor([3, 17, 40, 63], dtype=torch.int32).reshape(1, 1, 1, 4)
idx = ttnn.from_torch(ids, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
sp = ttnn.from_torch(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
for name, mem in [
    ("interleaved L1", ttnn.L1_MEMORY_CONFIG),
    ("width-sharded 8 cores", width_shard_layout(md, 8, H // 8)),
]:
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=mem)
    try:
        o = ttnn.sparse_matmul(
            x,
            w,
            sparsity=sp,
            indices=idx,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_mcast_1d_config((8, 3, 32, 1), 32, N),
            compute_kernel_config=_ckc(md, "hifi2"),
            dtype=ttnn.bfloat16,
        )
        got = ttnn.to_torch(o).float().reshape(4, 32, N)
        ref = torch.stack([x_t[0, 0] @ w_t[0, e] for e in ids.flatten().tolist()])
        print(
            f"SPMM_IN0 {name:24s}: ok, pcc {torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item():.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"SPMM_IN0 {name:24s}: ERR {str(e).splitlines()[0][:200]}", flush=True)
ttnn.close_mesh_device(md)
