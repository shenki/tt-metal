# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro: DRAM-sharded matmul ([2048, 32256] bfp8 per chip, (64 cores, in0 1, 3 readers)) with a fused bias
[1, 1, 1, N]: the bias is applied to the wrong output columns (only ~1/8 of the intended tail columns, some others hit)."""
import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import DRAM
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import _shard_like, mesh_device_params
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import (
    DTYPES,
    DramShardedLinear,
    _ckc,
    width_shard_layout,
)

p = mesh_device_params(64 << 20)
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), l1_small_size=p["l1_small_size"], trace_region_size=p["trace_region_size"]
)
try:
    N = 32256
    x = torch.randn(1, 1, 32, 2048).to(torch.bfloat16)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=width_shard_layout(mesh, 8, 256),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    W = [torch.randn(2048, N).to(torch.bfloat16) * 0.02 for _ in range(4)]
    lin = DramShardedLinear(mesh, W[0], DTYPES["bfp8"], (64, 1, 3), True)
    lin.weight = _shard_like(lin.weight, W, dim=-1)
    ckc = _ckc(mesh, "hifi2")
    for val in (5.0, -1e4, -3e38):
        bias = torch.zeros(4, N)
        bias[:, -1024:] = val
        b = ttnn.from_torch(
            torch.cat([bias[d].reshape(1, 1, 1, N) for d in range(4)], dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        xs = ttnn.to_memory_config(xt, lin.in_mem)
        o = ttnn.linear(
            xs, lin.weight, bias=b, program_config=lin.decode_cfg, compute_kernel_config=ckc, memory_config=lin.out_mem
        )
        g = ttnn.to_torch(ttnn.to_memory_config(o, DRAM), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()[
            0, 0
        ]
        ref = torch.cat([x.float()[0, 0] @ w.float() for w in W], dim=-1)
        d = (g - ref).reshape(32, 4, N)
        print(
            val,
            "tail diff mean",
            d[:, :, -1024:].mean().item(),
            "head diff absmax",
            d[:, :, :-1024].abs().max().item(),
            "tail min",
            g.reshape(32, 4, N)[:, :, -1024:].min().item(),
            "rows",
            d[:, 0, -1].tolist()[:4],
        )
finally:
    ttnn.close_mesh_device(mesh)
