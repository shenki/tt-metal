# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S3: ttnn.topk(k=32) on the sampler's per-chip [1, 1, 32, 32768] bf16 logits, stable vs unstable (traced wall time per
call), and whether stable top-k returns the lowest index among tied maxima (the property the TTSampling tie-break pass
provides for greedy rows)."""
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import mesh_device_params

p = mesh_device_params(64 << 20)
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), l1_small_size=p["l1_small_size"], trace_region_size=p["trace_region_size"]
)
ITERS = 50
try:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 4 * 32768).to(torch.bfloat16)
    x[0, 0, :, 100] = x[0, 0, :, 7000] = x[0, 0, :, 20000] = 50.0  # tied maxima on chip 0
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    for stable in (False, True):
        f = lambda: ttnn.topk(xt, k=32, dim=-1, stable=stable)
        v, i = f()
        idx = ttnn.to_torch(ttnn.get_device_tensors(i)[0]).long()[0, 0]
        first_ties = idx[:, :3].sort(dim=-1).values[0].tolist(), idx[:, 0].tolist()[:4]
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(ITERS):
            a, b = f()
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, tid)
        us = (time.perf_counter() - t0) / 5 / ITERS * 1e6
        print(
            f"TOPK stable={stable}: {us:.1f} us/call; tied maxima at top-3 {first_ties[0]}, top-1 per row {first_ties[1]}",
            flush=True,
        )
finally:
    ttnn.close_mesh_device(mesh)
