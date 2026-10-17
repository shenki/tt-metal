# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head decode config sweep: vocab-parallel DRAM-sharded [2048, 32000] per chip on the 1x4 mesh, input on the
decoder residual grid (8 cores x [32, 256]), output DRAM interleaved (the sampler input).  Wall time per op from a
traced loop of ITERS head calls; correctness vs torch.  Usage (from tt-metal/):
    python models/autoports/liquidai_lfm2_5_24b_a2b/doc/full_model/repro/head_sweep.py [dtype=bfp8]
"""
import sys
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import pcc
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import DRAM
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import _shard_like, mesh_device_params
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import (
    DTYPES,
    DramShardedLinear,
    _ckc,
    width_shard_layout,
)

DT = sys.argv[1] if len(sys.argv) > 1 else "bfp8"
# per-chip vocab columns: 32000 (= 128000 / 4: 125 tiles per DRAM bank, 1 reader only) or 32256 (global vocab padded
# to 129024 at the tail: 126 tiles per bank, 2 or 3 readers)
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 32256
CFGS = (
    [(8, 8, 1), (16, 4, 1), (16, 1, 1), (32, 1, 1), (64, 1, 1)]
    if NV == 32000
    else [
        (8, 8, 2),
        (8, 8, 3),
        (16, 4, 2),
        (16, 4, 3),
        (16, 2, 3),
        (32, 2, 2),
        (32, 2, 3),
        (32, 1, 3),
        (64, 1, 2),
        (64, 1, 3),
    ]
)
ITERS = 50

p = mesh_device_params(64 << 20)
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), l1_small_size=p["l1_small_size"], trace_region_size=p["trace_region_size"]
)
try:
    torch.manual_seed(0)
    W = [torch.randn(2048, NV).to(torch.bfloat16) * 0.02 for _ in range(4)]
    x = torch.randn(1, 1, 32, 2048).to(torch.bfloat16)
    ref = torch.cat([x.float()[0, 0] @ w.float() for w in W], dim=-1)
    resid_mem = width_shard_layout(mesh, 8, 256)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=resid_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ckc = _ckc(mesh, "hifi2")
    for cfg in CFGS:
        try:
            lin = DramShardedLinear(mesh, W[0], DTYPES[DT], cfg, True)
            lin.weight = _shard_like(lin.weight, W, dim=-1)
            o = lin.decode(xt, ckc, DRAM)
            got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()[0, 0]
            ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs = []
            for _ in range(ITERS):
                o = lin.decode(xt, ckc, DRAM)
                ttnn.deallocate(o)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.perf_counter() - t0) / 5 / ITERS * 1e6
            ttnn.release_trace(mesh, tid)
            print(f"HEAD {DT} N={NV} cfg={cfg}: {us:7.1f} us/call  pcc {pcc(got, ref):.5f}", flush=True)
            ttnn.deallocate(lin.weight)
        except Exception as e:
            print(f"HEAD {DT} N={NV} cfg={cfg}: FAIL {str(e).splitlines()[0][:160]}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
