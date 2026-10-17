# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head output-layout options for folding the invalid-vocab mask into the head (stage 7, S1): traced per-call wall time
on the 1x4 mesh, input on the decoder residual grid, output DRAM interleaved bf16.
  * N = 32256 per chip (current) with / without a fused bias (bias layouts tried: [1,1,32,N] and [1,1,1,N] tile DRAM);
  * N = 32768 per chip (pow2 shard: no sampler pad / tail mask) for 1 / 2-reader configs, with / without bias;
  * the stand-alone mask add on [1,1,32,32768] (fallback if no bias).
Usage (from tt-metal/): python <this>"""
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

ITERS = 50
p = mesh_device_params(64 << 20)
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), l1_small_size=p["l1_small_size"], trace_region_size=p["trace_region_size"]
)


def timed(fn):
    o = fn()
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(ITERS):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ttnn.release_trace(mesh, tid)
    return (time.perf_counter() - t0) / 5 / ITERS * 1e6


try:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 2048).to(torch.bfloat16)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=width_shard_layout(mesh, 8, 256),
        mesh_mapper=rep,
    )
    ckc = _ckc(mesh, "hifi2")
    for N, cfgs in (
        (32256, [(64, 1, 3)]),
        (32768, [(64, 1, 2), (32, 2, 2), (32, 1, 2), (16, 4, 2), (16, 2, 2), (64, 1, 1)]),
    ):
        W = [torch.randn(2048, N).to(torch.bfloat16) * 0.02 for _ in range(4)]
        ref = torch.cat([x.float()[0, 0] @ w.float() for w in W], dim=-1)
        bias = torch.zeros(4, N)
        bias[3, N - 1024 :] = -3.0e38  # chip 3's padded tail
        for cfg in cfgs:
            try:
                lin = DramShardedLinear(mesh, W[0], DTYPES["bfp8"], cfg, True)
                lin.weight = _shard_like(lin.weight, W, dim=-1)
                us = timed(lambda: lin.decode(xt, ckc, DRAM))
                msg = f"HEAD N={N} cfg={cfg}: {us:7.1f} us"
                for rows in (32, 1):
                    try:
                        b = ttnn.from_torch(
                            bias.reshape(4, 1, 1, N).expand(4, 1, rows, N).reshape(1, 1, 4 * rows, N)
                            if False
                            else torch.cat(
                                [bias[d].reshape(1, 1, 1, N).expand(1, 1, rows, N) for d in range(4)], dim=-1
                            ),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh,
                            memory_config=DRAM,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                        )

                        def biased():
                            xs = ttnn.to_memory_config(xt, lin.in_mem)
                            o = ttnn.linear(
                                xs,
                                lin.weight,
                                bias=b,
                                program_config=lin.decode_cfg,
                                compute_kernel_config=ckc,
                                memory_config=lin.out_mem,
                            )
                            ttnn.deallocate(xs)
                            oi = ttnn.to_memory_config(o, DRAM)
                            ttnn.deallocate(o)
                            return oi

                        ub = timed(biased)
                        got = ttnn.to_torch(biased(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()[0, 0]
                        ok = (
                            bool((got[:, 4 * N - 1024 :] < -1e37).all())
                            and pcc(got[:, : 4 * N - 1024], ref[:, : 4 * N - 1024]) > 0.9999
                        )
                        msg += f" | bias rows={rows}: {ub:7.1f} us mask_ok={ok}"
                    except Exception as e:
                        msg += f" | bias rows={rows}: FAIL {str(e).splitlines()[0][:90]}"
                print(msg, flush=True)
                ttnn.deallocate(lin.weight)
            except Exception as e:
                print(f"HEAD N={N} cfg={cfg}: FAIL {str(e).splitlines()[0][:150]}", flush=True)
    lg = ttnn.from_torch(
        torch.randn(1, 1, 32, 4 * 32768),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    mk = ttnn.from_torch(
        torch.zeros(1, 1, 32, 4 * 32768),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    print(f"MASK ADD [32, 32768]: {timed(lambda: ttnn.add(lg, mk, memory_config=DRAM)):.1f} us", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
