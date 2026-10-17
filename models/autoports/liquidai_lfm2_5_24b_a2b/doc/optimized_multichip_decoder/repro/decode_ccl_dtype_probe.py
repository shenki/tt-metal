# Stage 5: decode all-reduce payload dtype. DS wo [512, 2048] per chip writes its partial as bf16 (current) or bfp8
# (dtype= on the matmul, no extra op), all_reduce_async on the 8-core residual layout, bf16 output; traced wall / 50.
#   python decode_ccl_dtype_probe.py
import math
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import MeshPlan, MeshReduce, mesh_device_params
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, _ckc

p = mesh_device_params()
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
md = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=p["trace_region_size"], l1_small_size=p["l1_small_size"]
)
ckc = _ckc(md, "hifi2")
g = torch.Generator().manual_seed(0)
wo = [torch.randn(512, 2048, generator=g) / math.sqrt(512) for _ in range(4)]
a = [torch.randn(32, 512, generator=g) for _ in range(4)]
ref = sum(x @ w for x, w in zip(a, wo))
lin = DramShardedLinear(md, wo[0], ttnn.bfloat8_b, (8, 2, 2), True)
lin.weight = ttnn.from_torch(
    torch.cat(wo, -2),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    memory_config=lin.weight.memory_config(),
    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-2),
)
xin = ttnn.from_torch(
    torch.cat([t.reshape(1, 1, 32, 512) for t in a], -1),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    memory_config=lin.in_mem,
    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
)
red = MeshReduce(md, MeshPlan(), 2048)
for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
    buf = ttnn.empty((1, 1, 32, 8192), dt, ttnn.TILE_LAYOUT, md, red.buf_mem)

    def step():
        part = ttnn.linear(
            xin,
            lin.weight,
            program_config=lin.decode_cfg,
            compute_kernel_config=ckc,
            memory_config=lin.out_mem,
            dtype=dt,
        )
        r = ttnn.experimental.all_reduce_async(
            part,
            buf,
            cluster_axis=1,
            mesh_device=md,
            multi_device_global_semaphore=red.ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=ttnn.Topology.Ring,
            num_links=2,
            memory_config=red.resid_mem,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(part)
        return r

    try:
        o = step()
        err = max(
            (ttnn.to_torch(d).float().reshape(32, 2048) - ref).abs().max().item() for d in ttnn.get_device_tensors(o)
        )
        pc = min(
            torch.corrcoef(torch.stack([ttnn.to_torch(d).float().flatten(), ref.flatten()]))[0, 1].item()
            for d in ttnn.get_device_tensors(o)
        )
        ttnn.deallocate(o)
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        o = step()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(50):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        us = (time.perf_counter() - t0) / 50 * 1e6
        ttnn.release_trace(md, tid)
        print(
            f"CCL_DTYPE partial {dt}: DS wo + all_reduce_async {us:7.1f} us  max|err| {err:.4f}  pcc {pc:.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"CCL_DTYPE partial {dt}: ERR {str(e).splitlines()[0][:160]}", flush=True)
    ttnn.deallocate(buf)
ttnn.close_mesh_device(md)
