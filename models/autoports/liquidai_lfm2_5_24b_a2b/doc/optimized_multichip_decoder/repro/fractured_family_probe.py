# Stage 5, family "residual layout + collective placement + fused matmul-CCL" (attention sublayer boundary, decode 32 rows).
# Measured under each chain's OWN residual contract (no restore to the replicated residual inside a chain):
#   A   replicated residual [32, 2048] (8-core width-sharded L1, the stage-4 contract): ShardedNorm -> DS qkv [2048, 768]
#       | DS wo [512, 2048] partial -> all_reduce_async (persistent scratch) -> residual add
#   F   fractured residual (chip d holds columns 512d..512d+511, [32, 512]): dit_fused_distributed_rmsnorm (stats AG fused)
#       -> all_gather_minimal_matmul_async(qkv) | minimal_matmul_strided_reduce_scatter_async(wo) -> fractured add
#   F2  fractured residual: dit_fused_distributed_rmsnorm -> all_gather_async(xn) -> DS qkv | DS wo -> reduce_scatter_minimal_async
#       -> fractured add
# The attention core between qkv and wo is outside the chains (same in all of them).  Each chain: eager pass checked
# against torch on all 4 chips, then traced replay timing (wall / N; device-bound at these sizes).
#   python fractured_family_probe.py
import math
import time
import traceback

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import MeshPlan, MeshReduce, mesh_device_params
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, ShardedNorm, _ckc
from models.common.modules.tt_ccl import get_tt_ccl

TILE, H, TP, NQKV, NA = 32, 2048, 4, 768, 512
p = mesh_device_params()
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
md = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=p["trace_region_size"], l1_small_size=p["l1_small_size"]
)
ccl = get_tt_ccl(md)
RING = ttnn.Topology.Ring
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
ckc = _ckc(md, "hifi2")
eps = 1e-5
g = torch.Generator().manual_seed(0)
x_t = torch.randn(TILE, H, generator=g)  # residual (identical rows on all chips in A)
gamma = torch.rand(H, generator=g) + 0.5
wqkv = [torch.randn(H, NQKV, generator=g) / math.sqrt(H) for _ in range(TP)]  # per chip column block
wo = [torch.randn(NA, H, generator=g) / math.sqrt(NA) for _ in range(TP)]  # per chip row block
attn = [torch.randn(TILE, NA, generator=g) for _ in range(TP)]  # per chip attention output (stand-in)
xn_ref = x_t * torch.rsqrt(x_t.pow(2).mean(-1, keepdim=True) + eps) * gamma
qkv_ref = [xn_ref @ w for w in wqkv]
out_ref = x_t + sum(a @ w for a, w in zip(attn, wo))


def shard(ts, dim, dtype=ttnn.bfloat16, mem=DRAM, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.cat(ts, dim=dim),
        dtype=dtype,
        layout=layout,
        device=md,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=dim),
    )


def repl(t, dtype=ttnn.bfloat16, mem=DRAM, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=md, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
    )


def devs(t):
    return [ttnn.to_torch(d).float().reshape(TILE, -1) for d in ttnn.get_device_tensors(t)]


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def timed(step, n=50):
    outs = step()
    ttnn.synchronize_device(md)
    for o in outs:
        ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        outs = step()
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / n * 1e6
    finally:
        ttnn.release_trace(md, tid)


results = {}
# ---------------- A: replicated residual (stage 4) ----------------
try:
    norm = ShardedNorm(md, H, 8, eps, _ckc(md, "hifi4"))
    gam_rm = repl(gamma.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
    ds_qkv = DramShardedLinear(md, wqkv[0], ttnn.bfloat8_b, (8, 8, 3), True)
    ds_qkv.weight = shard([w for w in wqkv], -1, ttnn.bfloat8_b, ds_qkv.weight.memory_config())
    ds_wo = DramShardedLinear(md, wo[0], ttnn.bfloat8_b, (8, 2, 2), True)
    ds_wo.weight = shard(wo, -2, ttnn.bfloat8_b, ds_wo.weight.memory_config())
    red = MeshReduce(md, MeshPlan(), H)
    xr = repl(x_t.reshape(1, 1, TILE, H), mem=red.resid_mem)
    a_in = shard([a.reshape(1, 1, TILE, NA) for a in attn], -1, mem=ds_wo.in_mem)

    def chain_a():
        xn = norm(xr, gam_rm)
        qkv = ds_qkv.decode_sharded(xn, ckc)
        ttnn.deallocate(xn)
        part = ds_wo.decode_sharded(a_in, ckc)
        r = red.decode(part)
        out = ttnn.add(xr, r, memory_config=red.resid_mem)
        ttnn.deallocate(r)
        return [qkv, out]

    qkv, out = chain_a()
    ok = (min(pcc(d, q) for d, q in zip(devs(qkv), qkv_ref)), min(pcc(d, out_ref) for d in devs(out)))
    ttnn.deallocate(qkv)
    ttnn.deallocate(out)
    results["A replicated: norm+DSqkv | DSwo+AR+add"] = (timed(chain_a), ok)
except Exception as e:
    results["A"] = ("ERR " + str(e).splitlines()[0][:200], None)
    traceback.print_exc()

# ---------------- F: fractured residual, fused CCL ops ----------------
x_frac = [x_t[:, d * NA : (d + 1) * NA] for d in range(TP)]
try:
    xf = shard([t.reshape(1, 1, TILE, NA) for t in x_frac], -1)
    gam_f = shard([gamma[d * NA : (d + 1) * NA].reshape(1, 1, 1, NA) for d in range(TP)], -1)  # TILE, [1, H/tp]
    stats = ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(xf, 1, md, num_links=1, weight=gam_f)
    w_qkv_f = shard([w.reshape(1, 1, H, NQKV) for w in wqkv], -1, ttnn.bfloat8_b)
    w_o_f = shard([w.reshape(1, 1, NA, H) for w in wo], -2, ttnn.bfloat8_b)
    a_f = shard([a.reshape(1, 1, TILE, NA) for a in attn], -1)
    WORKERS = 5
    mm_cfg = lambda n: ttnn.MinimalMatmulConfig(
        M_block_size=1,
        K_block_size=8,
        N_block_size=min(8, n // TILE),
        subblock_h=1,
        subblock_w=4 if (n // TILE) % 4 == 0 else 2,
        compute_with_storage_grid_size=(10, 8),
    )

    def fused_norm(x):
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            1,
            md,
            ccl.get_and_cycle_ag_semaphore_handles(1),
            topology=RING,
            epsilon=eps,
            weight=gam_f,
            persistent_output_buffer=stats,
            memory_config=DRAM,
        )

    def chain_f():
        xn = fused_norm(xf)
        qkv = ttnn.experimental.all_gather_minimal_matmul_async(
            xn,
            w_qkv_f,
            config=mm_cfg(NQKV),
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
            topology=RING,
            cluster_axis=1,
            num_links=2,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
            compute_kernel_config=ckc,
            memory_config=DRAM,
            dtype=ttnn.bfloat16,
            num_workers_per_link=WORKERS,
            force_transpose=True,
        )[0]
        ttnn.deallocate(xn)
        part = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            a_f,
            w_o_f,
            3,
            ccl.get_and_cycle_rs_semaphore_handles(1),
            ttnn.CoreCoord(0, 8),
            compute_kernel_config=ckc,
            num_links=2,
            topology=RING,
            cluster_axis=1,
            config=mm_cfg(H),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config_mm=DRAM,
            rs_output_mem_config=DRAM,
            dtype=ttnn.bfloat16,
        )[1]
        out = ttnn.add(xf, part, memory_config=DRAM)
        ttnn.deallocate(part)
        return [qkv, out]

    import os

    for gx, gy in [tuple(int(v) for v in os.environ.get("F_GRID", "10x8").split("x"))]:  # one grid per process
        mm_cfg = lambda n, gx=gx, gy=gy: ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=8,
            N_block_size=min(8, n // TILE),
            subblock_h=1,
            subblock_w=4 if (n // TILE) % 4 == 0 else 2,
            compute_with_storage_grid_size=(gx, gy),
        )
        WORKERS = gx // 2  # all_gather_minimal_matmul_async: in0 sender axis = links x workers
        try:
            qkv, out = chain_f()
            q_d, o_d = devs(qkv), devs(out)
            ok = (
                min(pcc(q, r) for q, r in zip(q_d, qkv_ref)),
                min(pcc(o, out_ref[:, d * NA : (d + 1) * NA]) for d, o in enumerate(o_d)),
            )
            ttnn.deallocate(qkv)
            ttnn.deallocate(out)
            results[f"F fractured fused (mm grid {gx}x{gy}): dit_norm+AGMM | MMRS+add"] = (timed(chain_f), ok)
        except Exception as e:
            results[f"F grid {gx}x{gy}"] = ("ERR " + str(e).splitlines()[0][:200], None)
except Exception as e:
    results["F"] = ("ERR " + str(e).splitlines()[0][:220], None)
    traceback.print_exc()

# ---------------- F2: fractured residual, unfused CCL + DS matmuls ----------------
try:
    xf = shard([t.reshape(1, 1, TILE, NA) for t in x_frac], -1)

    def chain_f2():
        xn = fused_norm(xf)
        xg = ttnn.experimental.all_gather_async(
            xn,
            persistent_output_tensor=None,
            dim=3,
            cluster_axis=1,
            mesh_device=md,
            topology=RING,
            num_links=2,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config=L1,
        )
        ttnn.deallocate(xn)
        qkv = ds_qkv.decode_sharded(xg, ckc)
        ttnn.deallocate(xg)
        part = ds_wo.decode_sharded(a_in, ckc)
        rs = ttnn.experimental.reduce_scatter_minimal_async(
            part,
            persistent_output_buffers=None,
            dim=3,
            cluster_axis=1,
            topology=RING,
            num_links=2,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
            memory_config=DRAM,
        )
        ttnn.deallocate(part)
        out = ttnn.add(xf, rs, memory_config=DRAM)
        ttnn.deallocate(rs)
        return [qkv, out]

    qkv, out = chain_f2()
    q_d, o_d = devs(qkv), devs(out)
    ok = (
        min(pcc(q, r) for q, r in zip(q_d, qkv_ref)),
        min(pcc(o, out_ref[:, d * NA : (d + 1) * NA]) for d, o in enumerate(o_d)),
    )
    ttnn.deallocate(qkv)
    ttnn.deallocate(out)
    results["F2 fractured unfused: dit_norm+AG+DSqkv | DSwo+RS+add"] = (timed(chain_f2), ok)
except Exception as e:
    results["F2"] = ("ERR " + str(e).splitlines()[0][:220], None)
    traceback.print_exc()

for k, (us, ok) in results.items():
    print(f"FAMILY {k:58s} {us if isinstance(us, str) else f'{us:8.1f} us'}  pcc(qkv, out) {ok}", flush=True)
ttnn.close_mesh_device(md)
