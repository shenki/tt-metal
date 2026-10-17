# Stage 5: prefill all-reduce variants for a [1,1,2048,2048] per-chip partial on the 1x4 ring (traced wall / 20),
# checked against the host sum on all 4 chips:
#   composite ttnn.all_reduce (current, DRAM, bf16) | composite with a bfp8 payload (typecast in / out included)
#   | all_reduce_async on L1 width-sharded (64 / 32 cores) with a persistent buffer (input moved DRAM->L1 and the result
#   back to DRAM inside the measured step, as the decoder would) | reduce_scatter_minimal_async + all_gather_async
#   python prefill_ar_probe.py
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import mesh_device_params
from models.common.modules.tt_ccl import get_tt_ccl

S, H, ND = 2048, 2048, 4
p = mesh_device_params()
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
md = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=p["trace_region_size"], l1_small_size=p["l1_small_size"]
)
ccl = get_tt_ccl(md)
RING, DRAM = ttnn.Topology.Ring, ttnn.DRAM_MEMORY_CONFIG
grid = md.compute_with_storage_grid_size()
g = torch.Generator().manual_seed(0)
parts = torch.randn(ND, 1, 1, S, H, generator=g)
ref = parts.sum(0)
x = ttnn.from_torch(
    torch.cat(list(parts), -1),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    memory_config=DRAM,
    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
)


def wmem(cores, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True), [S, w], ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def timed(f, n=20):
    o = f()
    ttnn.synchronize_device(md)
    err = max(
        (ttnn.to_torch(d).float().reshape(ref.shape) - ref).abs().max().item() for d in ttnn.get_device_tensors(o)
    )
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        o = f()
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / n * 1e6, err
    finally:
        ttnn.release_trace(md, tid)


rows = {}
rows["composite all_reduce bf16 (current)"] = lambda: ttnn.all_reduce(x, num_links=2, topology=RING, memory_config=DRAM)


def bfp8():
    xb = ttnn.typecast(x, ttnn.bfloat8_b)
    r = ttnn.all_reduce(xb, num_links=2, topology=RING, memory_config=DRAM)
    ttnn.deallocate(xb)
    o = ttnn.typecast(r, ttnn.bfloat16)
    ttnn.deallocate(r)
    return o


rows["composite all_reduce bfp8 payload (+2 typecasts)"] = bfp8


def make_ara(cores):
    """all_reduce_async variant; its 4x scratch is allocated when the row runs (L1 holds one at a time)."""
    m_in = wmem(cores, H // cores)
    state = {}

    def ara():
        if "buf" not in state:
            state["buf"] = ttnn.empty(
                (1, 1, S, H * ND), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, wmem(cores, H * ND // cores)
            )
        xs = ttnn.to_memory_config(x, m_in)
        r = ttnn.experimental.all_reduce_async(
            xs,
            state["buf"],
            cluster_axis=1,
            mesh_device=md,
            multi_device_global_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=RING,
            num_links=2,
            memory_config=m_in,
        )
        ttnn.deallocate(xs)
        o = ttnn.to_memory_config(r, DRAM)
        ttnn.deallocate(r)
        return o

    ara.state = state
    return ara


for cores in (64, 32):
    rows[f"all_reduce_async L1 width-sharded {cores} cores (+ DRAM<->L1)"] = make_ara(cores)


def rs_ag():
    rs = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=3,
        cluster_axis=1,
        topology=RING,
        num_links=2,
        multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=DRAM,
    )
    o = ttnn.experimental.all_gather_async(
        rs,
        persistent_output_tensor=None,
        dim=3,
        cluster_axis=1,
        mesh_device=md,
        topology=RING,
        num_links=2,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=DRAM,
    )
    ttnn.deallocate(rs)
    return o


rows["reduce_scatter_minimal_async + all_gather_async"] = rs_ag
for name, f in rows.items():
    try:
        us, err = timed(f)
        print(f"PREFILL_AR {name:58s} {us:8.1f} us  max|err| {err:.3f}", flush=True)
    except Exception as e:
        print(f"PREFILL_AR {name:58s} ERR {str(e).splitlines()[0][:160]}", flush=True)
    if hasattr(f, "state") and "buf" in f.state:
        ttnn.deallocate(f.state.pop("buf"))
ttnn.close_mesh_device(md)
