# Stage-4 CCL topology probe on the 1x4 Blackhole ring (P150_X4): cost of restoring the residual after a
# row-parallel / expert-parallel partial, decode [1,1,32,2048] and prefill [1,1,2048,2048] bf16.
# Each variant: traced replay of the collective (plus its local epilogue), wall / N, and a correctness check
# against the host sum of the four per-chip partials.
#   python models/autoports/liquidai_lfm2_5_24b_a2b/doc/multichip_decoder/repro/ccl_topology_probe.py [decode] [prefill]
import os
import sys
import time

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL

router = ttnn.FabricRouterConfig()
router.max_packet_payload_size_bytes = 8192
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 << 20, l1_small_size=24576)
ccl = TT_CCL(md)
RING = ttnn.Topology.Ring
DRAM = ttnn.DRAM_MEMORY_CONFIG
H, ND = 2048, 4
grid = md.compute_with_storage_grid_size()
sections = set(sys.argv[1:]) or {"decode", "prefill"}
print(f"CCL grid {grid.x}x{grid.y} links axis1 {ttnn.get_num_links(md, 1)}", flush=True)


def width_mem(cores, h, w):
    g = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(g, [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def timed(f, n=50):
    out = f()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    o2 = f()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        us = (time.perf_counter() - t0) / n * 1e6
    finally:
        ttnn.release_trace(md, tid)
    return us, out


def check(out, ref, full_width=True):
    devs = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    if full_width:
        err = max((d.reshape(ref.shape) - ref).abs().max().item() for d in devs)
    else:  # fractured: chip d holds columns d*W..(d+1)*W
        w = ref.shape[-1] // ND
        err = max(
            (d.reshape(*ref.shape[:-1], w) - ref[..., i * w : (i + 1) * w]).abs().max().item()
            for i, d in enumerate(devs)
        )
    return err


def run(rows):
    g = torch.Generator().manual_seed(rows)
    parts = torch.randn(ND, 1, 1, rows, H, generator=g)
    ref = parts.sum(0)
    host = torch.cat(list(parts), dim=-1)  # [1,1,rows,4H] -> chip d gets partial d
    tag = "decode" if rows == 32 else f"prefill{rows}"
    mems = {"dram": DRAM}
    if rows == 32:
        mems["l1w32"] = width_mem(32, 32, H // 32)
        mems["l1"] = ttnn.L1_MEMORY_CONFIG
    for mname, mem in mems.items():
        x = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=mem,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
        )
        for links in [int(v) for v in os.environ.get("CCL_LINKS", "1,2").split(",")]:
            variants = {}
            # 1. composite all-reduce
            variants["ttnn.all_reduce"] = (
                lambda: ttnn.all_reduce(x, num_links=links, topology=RING, memory_config=mem),
                True,
            )
            # 2. direct all_reduce_async with a persistent buffer (Qwen QB2 decode recipe)
            if rows == 32:
                buf_mem = width_mem(32, 32, H * ND // 32) if mname == "l1w32" else mem
                buf = ttnn.from_torch(
                    torch.zeros(1, 1, rows, H * ND),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=buf_mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                )
                variants["all_reduce_async+buf"] = (
                    lambda: ttnn.experimental.all_reduce_async(
                        x,
                        buf,
                        cluster_axis=1,
                        mesh_device=md,
                        multi_device_global_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                        topology=RING,
                        num_links=links,
                        memory_config=mem,
                    ),
                    True,
                )

            # 3. reduce-scatter (fractured residual) and 4. RS + AG
            def rs():
                return ttnn.experimental.reduce_scatter_minimal_async(
                    x,
                    persistent_output_buffers=None,
                    dim=3,
                    cluster_axis=1,
                    topology=RING,
                    num_links=links,
                    multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
                    barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                    memory_config=DRAM if mname == "l1w32" else mem,
                )

            def ag(t, dim=3):
                return ttnn.experimental.all_gather_async(
                    t,
                    persistent_output_tensor=None,
                    dim=dim,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=RING,
                    num_links=links,
                    multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
                    barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
                    memory_config=DRAM if mname == "l1w32" else mem,
                )

            variants["RS(dim3)"] = (rs, False)
            variants["RS+AG(dim3)"] = (lambda: ag(rs()), True)

            # 5. all-gather partials along rows then local sum (one CCL, 4x bytes)
            def ag_sum():
                gth = ag(x, dim=2)  # [1,1,4*rows,H]
                r = ttnn.reshape(gth, [ND, 1, rows, H])
                return ttnn.sum(r, dim=0, keepdim=True)

            if rows == 32:
                variants["AG(dim2)+sum"] = (ag_sum, True)
            # 6. all-gather only (for column-parallel input gathers: [rows, H/4] -> [rows, H])
            xq = ttnn.from_torch(
                host[..., :H],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=DRAM if mname == "l1w32" else mem,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
            )
            variants["AG(dim3) of H/4"] = (lambda: ag(xq), None)

            for vname, (f, full) in variants.items():
                try:
                    us, out = timed(f)
                    err = "-" if full is None else f"{check(out, ref, full):.4f}"
                    print(f"CCL {tag:12s} {mname:6s} links {links} {vname:24s} {us:9.1f} us  maxerr {err}", flush=True)
                except Exception as e:
                    print(
                        f"CCL {tag:12s} {mname:6s} links {links} {vname:24s} ERR {str(e).splitlines()[0][:140]}",
                        flush=True,
                    )


if "decode" in sections:
    run(32)
if "prefill" in sections:
    run(2048)
ttnn.close_mesh_device(md)
