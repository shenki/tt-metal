import math
import sys
import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
T = 32
dram_banks = md.dram_grid_size().x
grid = md.compute_with_storage_grid_size()
print("dram banks", dram_banks, "grid", grid.x, grid.y)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
ROLES = {
    "wqkv": (2048, 3072),
    "wo": (2048, 2048),
    "conv_in": (2048, 6144),
    "conv_out": (2048, 2048),
    "w1": (2048, 11776),
    "w2": (11776, 2048),
}


def timeit(f, n=20, reps=5):
    """device-side time per call: trace of n calls (each output freed right away, so nothing
    accumulates in L1), replayed reps times. Capture is always closed, even on error."""
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)  # compile + validate outside capture
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(n):
            ttnn.deallocate(f())
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / (n * reps) * 1e6
    finally:
        ttnn.release_trace(md, tid)


def width_mem(cores, k):
    crs = ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return ttnn.create_sharded_memory_config(
        (T, k // cores),
        crs,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


dram_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))})
for role in sys.argv[1].split(","):
    k, n = ROLES[role]
    g = torch.Generator().manual_seed(0)
    w_t = torch.randn(k, n, generator=g) * 0.02
    x_t = torch.randn(1, 1, T, k, generator=g)
    xi = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wi = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md)
    ref = x_t[0, 0] @ w_t
    base = timeit(lambda: ttnn.linear(xi, wi, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG))
    print(f"ROLE {role} K={k} N={n}: interleaved bfp8 linear {base:7.1f} us")
    npad = math.ceil(n / (32 * dram_banks)) * 32 * dram_banks
    wp = torch.zeros(k, npad)
    wp[:, :n] = w_t
    wmem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, (k, npad // dram_banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    wd = ttnn.from_torch(wp, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=wmem)
    kt = k // 32
    results = []
    for cores in [c for c in range(8, grid.x * grid.y + 1) if kt % c == 0 and c <= 96]:
        in_mem = width_mem(cores, k)
        xs = ttnn.to_memory_config(xi, in_mem)
        per_core_n = math.ceil(npad / 32 / cores)
        out_mem = width_mem(cores, per_core_n * 32 * cores) if False else None
        for bw in [b for b in range(1, kt // cores + 1) if (kt // cores) % b == 0]:
            for readers in (1, 2, 3):
                if (npad // dram_banks // 32) % readers:
                    continue
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=bw,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fused_activation=None,
                        num_workers_per_dram_bank=readers,
                    )
                    omem = ttnn.create_sharded_memory_config(
                        (T, per_core_n * 32),
                        ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
                        ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    f = lambda: ttnn.linear(xs, wd, program_config=pc, compute_kernel_config=ckc, memory_config=omem)
                    o = f()
                    pcc_ok = torch.corrcoef(
                        torch.stack([ttnn.to_torch(o).float()[0, 0, :, :n].flatten(), ref.flatten()])
                    )[0, 1].item()
                    ttnn.deallocate(o)
                    us = timeit(f)
                    results.append((us, cores, bw, readers, pcc_ok))
                    print(f"    try cores={cores} bw={bw} r={readers}: {us:.1f}", flush=True)
                except Exception as e:
                    print(f"    FAIL cores={cores} bw={bw} r={readers}: {str(e).splitlines()[0][:150]}", flush=True)
        ttnn.deallocate(xs)
    results.sort()
    for r in results[:4]:
        print(f"  DRAMSHARD {role}: {r[0]:7.1f} us cores={r[1]} in0_bw={r[2]} readers={r[3]} pcc={r[4]:.5f}")
    if results:
        c = results[0][1]
        in_mem = width_mem(c, k)
        rs_in = timeit(lambda: ttnn.to_memory_config(xi, in_mem))
        print(f"  reshard interleaved->width({c} cores) {rs_in:6.1f} us")
ttnn.close_mesh_device(md)
