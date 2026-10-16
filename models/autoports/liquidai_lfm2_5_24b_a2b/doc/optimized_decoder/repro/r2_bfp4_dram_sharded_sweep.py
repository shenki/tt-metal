# Parametrized copy of dram_sharded_sweep.py (review P1 #2): DRAM-sharded decode matmul geometry
# under a chosen weight dtype and math fidelity, with inert zero padding so 3 readers per bank is
# legal (N padded so that per-bank tiles % readers == 0; K padded = zero rows in W and zero cols in x).
#   python r2_bfp4_dram_sharded_sweep.py <roles,> <bfp8|bfp4> <hifi2|lofi> [readers=1,2,3] [kpad=0]
# Timing: device-side traced replay (trace of 20 calls, replayed 5x, synchronize).
import math
import sys
import time

import torch

import ttnn

DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}
FI = {"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi, "hifi4": ttnn.MathFidelity.HiFi4}
roles_arg, dt_name, fi_name = sys.argv[1], sys.argv[2], sys.argv[3]
READERS = [int(r) for r in (sys.argv[4] if len(sys.argv) > 4 else "1,2,3").split(",")]
KPAD = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # padded K (0 = none)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
T = 32
dram_banks = md.dram_grid_size().x
grid = md.compute_with_storage_grid_size()
print("dram banks", dram_banks, "grid", grid.x, grid.y, "dtype", dt_name, "fidelity", fi_name, "kpad", KPAD)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=FI[fi_name], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
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
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
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


def crs_for(cores):  # same placement rule as tt/optimized_decoder.width_shard_layout (rect only if it fits)
    if cores % 8 == 0 and cores // 8 <= grid.y:
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, cores // 8 - 1))})
    return ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)


def width_mem(cores, shard_w):
    return ttnn.create_sharded_memory_config(
        (T, shard_w),
        crs_for(cores),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


dram_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))})
for role in roles_arg.split(","):
    k, n = ROLES[role]
    kp = max(k, KPAD)
    g = torch.Generator().manual_seed(0)
    w_t = torch.randn(k, n, generator=g) * 0.02
    x_t = torch.randn(1, 1, T, k, generator=g)
    ref = x_t[0, 0] @ w_t
    xi = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wi = ttnn.from_torch(w_t, dtype=DT[dt_name], layout=ttnn.TILE_LAYOUT, device=md)
    base = timeit(lambda: ttnn.linear(xi, wi, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG))
    ob = ttnn.linear(xi, wi, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
    pb = torch.corrcoef(torch.stack([ttnn.to_torch(ob).float()[0, 0].flatten(), ref.flatten()]))[0, 1].item()
    ttnn.deallocate(ob)
    print(
        f"ROLE {role} K={k} N={n} Kpad={kp}: interleaved {dt_name}/{fi_name} default linear {base:7.1f} us pcc={pb:.5f}",
        flush=True,
    )
    xp_t = torch.zeros(1, 1, T, kp)
    xp_t[..., :k] = x_t
    xpi = ttnn.from_torch(
        xp_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    kt = kp // 32
    results = []
    for readers in READERS:
        unit = 32 * dram_banks * readers
        npad = math.ceil(n / unit) * unit
        wp = torch.zeros(kp, npad)
        wp[:k, :n] = w_t
        wmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_crs, (kp, npad // dram_banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        wd = ttnn.from_torch(wp, dtype=DT[dt_name], layout=ttnn.TILE_LAYOUT, device=md, memory_config=wmem)
        for cores in [c for c in range(8, grid.x * grid.y + 1) if kt % c == 0 and c <= 96]:
            xs = ttnn.to_memory_config(xpi, width_mem(cores, kp // cores))
            per_core_n = math.ceil(npad / 32 / cores)
            omem = width_mem(cores, per_core_n * 32)
            for bw in [b for b in range(1, kt // cores + 1) if (kt // cores) % b == 0]:
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=bw,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fused_activation=None,
                        num_workers_per_dram_bank=readers,
                    )
                    f = lambda: ttnn.linear(xs, wd, program_config=pc, compute_kernel_config=ckc, memory_config=omem)
                    o = f()
                    p = torch.corrcoef(torch.stack([ttnn.to_torch(o).float()[0, 0, :, :n].flatten(), ref.flatten()]))[
                        0, 1
                    ].item()
                    ttnn.deallocate(o)
                    us = timeit(f)
                    results.append((us, cores, bw, readers, npad, p))
                    print(f"    try cores={cores} bw={bw} r={readers} npad={npad}: {us:.1f} us pcc={p:.5f}", flush=True)
                except Exception as e:
                    print(
                        f"    FAIL cores={cores} bw={bw} r={readers} npad={npad}: {str(e).splitlines()[0][:150]}",
                        flush=True,
                    )
            ttnn.deallocate(xs)
        ttnn.deallocate(wd)
    results.sort()
    for r in results[:6]:
        print(
            f"  DRAMSHARD {role} {dt_name}/{fi_name}: {r[0]:7.1f} us cores={r[1]} in0_bw={r[2]} readers={r[3]} npad={r[4]} kpad={kp} pcc={r[5]:.5f}"
        )
    for rd in READERS:
        best = [r for r in results if r[3] == rd][:1]
        for r in best:
            print(
                f"  BEST_R{rd} {role} {dt_name}/{fi_name}: {r[0]:7.1f} us cores={r[1]} in0_bw={r[2]} npad={r[4]} kpad={kp}"
            )
ttnn.close_mesh_device(md)
