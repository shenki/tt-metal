# Stage 4: DRAM-sharded decode matmul configs for the per-chip (TP = 4) projection shapes.
# One chip is enough: every chip runs the same local matmul.  Uses DramShardedLinear (decoder layouts:
# input width_shard_layout(cores), output computed row-major grid), traced timing of REPS calls.
#   TT_VISIBLE_DEVICES=<chip> python local_ds_sweep.py <role[,role]> > log
import math
import sys
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, _ckc, width_shard_layout

REPS, ITERS = 16, 8
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 << 20)
grid = md.compute_with_storage_grid_size()
banks = md.dram_grid_size().x
ckc = _ckc(md, "hifi2")
RESID = width_shard_layout(md, 8, 256)  # multichip residual / all-reduce layout
# role: (K, N, pad_n, k_pad, reshard_out_to_resid)
ROLES = {
    "qkv": (2048, 768, False, None, False),
    "wo": (512, 2048, False, None, True),
    "conv_in": (2048, 1536, False, None, False),
    "conv_out": (512, 2048, False, None, True),
    "w13": (2048, 2944, True, None, False),
    "w2_dec": (2944, 2048, False, 3072, True),
    "w2_dec_nopad": (2944, 2048, False, None, True),
}


def timeit(f):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(REPS):
            ttnn.deallocate(f())
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / (ITERS * REPS) * 1e6
    finally:
        ttnn.release_trace(md, tid)


for role in sys.argv[1].split(","):
    K, N, pad_n, k_pad, to_resid = ROLES[role]
    g = torch.Generator().manual_seed(0)
    w = torch.randn(K, N, generator=g) / math.sqrt(K)
    x_t = torch.randn(1, 1, 32, k_pad or K, generator=g)
    if k_pad:
        x_t[..., K:] = 0
    ref = x_t[0, 0, :, :K] @ w
    xi = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    kt = (k_pad or K) // 32
    res = []
    for cores in [c for c in (8, 12, 16, 24, 32, 48, 64, 96) if kt % c == 0]:
        for bw in [b for b in range(1, kt // cores + 1) if (kt // cores) % b == 0 and b <= 16]:
            for readers in (1, 2, 3, 4):
                tag = f"{role} cores={cores} bw={bw} r={readers}"
                try:
                    lin = DramShardedLinear(md, w, ttnn.bfloat8_b, (cores, bw, readers), True, pad_n=pad_n, k_pad=k_pad)
                except Exception as e:
                    print(f"SKIP {tag}: {str(e).splitlines()[0][:120]}", flush=True)
                    continue
                try:
                    xs = ttnn.to_memory_config(xi, lin.in_mem)
                    if to_resid and lin.out_mem != RESID:

                        def f():
                            o1 = lin.decode_sharded(xs, ckc)
                            o2 = ttnn.to_memory_config(o1, RESID)
                            ttnn.deallocate(o1)
                            return o2

                    else:
                        f = lambda: lin.decode_sharded(xs, ckc)
                    o = f()
                    got = ttnn.to_torch(o).float()[0, 0, :, :N]
                    p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                    ttnn.deallocate(o)
                    us = timeit(f)
                    rs = timeit(lambda: ttnn.to_memory_config(xi, lin.in_mem))
                    res.append((us, cores, bw, readers, p, rs, lin.out_mem == RESID))
                    print(
                        f"DS {tag}: {us:7.2f} us (in-reshard from L1 {rs:5.2f}) pcc {p:.5f} out_on_resid {lin.out_mem == RESID}",
                        flush=True,
                    )
                    ttnn.deallocate(xs)
                except Exception as e:
                    print(f"FAIL {tag}: {str(e).splitlines()[0][:150]}", flush=True)
                ttnn.deallocate(lin.weight)
    res.sort()
    for r in res[:5]:
        print(f"BEST {role}: {r[0]:7.2f} us cores={r[1]} bw={r[2]} readers={r[3]} pcc={r[4]:.5f} in-reshard {r[5]:.2f}")
ttnn.close_mesh_device(md)
