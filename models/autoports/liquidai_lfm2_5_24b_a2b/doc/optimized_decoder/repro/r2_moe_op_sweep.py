# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced (device-accurate) sweep of the decode expert sparse_matmuls at a small active count.

    TT_VISIBLE_DEVICES=<chip> python r2_moe_op_sweep.py <role> <mode> <dtypes> <fids> [n_active] > log

role: gate_up [32x2048 @ 2048x3072] | up_half [32x2048 @ 2048x1536] (split gate_up) | down [32x1536 @ 1536x2048]
mode: scan (current: dense sparsity scan, nnz=None) | nnz (scan with explicit nnz) | idx (indexed/gather mode)
Timing: one trace holding REPS calls (outputs freed inside the capture), replayed ITERS times,
synchronize; per-call us = wall / (ITERS*REPS).  Prints one JSON line per config.
"""

import json
import sys
import time

import torch

import ttnn

E, H, I, M = 64, 2048, 1536, 32
REPS, ITERS = 16, 8
TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}

role, mode = sys.argv[1], sys.argv[2]
dtypes, fids = sys.argv[3].split(","), sys.argv[4].split(",")
n_active = int(sys.argv[5]) if len(sys.argv) > 5 else 4
only = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None  # optional explicit config list

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
grid_max = md.compute_with_storage_grid_size()


def ckc(fid):
    return ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def cfg(gx, gy, in0_bw, sub_w, pcn):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


K, N = {"gate_up": (H, 2 * I), "up_half": (H, I), "down": (I, H)}[role]
kt, nt = K // 32, N // 32
g = torch.Generator().manual_seed(0)
w_t = torch.randn(1, E, K, N, generator=g) * 0.02
ids = torch.randperm(E, generator=g)[:n_active]
sp_t = torch.zeros(1, 1, 1, E)
sp_t[..., ids] = 1.0
sp = ttnn.from_torch(sp_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
idx = ttnn.from_torch(
    ids.to(torch.int32).reshape(1, 1, 1, -1), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md
)
if role == "down":
    a_t = torch.randn(1, n_active if mode == "idx" else E, M, K, generator=g)
else:
    a_t = torch.randn(1, 1, M, K, generator=g)
a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)


def ref():
    if role == "down":
        if mode == "idx":
            return torch.stack([a_t[0, i] @ w_t[0, e] for i, e in enumerate(ids)])
        return torch.stack([a_t[0, e] @ w_t[0, e] for e in ids])
    return torch.stack([a_t[0, 0] @ w_t[0, e] for e in ids])


def pick(out):
    o = ttnn.to_torch(out).float()
    if mode == "idx":
        return o.reshape(n_active, M, N)
    return o.reshape(E, M, N)[ids]


def pcc(x, y):
    return torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()


def candidates():
    if only:
        for c in only:
            yield tuple(c)
        return
    grids = [(11, 10), (11, 9), (8, 8), (8, 6), (8, 4), (12, 8), (8, 3), (11, 6), (6, 4), (8, 2), (4, 4)]
    seen = set()
    for gx, gy in grids:
        if gx > grid_max.x or gy > grid_max.y:
            continue
        pcn = -(-nt // (gx * gy))
        # smallest grid with this per-core N (fewer idle cores for the same per-core work)
        if pcn in seen and gx * gy > nt:
            continue
        seen.add(pcn)
        for bw in [b for b in (2, 4, 8, 16, 24, 32, 48) if kt % b == 0]:
            for sw in [s for s in (1, 2, 3, 4) if pcn % s == 0]:
                yield (gx, gy, bw, sw, pcn)


rf = ref()
for dname in dtypes:
    w = ttnn.from_torch(w_t, dtype=DT[dname], layout=ttnn.TILE_LAYOUT, device=md)
    for fid in fids:
        k_cfg = ckc(fid)
        for c in candidates():
            gx, gy, bw, sw, pcn = c
            kw = dict(
                sparsity=sp,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                program_config=cfg(gx, gy, bw, sw, pcn),
                compute_kernel_config=k_cfg,
                dtype=ttnn.bfloat16,
            )
            if mode == "idx":
                kw["indices"] = idx
                if role == "down":
                    kw["is_input_a_sparse"] = True
            else:
                kw["nnz"] = None if mode == "scan" else n_active
                if role == "down":
                    kw["is_input_a_sparse"] = True
                    kw["is_input_b_sparse"] = False
            row = dict(
                role=role,
                mode=mode,
                n_active=n_active,
                dtype=dname,
                fid=fid,
                grid=[gx, gy],
                in0_bw=bw,
                sub_w=sw,
                pcn=pcn,
            )
            try:
                o = ttnn.sparse_matmul(a, w, **kw)
                row["pcc"] = round(pcc(pick(o), rf), 5)
                ttnn.deallocate(o)
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                try:
                    for _ in range(REPS):
                        ttnn.deallocate(ttnn.sparse_matmul(a, w, **kw))
                finally:
                    ttnn.end_trace_capture(md, tid, cq_id=0)
                try:
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(md)
                    us = (time.perf_counter() - t0) / (ITERS * REPS) * 1e6
                finally:
                    ttnn.release_trace(md, tid)
                mb = n_active * kt * nt * TILE_BYTES[dname] / 1e6
                row.update(
                    us=round(us, 2),
                    weight_MB=round(mb, 2),
                    GBps=round(mb * 1e3 / us, 1),
                    dram_pct=round(mb * 1e3 / us / 512 * 100, 1),
                )
            except Exception as e:
                row["error"] = " ".join(l for l in str(e).splitlines()[:3])[-300:]
            print(json.dumps(row), flush=True)
    ttnn.deallocate(w)
ttnn.close_mesh_device(md)
