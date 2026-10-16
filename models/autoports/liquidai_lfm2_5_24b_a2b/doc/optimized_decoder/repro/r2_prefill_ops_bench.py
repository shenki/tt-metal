# Review P2 #4: MoE-dispatch glue / SDPA / projection alternatives at the 2048-row prefill chunk.
# Timing = traced replay (trace of one call, executed N times, wall / N) -> device time + tiny launch.
#   TT_VISIBLE_DEVICES=3 python models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/repro/r2_prefill_ops_bench.py [sections]
import math
import sys
import time

import torch

import ttnn

TILE = 32
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
grid = md.compute_with_storage_grid_size()
DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
sections = set(sys.argv[1:]) or {"scan", "scatter", "convert", "sdpa", "mm"}


def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=md)


def host(t):
    return ttnn.to_torch(t).float()


def timed(f, n=20):
    """Traced replay time in us of f() (compile run first). Returns (us, output of compile run)."""
    try:
        out = f()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        o2 = f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(md, tid)
        return us, out
    except Exception as e:
        return "ERR " + str(e).strip().splitlines()[0][:160], None


def ckc(fid, fp32=True, l1acc=False, approx=False):
    return ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=l1acc,
    )


def show(name, r, extra=""):
    us = r if isinstance(r, str) else f"{r:8.1f} us"
    print(f"R2 {name:60s} {us} {extra}", flush=True)


T, K, E, H = 2048, 4, 64, 2048
P = T * K
G = P // TILE
g = torch.Generator().manual_seed(0)
idx_t = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])  # [T, K]
onehot_t = (idx_t.reshape(P, 1) == torch.arange(E)).float()  # [P, E]
cum_ref = onehot_t.cumsum(0)

# ---------------- 1. rank scan ----------------
if "scan" in sections:
    oh = dev(onehot_t.reshape(1, G, TILE, E))
    tri = dev(torch.tril(torch.ones(TILE, TILE)).reshape(1, 1, TILE, TILE).expand(1, G, TILE, TILE).contiguous())
    hifi4 = ckc("HiFi4")
    lofi = ckc("LoFi")
    local_ref = torch.einsum("ij,gje->gie", torch.tril(torch.ones(TILE, TILE)), onehot_t.reshape(G, TILE, E))

    def chk(o):
        return "exact" if o is not None and torch.equal(host(o).reshape(G, TILE, E), local_ref) else "MISMATCH"

    r, o = timed(lambda: ttnn.matmul(tri, oh, compute_kernel_config=hifi4, dtype=ttnn.float32))
    show("scan current batched mm (no cfg) hifi4", r, chk(o))
    for gx, gy in ((grid.x, grid.y), (8, 8)):
        for fid in ("HiFi4", "LoFi"):
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=E // TILE,
            )
            r, o = timed(
                lambda: ttnn.matmul(tri, oh, compute_kernel_config=ckc(fid), dtype=ttnn.float32, program_config=cfg)
            )
            show(f"scan batched mm MultiCoreReuse grid {gx}x{gy} {fid}", r, chk(o))
    # bcast tri: in1 batched, in0 broadcast? (tri as [1,1,32,32])
    tri1 = dev(torch.tril(torch.ones(TILE, TILE)).reshape(1, 1, TILE, TILE))
    r, o = timed(lambda: ttnn.matmul(tri1, oh, compute_kernel_config=hifi4, dtype=ttnn.float32))
    show("scan batched mm tri broadcast [1,1,32,32] (no cfg)", r, chk(o))
    # transposed single matmul: ohT [E, P] -> reshape [E*G, 32] @ U32 -> [E*G, 32]
    ohT = dev(onehot_t.T.contiguous())
    U = dev(torch.triu(torch.ones(TILE, TILE)))

    def transposed():
        a = ttnn.reshape(ohT, (E * G, TILE))
        return ttnn.matmul(a, U, compute_kernel_config=lofi, dtype=ttnn.float32)

    r, o = timed(transposed)
    ok = o is not None and torch.equal(host(o).reshape(E, G, TILE).permute(1, 2, 0), local_ref)
    show("scan transposed: reshape ohT [E*G,32] @ triu32 (single mm, lofi)", r, "exact" if ok else "MISMATCH")
    r, _ = timed(lambda: ttnn.reshape(ohT, (E * G, TILE)))
    show("  (its tile reshape alone)", r)
    ttnn.deallocate(oh)

# ---------------- 2. scatter ----------------
if "scatter" in sections:
    # realistic dest (tile-aligned regions), source token ids t+1
    counts = onehot_t.sum(0)
    tiles = torch.ceil(counts / TILE)
    region = torch.cat([torch.zeros(1), tiles.cumsum(0)[:-1]]) * TILE
    dest_t = (onehot_t * (cum_ref + region - 1)).sum(-1)  # [P]
    maxbuf = ((P + E * 31) + TILE - 1) // TILE * TILE
    src_tok_t = (torch.arange(P) // K + 1).float()
    ref = torch.zeros(maxbuf)
    ref[dest_t.long()] = src_tok_t
    dest_u = dev(dest_t.reshape(1, P), ttnn.uint32, RM)
    src_tok = dev(src_tok_t.reshape(1, P), ttnn.float32, RM)
    zero = dev(torch.zeros(1, maxbuf), ttnn.float32, RM)
    r, o = timed(lambda: ttnn.scatter(zero, -1, dest_u, src_tok))
    show(
        f"scatter current 1 row [1,{maxbuf}] <- {P}",
        r,
        "exact" if o is not None and torch.equal(host(o)[0], ref) else "MISMATCH",
    )
    for R in (8, 32, 64, 128):
        dR = ttnn.reshape(dest_u, (R, P // R))
        sR = ttnn.reshape(src_tok, (R, P // R))
        zR = dev(torch.zeros(R, maxbuf), ttnn.float32, RM)

        def split():
            s = ttnn.scatter(zR, -1, dR, sR)  # each row: its own P/R pairs, zeros elsewhere
            st = ttnn.to_layout(s, ttnn.TILE_LAYOUT)
            ttnn.deallocate(s)
            red = ttnn.sum(st, dim=0, keepdim=True)  # every dest written once: sum == placement (exact fp32)
            ttnn.deallocate(st)
            return ttnn.typecast(ttnn.to_layout(red, RM), ttnn.uint32)

        r, o = timed(split)
        ok = o is not None and torch.equal(host(o)[0, :maxbuf], ref)
        show(f"scatter split rows R={R} + tile sum(dim0) + uint32", r, "exact" if ok else "MISMATCH")
        r, _ = timed(lambda: ttnn.scatter(zR, -1, dR, sR))
        show(f"  (scatter alone R={R})", r)

# ---------------- 3. FFN output BFP8 tile -> BF16 RM ----------------
if "convert" in sections:
    maxbuf = ((P + E * 31) + TILE - 1) // TILE * TILE
    x = torch.randn(maxbuf, H)
    o8 = dev(x, ttnn.bfloat8_b)
    ref = host(ttnn.typecast(o8, ttnn.bfloat16))
    r, o = timed(lambda: ttnn.to_layout(ttnn.typecast(o8, ttnn.bfloat16), RM))
    show("convert current typecast + to_layout", r)
    for name, f in (
        ("untilize(bfp8) direct", lambda: ttnn.untilize(o8)),
        ("to_layout(bfp8, RM) direct", lambda: ttnn.to_layout(o8, RM)),
        ("to_layout(bfp8, RM, dtype=bf16)", lambda: ttnn.to_layout(o8, RM, dtype=ttnn.bfloat16)),
        ("untilize(bfp8, use_multicore, pack)", lambda: ttnn.untilize(o8, use_multicore=True, use_pack_untilize=True)),
    ):
        r, o = timed(f)
        extra = ""
        if o is not None:
            extra = f"dtype={o.dtype} layout={o.layout} exact={torch.equal(host(o), ref)}"
        show("convert " + name, r, extra)
    o16 = dev(x, ttnn.bfloat16)
    r, _ = timed(lambda: ttnn.to_layout(o16, RM))
    show("convert untilize bf16 tile alone (if FFN wrote bf16)", r)

# ---------------- 4. prefill SDPA ----------------
if "sdpa" in sections:
    NH, NKV, HD, BLK = 32, 8, 64, 64
    for start, S in ((0, 2048), (2048, 2048), (0, 128), (0, 32), (30720, 2048)):
        kv_len = start + S
        nblk = kv_len // BLK + 1 if kv_len % BLK else kv_len // BLK
        nbp = max(nblk, 1)
        kt = torch.randn(nbp + 1, NKV, BLK, HD, generator=g)
        vt = torch.randn(nbp + 1, NKV, BLK, HD, generator=g)
        perm = (torch.randperm(nbp, generator=g) + 1).int()
        pt = dev(perm.reshape(1, -1), ttnn.int32, RM)
        q_t = torch.randn(1, NH, S, HD, generator=g)
        q = dev(q_t)
        cs = dev(torch.tensor([start], dtype=torch.int32), ttnn.int32, RM)
        kc8 = dev(kt, ttnn.bfloat8_b)
        vc8 = dev(vt, ttnn.bfloat8_b)
        # torch reference
        kf = kt[perm.long()].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :kv_len]
        vf = vt[perm.long()].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :kv_len]
        kq = host(kc8)[perm.long()].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :kv_len]
        vq = host(vc8)[perm.long()].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :kv_len]
        kq = kq.repeat_interleave(NH // NKV, 0)
        vq = vq.repeat_interleave(NH // NKV, 0)
        s = (q_t[0].bfloat16().float() @ kq.transpose(-1, -2)) * HD**-0.5
        mask = torch.arange(kv_len)[None, :] > (torch.arange(S)[:, None] + start)
        s.masked_fill_(mask, float("-inf"))
        ref = torch.softmax(s, -1) @ vq
        for qc, kc in (
            (128, 128),
            (256, 256),
            (128, 256),
            (128, 512),
            (256, 512),
            (64, 256),
            (64, 512),
            (128, 1024),
            (256, 128),
        ):
            qc_ = min(qc, S)
            if S % qc_ or (start % qc_) or (start % kc and start):
                continue
            for fid in ("HiFi4", "HiFi2"):
                cfg = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=qc_, k_chunk_size=kc, exp_approx_mode=False
                )
                ck = ckc(fid)
                r, o = timed(
                    lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                        q,
                        kc8,
                        vc8,
                        pt,
                        chunk_start_idx_tensor=cs,
                        scale=HD**-0.5,
                        program_config=cfg,
                        compute_kernel_config=ck,
                    ),
                    n=10,
                )
                extra = ""
                if o is not None:
                    a = host(o)[0].flatten()
                    b = ref.flatten()
                    extra = f"pcc={torch.corrcoef(torch.stack([a, b]))[0, 1].item():.6f} max|err|={(a - b).abs().max().item():.4f}"
                show(f"sdpa start={start} S={S} q{qc_} k{kc} {fid}", r, extra)
        for t_ in (q, kc8, vc8, pt, cs):
            ttnn.deallocate(t_)

# ---------------- 5. prefill projections ----------------
if "mm" in sections:

    def cfg2d(m, k, n, gx, gy, bw, sh, sw, obh=None, obw=None):
        mt, nt = m // TILE, n // TILE
        pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            out_block_h=obh or pm,
            out_block_w=obw or pn,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    import importlib

    od = importlib.import_module("models.autoports.lfm2_s3snap.tt.optimized_decoder")
    hifi2 = ckc("HiFi2", True, True)
    for name, m, k, n in (("qkv", 2048, 2048, 3072), ("wo/conv_out", 2048, 2048, 2048), ("conv_in", 2048, 2048, 6144)):
        a = dev(torch.randn(1, 1, m, k))
        w = dev(torch.randn(k, n) * 0.02, ttnn.bfloat8_b)
        ref = None
        cur = od.prefill_2d_config(md, m, k, n)
        r, o = timed(lambda: ttnn.linear(a, w, program_config=cur, compute_kernel_config=hifi2, memory_config=DRAM))
        ref = host(o) if o is not None else None
        show(
            f"mm {name} {m}x{k}x{n} current prefill_2d_config (bw {cur.in0_block_w}, sub {cur.out_subblock_h}x{cur.out_subblock_w}, blk {cur.out_block_h}x{cur.out_block_w}, pc {cur.per_core_M}x{cur.per_core_N})",
            r,
        )
        r, o = timed(lambda: ttnn.linear(a, w, compute_kernel_config=hifi2, memory_config=DRAM))
        show(f"mm {name} auto config", r)
        mt, nt = m // TILE, n // TILE
        cands = []
        for gx in (grid.x, 8, 12 if grid.x >= 12 else None):
            if gx is None:
                continue
            for gy in (grid.y, 8):
                pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
                for bw in (2, 4, 8):
                    for sh, sw in ((1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (2, 1), (4, 1)):
                        if pm % sh or pn % sw:
                            continue
                        # L1: double-buffered in0/in1 blocks + out (bf16) + fp32 partials
                        used = 2 * pm * bw * 2048 + 2 * bw * pn * 1088 + pm * pn * (2048 + 4096)
                        if used > 1_300_000:
                            continue
                        cands.append((gx, gy, bw, sh, sw))
        best = None
        for c in cands:
            cfg = cfg2d(m, k, n, *c)
            r, o = timed(
                lambda: ttnn.linear(a, w, program_config=cfg, compute_kernel_config=hifi2, memory_config=DRAM), n=10
            )
            if isinstance(r, str):
                show(f"mm {name} grid {c[0]}x{c[1]} bw {c[2]} sub {c[3]}x{c[4]}", r)
                continue
            ok = ref is not None and torch.allclose(host(o), ref, atol=1e-1)
            show(f"mm {name} grid {c[0]}x{c[1]} bw {c[2]} sub {c[3]}x{c[4]}", r, "" if ok else "DIFF")
            if best is None or r < best[0]:
                best = (r, c)
        show(f"mm {name} BEST {best[1]}", best[0])
        ttnn.deallocate(a)
        ttnn.deallocate(w)

ttnn.close_mesh_device(md)
