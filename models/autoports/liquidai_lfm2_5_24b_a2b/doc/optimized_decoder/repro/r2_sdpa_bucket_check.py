# Bucketed SDPA decode: Q (32 users) + page_table[:b] + cur_pos[:b] vs the full 32-row call.
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import decode_head_mem

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
B, NH, NKV, HD, BLK, CTX = 32, 32, 8, 64, 64, 2048
nb = (CTX + 64 + BLK - 1) // BLK
nbp = (nb + 7) // 8 * 8
nblk = B * nb + 1
g = torch.Generator().manual_seed(0)
grid = md.compute_with_storage_grid_size()
kt = torch.randn(nblk, NKV, BLK, HD, generator=g)
vt = torch.randn(nblk, NKV, BLK, HD, generator=g)
k = ttnn.from_torch(kt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md)
v = ttnn.from_torch(vt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md)
ptt = torch.zeros(B, nbp, dtype=torch.int32)
ptt[:, :nb] = (torch.randperm(nblk - 1, generator=g)[: B * nb] + 1).reshape(B, nb).int()
RM = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
pt = ttnn.from_torch(ptt, **RM)
curt = torch.randint(100, CTX, (B,), generator=g).int()
curt[20:] = -1
cur = ttnn.from_torch(curt, **RM)
qt = torch.randn(1, B, NH, HD, generator=g).bfloat16()
q_sh = ttnn.from_torch(qt, layout=ttnn.TILE_LAYOUT, device=md, memory_config=decode_head_mem(md, B, HD))
q_il = ttnn.from_torch(qt, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.L1_MEMORY_CONFIG)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def sdpa(q, p, c, kc):
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), q_chunk_size=0, k_chunk_size=kc, exp_approx_mode=False
    )
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k,
        v,
        page_table_tensor=p,
        cur_pos_tensor=c,
        scale=HD**-0.5,
        program_config=cfg,
        compute_kernel_config=ckc,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


ref = ttnn.to_torch(sdpa(q_sh, pt, cur, 512)).float()


def pcc(a, b):
    a, b = a.flatten(), b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def timed(f):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        o = f()
        ttnn.deallocate(o)
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(50):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    return (time.perf_counter() - t0) / 50 * 1e6


print("full B=32 kc512:", f"{timed(lambda: sdpa(q_sh, pt, cur, 512)):.1f} us")
for b in (1, 2, 4, 8, 16):
    for qname, q in (("sharded", q_sh), ("interleaved", q_il)):
        for kc in (128, 256, 512):

            def f(q=q, b=b, kc=kc):
                p = ttnn.slice(pt, [0, 0], [b, nbp])
                c = ttnn.slice(cur, [0], [b])
                o = sdpa(q, p, c, kc)
                ttnn.deallocate(p)
                ttnn.deallocate(c)
                o2 = ttnn.pad(o, [(0, 0), (0, B - b), (0, 0), (0, 0)], 0.0)
                ttnn.deallocate(o)
                return o2

            try:
                out = ttnn.to_torch(f()).float()
                act = min(b, 20)
                p_ = pcc(out[:, :act], ref[:, :act])
                mx = (out[:, :act] - ref[:, :act]).abs().max().item()
                zero = out[:, b:].abs().max().item() if b < B else 0.0
                print(
                    f"b={b:2d} q={qname:11s} kc={kc}: pcc {p_:.6f} maxdiff {mx:.4f} pad-max {zero} shape {tuple(out.shape)} time {timed(f):.1f} us",
                    flush=True,
                )
            except Exception as e:
                print(f"b={b:2d} q={qname:11s} kc={kc}: ERR {str(e).splitlines()[0][:150]}", flush=True)
ttnn.close_mesh_device(md)
