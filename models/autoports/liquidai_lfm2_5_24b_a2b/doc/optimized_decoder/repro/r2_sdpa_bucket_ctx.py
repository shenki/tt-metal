# Bucketed SDPA decode timing vs context length (pages shared between users; timing only).
import sys
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import decode_head_mem

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
B, NH, NKV, HD, BLK = 32, 32, 8, 64, 64
grid = md.compute_with_storage_grid_size()
g = torch.Generator().manual_seed(0)
RM = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
q = ttnn.from_torch(
    torch.randn(1, B, NH, HD, generator=g).bfloat16(),
    layout=ttnn.TILE_LAYOUT,
    device=md,
    memory_config=decode_head_mem(md, B, HD),
)


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
    for _ in range(20):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    return (time.perf_counter() - t0) / 20 * 1e6


for CTX in [int(c) for c in sys.argv[1].split(",")]:
    nb = (CTX + BLK - 1) // BLK + 1
    nbp = (nb + 7) // 8 * 8
    k = ttnn.from_torch(
        torch.randn(nb + 1, NKV, BLK, HD, generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md
    )
    v = ttnn.from_torch(
        torch.randn(nb + 1, NKV, BLK, HD, generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md
    )
    ptt = torch.zeros(B, nbp, dtype=torch.int32)
    ptt[:, :nb] = torch.stack([torch.randperm(nb, generator=g) + 1 for _ in range(B)]).int()
    pt = ttnn.from_torch(ptt, **RM)
    cur = ttnn.from_torch(torch.full((B,), CTX - 1, dtype=torch.int32), **RM)
    for b in [int(x) for x in sys.argv[2].split(",")]:
        for kc in (128, 256, 512, 1024):

            def f(b=b, kc=kc):
                cfg = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    q_chunk_size=0,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                )
                p = ttnn.slice(pt, [0, 0], [b, nbp]) if b < B else pt
                c = ttnn.slice(cur, [0], [b]) if b < B else cur
                o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
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
                if b < B:
                    ttnn.deallocate(p)
                    ttnn.deallocate(c)
                    o2 = ttnn.pad(o, [(0, 0), (0, B - b), (0, 0), (0, 0)], 0.0)
                    ttnn.deallocate(o)
                    o = o2
                return o

            try:
                print(f"ctx={CTX:6d} b={b:2d} kc={kc:4d}: {timed(f):8.1f} us", flush=True)
            except Exception as e:
                print(f"ctx={CTX:6d} b={b:2d} kc={kc:4d}: ERR {str(e).splitlines()[0][:120]}", flush=True)
    for t in (k, v, pt, cur):
        ttnn.deallocate(t)
ttnn.close_mesh_device(md)
