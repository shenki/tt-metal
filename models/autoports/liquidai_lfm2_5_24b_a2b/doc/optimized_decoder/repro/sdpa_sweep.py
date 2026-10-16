import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import decode_head_mem

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
B, NH, NKV, HD, BLK = 32, 32, 8, 64, 64
ctx = 2048 + 64
nb_seq = (ctx + BLK - 1) // BLK
nb_seq_p = (nb_seq + 7) // 8 * 8
nblk = B * nb_seq + 1
g = torch.Generator().manual_seed(0)
kt = torch.randn(nblk, NKV, BLK, HD, generator=g)
vt = torch.randn(nblk, NKV, BLK, HD, generator=g)
pt_t = torch.zeros(B, nb_seq_p, dtype=torch.int32)
pt_t[:, :nb_seq] = (torch.randperm(nblk - 1, generator=g)[: B * nb_seq] + 1).reshape(B, nb_seq).int()
pt = ttnn.from_torch(pt_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
grid = md.compute_with_storage_grid_size()
head_mem = decode_head_mem(md, B, HD)
q_t = torch.randn(1, B, NH, HD, generator=g).bfloat16()
q = ttnn.from_torch(q_t, layout=ttnn.TILE_LAYOUT, device=md, memory_config=head_mem)


def bench(kdt, active, kc, grid_sz, fid, approx=False, mmfid=None, ksplit=1):
    kc_t = ttnn.from_torch(kt, dtype=kdt, layout=ttnn.TILE_LAYOUT, device=md)
    vc_t = ttnn.from_torch(vt, dtype=kdt, layout=ttnn.TILE_LAYOUT, device=md)
    cur_t = torch.full((B,), -1, dtype=torch.int32)
    cur_t[:active] = torch.randint(2048 - 32, 2048 + 32, (active,), generator=g).int()
    cur = ttnn.from_torch(cur_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    kw = {} if mmfid is None else {"matmul_math_fidelity": mmfid}
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_sz,
        q_chunk_size=0,
        k_chunk_size=kc,
        exp_approx_mode=approx,
        max_k_splits=ksplit,
        **kw,
    )
    f = lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        kc_t,
        vc_t,
        page_table_tensor=pt,
        cur_pos_tensor=cur,
        scale=HD**-0.5,
        program_config=cfg,
        compute_kernel_config=ckc,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        o = f()
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(30):
            o = f()
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / 30 * 1e6
    except Exception as e:
        return "ERR " + str(e).splitlines()[0][:90]
    finally:
        ttnn.deallocate(kc_t)
        ttnn.deallocate(vc_t)


L = ttnn.MathFidelity
cases = [
    ("bfp8 kc512 hifi2", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2)),
    ("bfp8 kc512 hifi2 approx", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2, approx=True)),
    ("bfp8 kc512 hifi2 mm-lofi", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2, mmfid=L.LoFi)),
    (
        "bfp8 kc512 hifi2 approx mm-lofi",
        dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2, approx=True, mmfid=L.LoFi),
    ),
    ("bfp8 kc1024 hifi2", dict(kdt=ttnn.bfloat8_b, kc=1024, grid_sz=(grid.x, grid.y), fid=L.HiFi2)),
    ("bfp8 kc512 lofi", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.LoFi)),
    ("bfp8 kc512 hifi2 ksplit2", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2, ksplit=2)),
    ("bfp8 kc512 hifi2 ksplit4", dict(kdt=ttnn.bfloat8_b, kc=512, grid_sz=(grid.x, grid.y), fid=L.HiFi2, ksplit=4)),
]
for name, kw in cases:
    for active in (1, 32):
        r = bench(active=active, **kw)
        print(f"SDPA {name:34s} active={active:2d}: {r if isinstance(r,str) else f'{r:8.1f} us'}")
ttnn.close_mesh_device(md)
