"""Minimal repro: paged_scaled_dot_product_attention_decode with HEIGHT-sharded Q on a program-cache hit."""
import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
B, NH, NKV, HD, BLK = 32, 32, 8, 64, 64
nblk = 32 * 32 + 1
head_mem = ttnn.create_sharded_memory_config(
    shape=(32, HD),
    core_grid=ttnn.CoreGrid(y=4, x=8),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
L1 = ttnn.L1_MEMORY_CONFIG
g = torch.Generator().manual_seed(0)
kc = ttnn.from_torch(
    torch.randn(nblk, NKV, BLK, HD, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md
)
vc = ttnn.from_torch(
    torch.randn(nblk, NKV, BLK, HD, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md
)
pt = ttnn.from_torch(
    (torch.randperm(nblk - 1, generator=g)[: B * 32] + 1).reshape(B, 32).int(),
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=md,
)
cur = ttnn.from_torch(
    torch.randint(0, 2000, (B,), generator=g).int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=md
)
q_t = torch.randn(1, B, NH, HD, generator=g).bfloat16()
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
grid = md.compute_with_storage_grid_size()
cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(grid.x, grid.y), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
)


def run(q_mem, out_mem):
    q = ttnn.from_torch(q_t, layout=ttnn.TILE_LAYOUT, device=md, memory_config=q_mem)
    o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        kc,
        vc,
        page_table_tensor=pt,
        cur_pos_tensor=cur,
        scale=HD**-0.5,
        program_config=cfg,
        compute_kernel_config=ckc,
        memory_config=out_mem,
    )
    return ttnn.to_torch(o).float()


# torch reference for the sharded case
kt, vt, ptt, curt = (
    ttnn.to_torch(kc).float(),
    ttnn.to_torch(vc).float(),
    ttnn.to_torch(pt).long(),
    ttnn.to_torch(cur).long(),
)


def ref():
    out = torch.zeros(1, B, NH, HD)
    for b in range(B):
        L = int(curt[b]) + 1
        blocks = ptt[b, : (L + BLK - 1) // BLK]
        K = kt[blocks].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :L]
        V = vt[blocks].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :L]
        q = q_t.float()[0, b].reshape(NKV, NH // NKV, HD)
        a = torch.softmax(q @ K.transpose(-1, -2) * HD**-0.5, -1) @ V
        out[0, b] = a.reshape(NH, HD)
    return out


R = ref()
err = lambda o: float((o[:, :, :NH] - R).abs().max())
full = (grid.x, grid.y)
rm_cores = ttnn.num_cores_to_corerangeset(B, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
rm_mem = ttnn.create_sharded_memory_config(
    shape=(32, HD),
    core_grid=rm_cores,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
for name, q_mem, g2 in [
    ("DRAM q, full grid", ttnn.DRAM_MEMORY_CONFIG, full),
    ("8x4 shard, full grid (current)", head_mem, full),
    ("8x4 shard, 8x8 grid", head_mem, (8, 8)),
    ("row-major shard over grid.x, full grid", rm_mem, full),
]:
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=g2, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
    )
    md.clear_program_cache()
    fresh = run(q_mem, L1)
    pad = ttnn.from_torch(
        torch.full((1, 1, 32, 32 * 13), 7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=L1
    )
    cached = run(q_mem, L1)
    ttnn.deallocate(pad)
    print(
        f"FIX grid={full} {name:40s}: max|err| fresh {err(fresh):.4f} cache-hit {err(cached):.4f} fresh==cached {torch.equal(fresh, cached)}"
    )
ttnn.close_mesh_device(md)
