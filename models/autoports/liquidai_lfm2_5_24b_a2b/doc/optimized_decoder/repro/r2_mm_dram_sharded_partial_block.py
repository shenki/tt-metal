# Minimal repro (round 2): 2D-mcast matmul with a DRAM-WIDTH-SHARDED weight and out_block_w < per_core_N is
# nondeterministic run to run; the same config with an interleaved weight, or full-width blocks, is deterministic.
#   TT_VISIBLE_DEVICES=0 python doc/optimized_decoder/repro/r2_mm_dram_sharded_partial_block.py
# Which 2D-mcast config parameters make the [2048 x K] @ [K x N] prefill matmul nondeterministic?
import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DramShardedLinear, _ckc

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
ckc = _ckc(md, "hifi2")
g = torch.Generator().manual_seed(0)


def run(k, n, wmode, dtype, gx, gy, pcm, pcn, bw, bh, bwid, sh=1, sw=1, reps=6):
    w = torch.randn(k, n, generator=g) * 0.02
    if wmode == "ds":
        lin = DramShardedLinear(md, w, dtype, (8, 1, 1) if k % 256 == 0 else (k // 32, 1, 1))
        wt = lin.weight
    else:
        wt = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=md)
    x = ttnn.from_torch(
        torch.randn(1, 1, 2048, k, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md
    )
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=bh,
        out_block_w=bwid,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    try:
        outs = []
        for _ in range(reps):
            o = ttnn.linear(x, wt, program_config=cfg, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            outs.append(ttnn.to_torch(o))
            ttnn.deallocate(o)
        r = all(torch.equal(outs[0], o) for o in outs[1:])
    except Exception as e:
        r = "ERR " + str(e).splitlines()[0][:80]
    ttnn.deallocate(x)
    ttnn.deallocate(wt)
    return r


L = [
    # k, n, wmode, dtype, gx, gy, pcm, pcn, bw, bh, bwid
    (2048, 12288, "ds", ttnn.bfloat8_b, 11, 10, 7, 35, 8, 7, 7),
    (2048, 12288, "ds", ttnn.bfloat8_b, 11, 10, 7, 35, 4, 7, 7),
    (2048, 12288, "ds", ttnn.bfloat8_b, 11, 10, 7, 35, 8, 1, 35),
    (2048, 12288, "ds", ttnn.bfloat8_b, 11, 10, 7, 35, 8, 7, 5),
    (2048, 12288, "ds", ttnn.bfloat8_b, 11, 10, 7, 35, 8, 1, 7),
    (2048, 12288, "il", ttnn.bfloat8_b, 11, 10, 7, 35, 8, 7, 7),
    (2048, 11264, "il", ttnn.bfloat8_b, 11, 10, 7, 32, 8, 7, 8),  # no N overhang (352 = 11*32)
    (2048, 11264, "il", ttnn.bfloat8_b, 11, 10, 7, 32, 8, 7, 32),
    (2048, 12288, "il", ttnn.bfloat8_b, 8, 8, 8, 48, 8, 8, 8),  # no overhang at all (64 = 8*8, 384 = 8*48)
    (2048, 12288, "il", ttnn.bfloat8_b, 8, 8, 8, 48, 8, 8, 48),
    (2048, 3072, "il", ttnn.bfloat8_b, 11, 10, 7, 9, 8, 7, 3),  # qkv-like with partial width
]
for c in L:
    print(
        "DET2",
        c[:2],
        c[2],
        "bfp8",
        "grid",
        c[4:6],
        "per_core",
        c[6:8],
        "in0_bw",
        c[8],
        "block",
        c[9:11],
        "->",
        run(*c),
        flush=True,
    )
ttnn.close_mesh_device(md)
