"""r3_attn: zero-filled concat-heads output buffer and DRAM-sharded wo on the concat grid (Tracy micro)."""
import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_attn.tt.optimized_decoder import DramShardedLinear, _ckc, width_shard_layout

torch.manual_seed(0)
B, NH, HD, H = 32, 32, 64, 2048
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ckc = _ckc(dev, "hifi2")
cmem = width_shard_layout(dev, 32, HD)
zd = ttnn.from_torch(
    torch.zeros(1, 1, B, NH * HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM
)
w = torch.randn(H, H) * 0.02
x = torch.randn(1, 1, B, H)
xs = ttnn.to_memory_config(
    ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1), cmem
)
lin = {
    cfg: DramShardedLinear(dev, w, ttnn.bfloat8_b, cfg)
    for cfg in [(8, 8, 2), (32, 2, 1), (32, 2, 2), (32, 1, 2), (16, 4, 2)]
}
ref = x.bfloat16().float().reshape(B, H) @ w


def v_zero_i2s():
    return ttnn.to_memory_config(zd, cmem)


def v_zero_dev():
    return ttnn.zeros((1, 1, B, NH * HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=cmem)


def mk_wo(cfg):
    def f():
        return lin[cfg].decode(xs, ckc, DRAM)

    return f


V = {"zero_i2s": v_zero_i2s, "zero_dev": v_zero_dev}
V.update({f"wo_{c[0]}_{c[1]}_{c[2]}": mk_wo(c) for c in lin})
for name, f in V.items():
    try:
        o = ttnn.to_torch(f())
        if name.startswith("wo"):
            print(
                f"EXACT {name}: pcc {torch.corrcoef(torch.stack([o.reshape(-1).double(), ref.reshape(-1).double()]))[0,1].item():.6f}"
            )
        else:
            print(f"EXACT {name}: zero {bool((o == 0).all())}")
        for _ in range(2):
            f()
        ttnn.synchronize_device(dev)
        signpost(f"V_{name}")
        f()
        ttnn.synchronize_device(dev)
        signpost(f"E_{name}")
    except Exception as e:
        print(f"FAIL {name}: {str(e).splitlines()[0][:300]}")
ttnn.close_mesh_device(dev)
