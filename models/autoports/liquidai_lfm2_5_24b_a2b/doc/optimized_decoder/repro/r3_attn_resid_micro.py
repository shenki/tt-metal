"""r3_attn: residual add on the width-sharded out-proj output vs S2I + DRAM add (Tracy micro)."""
import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_attn.tt.optimized_decoder import width_shard_layout

B, H = 32, 2048
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
wm = width_shard_layout(dev, 8, H // 8)
a, r = torch.randn(1, 1, B, H), torch.randn(1, 1, B, H)
mk = lambda t, m: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=m)
o_s, x_d = mk(a, wm), mk(r, DRAM)
x_s = mk(r, wm)


def v_base():  # today: S2I to DRAM then DRAM add, then the ffn norm's I2S
    o = ttnn.to_memory_config(o_s, DRAM)
    h = ttnn.add(x_d, o, memory_config=DRAM)
    return ttnn.to_memory_config(h, wm)


def v_sharded():  # add on the shard grid (+ h interleaved for the final residual)
    h = ttnn.add(x_s, o_s, memory_config=wm)
    hi = ttnn.to_memory_config(h, DRAM)
    return h


def v_sharded_only():
    return ttnn.add(x_s, o_s, memory_config=wm)


for name, f in {"base": v_base, "sharded": v_sharded, "sharded_only": v_sharded_only}.items():
    o = ttnn.to_torch(f())
    print(f"EXACT {name}: {torch.equal(o, (a.bfloat16() + r.bfloat16()).float())}")
    for _ in range(2):
        f()
    ttnn.synchronize_device(dev)
    signpost(f"V_{name}")
    f()
    ttnn.synchronize_device(dev)
    signpost(f"E_{name}")
ttnn.close_mesh_device(dev)
