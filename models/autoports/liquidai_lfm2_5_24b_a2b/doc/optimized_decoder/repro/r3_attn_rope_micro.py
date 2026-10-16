"""r3_attn: decode cos/sin construction variants (Tracy micro-bench; then r3_attn_micro_report.py)."""
import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_attn.tt.fused_decoder import decode_head_mem

torch.manual_seed(0)
B, HD, S = 32, 64, 4096
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
cos = torch.randn(S, HD)
cos_rm = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
pos = torch.randint(0, S, (B,), dtype=torch.int32)
pos_1b = ttnn.from_torch(pos.reshape(1, B), dtype=ttnn.uint32, device=dev, memory_config=DRAM)
pos_b1 = ttnn.from_torch(pos.reshape(B, 1), dtype=ttnn.uint32, device=dev, memory_config=DRAM)
hm = decode_head_mem(dev, B, HD)
ref = cos[pos.long()].reshape(1, B, 1, HD).bfloat16().float()


def v_base():
    t = ttnn.embedding(pos_1b, cos_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t = ttnn.transpose(ttnn.unsqueeze_to_4D(t), 1, 2)
    return ttnn.interleaved_to_sharded(t, hm)


def v_b1():  # [B, 1] indices: one padded tile per user, no transpose
    t = ttnn.embedding(pos_b1, cos_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [B, 1, HD]
    t = ttnn.reshape(t, (1, B, 1, HD))
    return ttnn.interleaved_to_sharded(t, hm)


def v_b1_l1():
    t = ttnn.embedding(pos_b1, cos_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=L1)
    t = ttnn.reshape(t, (1, B, 1, HD))
    return ttnn.interleaved_to_sharded(t, hm)


def v_rm():  # row-major embedding -> view -> tilize
    t = ttnn.embedding(pos_b1, cos_rm, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=L1)
    t = ttnn.to_layout(ttnn.reshape(t, (1, B, 1, HD)), ttnn.TILE_LAYOUT)
    return ttnn.interleaved_to_sharded(t, hm)


def v_reshape_pos():  # caller keeps [1, B]: device reshape of the indices first
    p = ttnn.reshape(pos_1b, (B, 1))
    t = ttnn.embedding(p, cos_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    t = ttnn.reshape(t, (1, B, 1, HD))
    return ttnn.interleaved_to_sharded(t, hm)


for name, f in {"base": v_base, "b1": v_b1, "b1_l1": v_b1_l1, "rm": v_rm, "reshape_pos": v_reshape_pos}.items():
    try:
        o = ttnn.to_torch(f())
        print(f"EXACT {name}: shape {tuple(o.shape)} equal {torch.equal(o[..., :1, :], ref)}")
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
