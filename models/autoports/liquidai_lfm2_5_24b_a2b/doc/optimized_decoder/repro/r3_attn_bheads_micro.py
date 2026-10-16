"""r3_attn: b-user head split (nlp_create_qkv_heads_decode slice_size/batch_offset) and the per-user ops
on b users vs 32 users (Tracy micro-bench; then r3_attn_micro_report.py)."""
import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_attn.tt.fused_decoder import decode_head_mem
from models.autoports.lfm2_r3_attn.tt.optimized_decoder import width_shard_layout

torch.manual_seed(0)
B, NH, NKV, HD = 32, 32, 8, 64
N = (NH + 2 * NKV) * HD
eps = 1e-5
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
tt = lambda x, mem=L1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
    x.contiguous(), dtype=dtype, layout=layout, device=dev, memory_config=mem
)
x = torch.randn(1, 1, B, N)
xs = ttnn.to_memory_config(tt(x), width_shard_layout(dev, 8, N // 8))
g_rm = tt((torch.rand(HD) + 0.5).reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
boff = ttnn.from_torch(
    torch.zeros(1, dtype=torch.int32), dtype=ttnn.uint32, device=dev, memory_config=L1
)  # batch offset 0
cs = {}
for b in (1, 4, 32):
    hm = decode_head_mem(dev, b, HD)
    c = torch.randn(1, b, 1, HD)
    cs[b] = (hm, ttnn.interleaved_to_sharded(tt(c), hm), ttnn.interleaved_to_sharded(tt(c), hm))


def run(b):
    hm, cos, sin = cs[b]
    kw = {} if b == B else dict(batch_offset=boff, slice_size=b)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(xs, num_heads=NH, num_kv_heads=NKV, memory_config=hm, **kw)
    q = ttnn.to_memory_config(
        ttnn.rms_norm(
            ttnn.to_memory_config(q, L1), epsilon=eps, weight=g_rm, memory_config=L1, compute_kernel_config=ckc
        ),
        hm,
    )
    k = ttnn.to_memory_config(
        ttnn.rms_norm(
            ttnn.to_memory_config(k, L1), epsilon=eps, weight=g_rm, memory_config=L1, compute_kernel_config=ckc
        ),
        hm,
    )
    q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True, compute_kernel_config=ckc)
    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True, compute_kernel_config=ckc)
    return q, k, v


ref = [
    ttnn.to_torch(t)
    for t in ttnn.experimental.nlp_create_qkv_heads_decode(xs, num_heads=NH, num_kv_heads=NKV, memory_config=cs[32][0])
]
for b in (1, 4, 32):
    try:
        kw = {} if b == B else dict(batch_offset=boff, slice_size=b)
        o = [
            ttnn.to_torch(t)
            for t in ttnn.experimental.nlp_create_qkv_heads_decode(
                xs, num_heads=NH, num_kv_heads=NKV, memory_config=cs[b][0], **kw
            )
        ]
        print(
            f"EXACT heads b{b}: shapes {[tuple(t.shape) for t in o]} equal {[torch.equal(t, r[:, :b]) for t, r in zip(o, ref)]}"
        )
        for _ in range(2):
            run(b)
        ttnn.synchronize_device(dev)
        signpost(f"V_b{b}")
        run(b)
        ttnn.synchronize_device(dev)
        signpost(f"E_b{b}")
    except Exception as e:
        print(f"FAIL b{b}: {str(e).splitlines()[0][:300]}")
ttnn.close_mesh_device(dev)
