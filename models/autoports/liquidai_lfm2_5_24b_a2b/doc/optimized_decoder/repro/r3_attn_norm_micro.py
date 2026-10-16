"""r3_attn: per-head q/k RMSNorm variants on the decode qkv [32, 3072] (Tracy micro-bench + exactness).
flock /tmp/claude-1000/lfm2_tracy.lock python -m tracy -r -p -v r3_attn_norm_micro.py ; then r3_attn_micro_report.py"""
import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_attn.tt.fused_decoder import decode_head_mem
from models.autoports.lfm2_r3_attn.tt.optimized_decoder import width_shard_layout

torch.manual_seed(0)
B, NH, NKV, HD = 32, 32, 8, 64
N = (NH + 2 * NKV) * HD
G = 64
eps = 1e-5
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def tt(x, mem=L1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, device=dev, memory_config=mem)


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


x = torch.randn(1, 1, B, N) * 3
gq, gk = torch.rand(HD) + 0.5, torch.rand(HD) + 0.5
xb = x.bfloat16().float()
r = xb.reshape(B, NH + 2 * NKV, HD).clone()
nn_ = r[:, : NH + NKV] * torch.rsqrt(r[:, : NH + NKV].pow(2).mean(-1, keepdim=True) + eps)
nn_[:, :NH] *= gq
nn_[:, NH:] *= gk
R = r.clone()
R[:, : NH + NKV] = nn_
R = R.reshape(1, 1, B, N)

wmem = width_shard_layout(dev, 8, N // 8)
xs = ttnn.to_memory_config(tt(x), wmem)
red = torch.zeros(N, G)
exp = torch.zeros(G, N)
for h in range(NH + NKV):
    red[h * HD : (h + 1) * HD, h] = 1.0 / HD
    exp[h, h * HD : (h + 1) * HD] = gq if h < NH else gk
VC = NH + NKV  # constant column: r = rsqrt(0 + 1) = 1 for the v lanes
exp[VC, (NH + NKV) * HD :] = 1.0
bias = torch.full((1, G), eps)
bias[0, VC] = 1.0
red_l1, exp_l1 = tt(red), tt(exp)
red_d, exp_d = tt(red, DRAM), tt(exp, DRAM)
bias_f = tt(bias, dtype=ttnn.float32)
bias_b = tt(bias)
head_mem = decode_head_mem(dev, B, HD)
gq_rm = tt(gq.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
gk_rm = tt(gk.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)


def cfg1d(grid, in0_bw, per_m, per_n, act=None, mcast_in0=True):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_m,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=mcast_in0,
    )


def v_ln():  # current: heads then LN round trip
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(xs, num_heads=NH, num_kv_heads=NKV, memory_config=head_mem)
    q = ttnn.to_memory_config(
        ttnn.rms_norm(
            ttnn.to_memory_config(q, L1), epsilon=eps, weight=gq_rm, memory_config=L1, compute_kernel_config=ckc
        ),
        head_mem,
    )
    k = ttnn.to_memory_config(
        ttnn.rms_norm(
            ttnn.to_memory_config(k, L1), epsilon=eps, weight=gk_rm, memory_config=L1, compute_kernel_config=ckc
        ),
        head_mem,
    )
    return q, k, v


def heads(y):
    return ttnn.experimental.nlp_create_qkv_heads_decode(y, num_heads=NH, num_kv_heads=NKV, memory_config=head_mem)


def v_mm_a():  # mm v1 (as in first impl, vcol add replaced by the const column)
    sq = ttnn.multiply(xs, xs, dtype=ttnn.float32, memory_config=wmem)
    ms = ttnn.linear(sq, red_d, bias=bias_f, compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.float32)
    rr = ttnn.rsqrt(ms, memory_config=L1)
    sc = ttnn.linear(rr, exp_d, compute_kernel_config=ckc, memory_config=wmem, dtype=ttnn.float32)
    return heads(ttnn.multiply(xs, sc, dtype=ttnn.bfloat16, memory_config=wmem))


def v_mm_b():  # bf16 square (unary), 1D config w/ fused bias, separate rsqrt, L1 weights
    sq = ttnn.square(xs, memory_config=wmem)
    ms = ttnn.linear(
        sq,
        red_l1,
        bias=bias_f,
        program_config=cfg1d((8, 1), 12, 1, 1),
        compute_kernel_config=ckc,
        memory_config=L1,
        dtype=ttnn.float32,
    )
    rr = ttnn.rsqrt(ms, memory_config=L1)
    sc = ttnn.linear(
        rr,
        exp_l1,
        program_config=cfg1d((8, 1), 2, 1, 12),
        compute_kernel_config=ckc,
        memory_config=wmem,
        dtype=ttnn.bfloat16,
    )
    return heads(ttnn.multiply(xs, sc, memory_config=wmem))


def v_mm_c():  # as b, fp32 square + fp32 scale
    sq = (
        ttnn.square(xs, memory_config=wmem, dtype=ttnn.float32)
        if False
        else ttnn.typecast(ttnn.square(xs, memory_config=wmem), ttnn.float32)
    )
    ms = ttnn.linear(
        sq,
        red_l1,
        bias=bias_f,
        program_config=cfg1d((8, 1), 12, 1, 1),
        compute_kernel_config=ckc,
        memory_config=L1,
        dtype=ttnn.float32,
    )
    rr = ttnn.rsqrt(ms, memory_config=L1)
    sc = ttnn.linear(
        rr,
        exp_l1,
        program_config=cfg1d((8, 1), 2, 1, 12),
        compute_kernel_config=ckc,
        memory_config=wmem,
        dtype=ttnn.float32,
    )
    return heads(ttnn.multiply(xs, sc, dtype=ttnn.bfloat16, memory_config=wmem))


def v_mm_d():  # default configs, bf16 everywhere, L1 weights
    sq = ttnn.square(xs, memory_config=wmem)
    ms = ttnn.linear(sq, red_l1, bias=bias_b, compute_kernel_config=ckc, memory_config=L1)
    rr = ttnn.rsqrt(ms, memory_config=L1)
    sc = ttnn.linear(rr, exp_l1, compute_kernel_config=ckc, memory_config=wmem)
    return heads(ttnn.multiply(xs, sc, memory_config=wmem))


def v_mm_e():  # reduction as DRAM-sharded-like local: in0 interleaved L1, 1D mcast_in0 over 2 output cores
    sq = ttnn.square(xs, memory_config=L1)
    ms = ttnn.linear(
        sq,
        red_l1,
        bias=bias_f,
        program_config=cfg1d((2, 1), 8, 1, 1),
        compute_kernel_config=ckc,
        memory_config=L1,
        dtype=ttnn.float32,
    )
    rr = ttnn.rsqrt(ms, memory_config=L1)
    sc = ttnn.linear(
        rr,
        exp_l1,
        program_config=cfg1d((8, 1), 2, 1, 12),
        compute_kernel_config=ckc,
        memory_config=wmem,
        dtype=ttnn.bfloat16,
    )
    return heads(ttnn.multiply(xs, sc, memory_config=wmem))


VARIANTS = {"ln": v_ln, "mm_a": v_mm_a, "mm_b": v_mm_b, "mm_c": v_mm_c, "mm_d": v_mm_d, "mm_e": v_mm_e}
qr = R.reshape(B, -1, HD)
for name, f in VARIANTS.items():
    try:
        out = f()
        q, k = ttnn.to_torch(out[0])[0, :, :NH], ttnn.to_torch(out[1])[0, :, :NKV]
        v = ttnn.to_torch(out[2])[0, :, :NKV]
        print(
            f"EXACT {name}: q pcc {pcc(q, qr[:, :NH]):.7f} max {float((q - qr[:, :NH]).abs().max()):.4f}  k pcc {pcc(k, qr[:, NH:NH+NKV]):.7f}  v equal {torch.equal(v, qr[:, NH+NKV:])}"
        )
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
