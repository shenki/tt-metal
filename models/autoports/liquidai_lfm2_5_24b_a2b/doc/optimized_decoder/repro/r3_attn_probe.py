"""r3_attn feasibility probe: sharded create_qkv_heads_decode, matmul per-head RMSNorm, pad-free concat."""
import torch

import ttnn
from models.autoports.lfm2_r3_attn.tt.fused_decoder import _height_shard, decode_head_mem
from models.autoports.lfm2_r3_attn.tt.optimized_decoder import width_shard_layout

torch.manual_seed(0)
B, NH, NKV, HD = 32, 32, 8, 64
N = (NH + 2 * NKV) * HD
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
L1 = ttnn.L1_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def tt(x, mem=L1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev, memory_config=mem)


x = torch.randn(1, 1, B, N) * 3
head_mem = decode_head_mem(dev, B, HD)
xi = tt(x)
q0, k0, v0 = ttnn.experimental.nlp_create_qkv_heads_decode(xi, num_heads=NH, num_kv_heads=NKV, memory_config=head_mem)
xs = ttnn.to_memory_config(xi, width_shard_layout(dev, 8, N // 8))
q1, k1, v1 = ttnn.experimental.nlp_create_qkv_heads_decode(xs, num_heads=NH, num_kv_heads=NKV, memory_config=head_mem)
for a, b, n in ((q0, q1, "q"), (k0, k1, "k"), (v0, v1, "v")):
    ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
    print("TASK1 sharded create_heads", n, tuple(tb.shape), "equal", torch.equal(ta, tb), b.memory_config().shard_spec)

# ---- task 2: per-head RMSNorm as matmuls on qkv rows ----
eps = 1e-5
gq, gk = torch.rand(HD) + 0.5, torch.rand(HD) + 0.5
S = torch.zeros(N, 64)
for h in range(NH + NKV):
    S[h * HD : (h + 1) * HD, h] = 1.0 / HD
E = torch.zeros(64, N)
for h in range(NH + NKV):
    E[h, h * HD : (h + 1) * HD] = gq if h < NH else gk
vmask = torch.zeros(1, N)
vmask[0, (NH + NKV) * HD :] = 1.0
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
xb = x.bfloat16().float()


def ref():
    r = xb.reshape(B, NH + 2 * NKV, HD)
    out = r.clone()
    rr = r[:, : NH + NKV]
    n = rr * torch.rsqrt(rr.pow(2).mean(-1, keepdim=True) + eps)
    n[:, :NH] *= gq
    n[:, NH:] *= gk
    out[:, : NH + NKV] = n
    return out.reshape(1, 1, B, N)


R = ref()
S_t, E_t = tt(S), tt(E)
eps_b, vm_t = tt(torch.full((1, 64), eps)), tt(vmask)
wmem = xs.memory_config()
sq = ttnn.multiply(xs, xs, dtype=ttnn.float32, memory_config=wmem)
ss = ttnn.linear(sq, S_t, bias=eps_b, compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.float32)
r = ttnn.rsqrt(ss, memory_config=L1)
sc = ttnn.linear(r, E_t, bias=vm_t, compute_kernel_config=ckc, memory_config=wmem, dtype=ttnn.float32)
y = ttnn.multiply(xs, sc, memory_config=wmem, dtype=ttnn.bfloat16)
Y = ttnn.to_torch(y)
print("TASK2 mm-norm pcc", pcc(Y, R), "maxabs", (Y - R).abs().max().item(), "mem", y.memory_config().memory_layout)
# reference LN path
q_ln = ttnn.rms_norm(
    ttnn.to_memory_config(q1, L1),
    epsilon=eps,
    weight=tt(gq.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT),
    compute_kernel_config=ckc,
)
Qr = R.reshape(B, NH + 2 * NKV, HD)[:, :NH]
print(
    "TASK2 LN-path q pcc",
    pcc(ttnn.to_torch(q_ln)[0, :, :NH], Qr),
    " mm-path q pcc",
    pcc(Y.reshape(B, -1, HD)[:, :NH], Qr),
)
print("TASK2 mm vs LN q maxabs", (Y.reshape(B, -1, HD)[:, :NH] - ttnn.to_torch(q_ln)[0, :, :NH]).abs().max().item())

# ---- task 4: concat heads with b users into a preallocated zero buffer ----
for b in (1, 4, 16, 32):
    cin = _height_shard(ttnn.CoreGrid(y=max(1, b // 8), x=min(b, 8)), (32, HD))
    a = torch.randn(1, b, 32, HD)
    a[:, :, NH:] = 0
    at = ttnn.to_memory_config(tt(a), cin)
    omem = width_shard_layout(dev, 32, HD)
    z = ttnn.from_torch(
        torch.zeros(1, 1, B, NH * HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=omem
    )
    o = ttnn.experimental.nlp_concat_heads_decode(at, num_heads=NH, output_tensor=z)
    O = ttnn.to_torch(z)
    exp = torch.zeros(1, 1, B, NH * HD)
    exp[0, 0, :b] = a[0, :, :NH].reshape(b, -1).bfloat16().float()
    print("TASK4 concat b", b, "equal", torch.equal(O, exp), tuple(o.shape))
ttnn.close_mesh_device(dev)
