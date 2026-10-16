# Decode routing primitives (T=32, E=64, K=4): traced-replay time per candidate op + exactness.
import torch

import ttnn

L1 = ttnn.L1_MEMORY_CONFIG
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
T, E, K = 32, 64, 4
torch.manual_seed(0)


def dev(t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dt, layout=lay, device=md, memory_config=L1)


s = torch.rand(1, 1, T, E)
bias = torch.randn(1, 1, 1, E) * 0.05
scores, bias_r, bias_f = dev(s), dev(bias), dev(bias.expand(1, 1, T, E).contiguous())
biased = ttnn.add(scores, bias_r, memory_config=L1)
vals, idx = ttnn.topk(biased, k=K, dim=-1, memory_config=L1)
kth = ttnn.min(vals, dim=-1, keepdim=True, memory_config=L1)
kth_s = ttnn.slice(vals, (0, 0, 0, K - 1), (1, 1, T, K), memory_config=L1)
mask_ref = ttnn.ge(biased, kth, memory_config=L1)
sel = ttnn.multiply(scores, mask_ref, memory_config=L1)
ssum = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1)
den = ttnn.add(ssum, 1e-6, memory_config=L1)
idx_f = ttnn.typecast(idx, ttnn.bfloat16, memory_config=L1)
ex = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
R = torch.zeros(K, K * E)
S = torch.zeros(K * E, E)
C = torch.zeros(T, K * E)
for k in range(K):
    R[k, k * E : (k + 1) * E] = 1
    S[k * E : (k + 1) * E] = torch.eye(E)
    C[:, k * E : (k + 1) * E] = torch.arange(E).float()
R_t, S_t, C_t = dev(R, ttnn.bfloat16), dev(S, ttnn.bfloat16), dev(C, ttnn.bfloat16)
ones_ee = dev(torch.ones(E, E), ttnn.bfloat16)
ones_ee32 = dev(torch.ones(E, E))
active = dev(torch.ones(1, 1, T, 1), ttnn.bfloat16)
onesrow = dev(torch.ones(1, E), ttnn.bfloat16)
dn = dev(torch.randn(1, 4, T, 2048), ttnn.bfloat16)
wt = dev(torch.rand(1, 4, T, 1), ttnn.bfloat16)
wt_full = dev(torch.rand(1, 4, T, 2048), ttnn.bfloat16)
x_dn = dev(torch.randn(1, 1, T, 64), ttnn.bfloat16)


REPS = 5

st = lambda t: ttnn.to_torch(t).float()
sel_h, den_h = st(sel), st(den)[..., :1]
rec = ttnn.reciprocal(den, memory_config=L1)
a = st(ttnn.div(sel, den, memory_config=L1))
b = st(ttnn.bcast(sel, rec, math_op=ttnn.BcastOpMath.MUL, dim=ttnn.BcastOpDim.W, memory_config=L1))
c = st(ttnn.multiply(sel, rec, memory_config=L1))
print("CHECK div vs torch fp32: max", (a - sel_h / den_h).abs().max().item(), "equal", torch.equal(a, sel_h / den_h))
print("CHECK mul(recip) bcast BinaryNg == div:", torch.equal(a, c), (a - c).abs().max().item())
print("CHECK legacy bcast mul(recip) == BinaryNg mul(recip):", torch.equal(b, c), (b - c).abs().max().item())
ab, bb, cb = (
    ttnn.to_torch(
        ttnn.typecast(ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=md), ttnn.bfloat16)
    ).float()
    for v in (a, b, c)
)
print(
    "CHECK bf16: legacy-bcast vs div equal frac",
    (ab == bb).float().mean().item(),
    "binaryng-mul vs div",
    (ab == cb).float().mean().item(),
)
z = dev(torch.zeros(1, 1, T, E))
d1 = st(ttnn.bcast(z, den, math_op=ttnn.BcastOpMath.ADD, dim=ttnn.BcastOpDim.W, memory_config=L1))
print("CHECK legacy bcast ADD zeros+den exact:", torch.equal(d1, den_h.expand_as(d1)), (d1 - den_h).abs().max().item())
ttnn.close_mesh_device(md)
