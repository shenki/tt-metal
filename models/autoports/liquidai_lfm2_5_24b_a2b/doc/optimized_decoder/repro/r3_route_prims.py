# Decode routing primitives (T=32, E=64, K=4): traced-replay time per candidate op + exactness.
import time

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


def timed(f, n=200):
    o = f()
    ttnn.synchronize_device(md)
    outs = o if isinstance(o, (list, tuple)) else [o]
    for t in outs:
        ttnn.deallocate(t)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        o = f()
        outs = o if isinstance(o, (list, tuple)) else [o]
        for t in outs:
            ttnn.deallocate(t)
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    return (time.perf_counter() - t0) / n * 1e6


def run(name, f):
    try:
        print(f"PRIM {name:55s} {timed(f):7.2f} us", flush=True)
    except Exception as e:
        print(f"PRIM {name:55s} ERR {str(e).splitlines()[0][:200]}", flush=True)


run("noop(typecast tiny)", lambda: ttnn.typecast(idx, ttnn.bfloat16, memory_config=L1))
run("add bias row-bcast", lambda: ttnn.add(scores, bias_r, memory_config=L1))
run("add bias full", lambda: ttnn.add(scores, bias_f, memory_config=L1))
run("topk fp32", lambda: list(ttnn.topk(biased, k=K, dim=-1, memory_config=L1)))
run("topk fp32 sorted=False", lambda: list(ttnn.topk(biased, k=K, dim=-1, sorted=False, memory_config=L1)))
run(
    "topk bf16 (ref only)", lambda: list(ttnn.topk(ttnn.typecast(biased, ttnn.bfloat16), k=K, dim=-1, memory_config=L1))
)
run("min kth", lambda: ttnn.min(vals, dim=-1, keepdim=True, memory_config=L1))
run("slice kth", lambda: ttnn.slice(vals, (0, 0, 0, K - 1), (1, 1, T, K), memory_config=L1))
run("ge bcast kth(min)", lambda: ttnn.ge(biased, kth, memory_config=L1))
run("ge bcast kth(slice)", lambda: ttnn.ge(biased, kth_s, memory_config=L1))
run("repeat kth -> [T,E]", lambda: ttnn.repeat(kth_s, ttnn.Shape((1, 1, 1, E))))
run("ge full", lambda: ttnn.ge(biased, scores, memory_config=L1))
run("idx route: mm R", lambda: ttnn.matmul(idx_f, R_t, compute_kernel_config=ex, memory_config=L1))
X = ttnn.matmul(idx_f, R_t, compute_kernel_config=ex, memory_config=L1)
run("idx route: eq C full", lambda: ttnn.eq(X, C_t, memory_config=L1))
OH = ttnn.eq(X, C_t, memory_config=L1)
run("idx route: mm S", lambda: ttnn.matmul(OH, S_t, compute_kernel_config=ex, memory_config=L1))
mask_new = ttnn.matmul(OH, S_t, compute_kernel_config=ex, memory_config=L1)
run("multiply fp32 x bf16 mask", lambda: ttnn.multiply(scores, mask_new, memory_config=L1))
run("multiply fp32 x fp32 mask", lambda: ttnn.multiply(scores, mask_ref, memory_config=L1))
run("where(mask_bf16, scores, 0)", lambda: ttnn.where(mask_new, scores, 0.0, memory_config=L1))
run("sum fp32", lambda: ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1))
run("add eps [T,1]", lambda: ttnn.add(ssum, 1e-6, memory_config=L1))
run("add eps full", lambda: ttnn.add(sel, 1e-6, memory_config=L1))
run("div bcast", lambda: ttnn.div(sel, den, memory_config=L1))
run("div full", lambda: ttnn.div(sel, scores, memory_config=L1))
run("mm sel@ones fp32 (full denom)", lambda: ttnn.matmul(sel, ones_ee32, compute_kernel_config=ex, memory_config=L1))
run("mm active@onesrow", lambda: ttnn.matmul(active, onesrow, compute_kernel_config=ex, memory_config=L1))
run("typecast fp32->bf16", lambda: ttnn.typecast(sel, ttnn.bfloat16, memory_config=L1))
run("div full out bf16", lambda: ttnn.div(sel, scores, dtype=ttnn.bfloat16, memory_config=L1))
run(
    "untilize idx uint16 rows 0:1",
    lambda: ttnn.untilize_with_unpadding(idx, output_tensor_end=(0, 0, 0, K - 1), memory_config=L1),
)
run(
    "untilize idx uint16 rows 0:4",
    lambda: ttnn.untilize_with_unpadding(idx, output_tensor_end=(0, 0, 3, K - 1), memory_config=L1),
)
run("down*wt bcast (BinaryNg)", lambda: ttnn.multiply(dn, wt, memory_config=L1))
run("down*wt full", lambda: ttnn.multiply(dn, wt_full, memory_config=L1))
run(
    "down*wt legacy bcast W",
    lambda: ttnn.bcast(dn, wt, math_op=ttnn.BcastOpMath.MUL, dim=ttnn.BcastOpDim.W, memory_config=L1),
)
run("where bcast [T,1]", lambda: ttnn.where(active, sel, 0.0, memory_config=L1))

# exactness checks
st = lambda t: ttnn.to_torch(t).float()
m_ref, m_new = st(mask_ref), st(mask_new)
print("CHECK idx-route mask == ge mask:", torch.equal(m_ref, m_new))
sel_h = st(sel)
sum_h = st(ssum)[..., :1]
print(
    "CHECK ttnn.sum vs torch fp32 sum max abs diff:",
    (sum_h - sel_h.sum(-1, keepdim=True)).abs().max().item(),
    "vs fp64:",
    (sum_h.double() - sel_h.double().sum(-1, keepdim=True)).abs().max().item(),
)
mmf = st(ttnn.matmul(sel, ones_ee32, compute_kernel_config=ex, memory_config=L1))
print(
    "CHECK mm fp32 ones sum vs ttnn.sum: max",
    (mmf - sum_h).abs().max().item(),
    "equal",
    torch.equal(mmf, sum_h.expand_as(mmf)),
)
try:
    rk = st(ttnn.repeat(kth_s, ttnn.Shape((1, 1, 1, E))))
    print("CHECK repeat exact:", torch.equal(rk, st(kth_s)[..., :1].expand_as(rk)))
except Exception as e:
    print("CHECK repeat ERR", str(e)[:100])
try:
    u = ttnn.untilize_with_unpadding(idx, output_tensor_end=(0, 0, 3, K - 1), memory_config=L1)
    print(
        "CHECK untilize idx:",
        u.dtype,
        u.layout,
        u.shape,
        torch.equal(ttnn.to_torch(u).long(), ttnn.to_torch(idx).long()[..., :4, :K]),
    )
except Exception as e:
    print("CHECK untilize ERR", str(e)[:100])
try:
    a = st(ttnn.bcast(dn, wt, math_op=ttnn.BcastOpMath.MUL, dim=ttnn.BcastOpDim.W, memory_config=L1))
    b = st(ttnn.multiply(dn, wt, memory_config=L1))
    print("CHECK legacy bcast == BinaryNg:", torch.equal(a, b), (a - b).abs().max().item())
except Exception as e:
    print("CHECK bcast ERR", str(e)[:100])
ttnn.close_mesh_device(md)
