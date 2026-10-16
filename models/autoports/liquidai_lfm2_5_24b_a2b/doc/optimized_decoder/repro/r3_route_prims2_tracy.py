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

from tracy import signpost

REPS = 5


def run(name, f):
    try:
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
        ttnn.synchronize_device(md)
        signpost(f"{name}#R{REPS}#D")
        for _ in range(REPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        signpost(f"{name}#R{REPS}#D_END")
        ttnn.release_trace(md, tid)
        ttnn.ReadDeviceProfiler(md)
    except Exception as e:
        print(f"PRIM {name:55s} ERR {str(e).splitlines()[0][:200]}", flush=True)


lofi = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
hifi2 = ttnn.init_device_compute_kernel_config(
    md.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)
X = ttnn.matmul(idx_f, R_t, compute_kernel_config=ex, memory_config=L1)
OH = ttnn.eq(X, C_t, memory_config=L1)
run("mm S hifi4 fp32acc", lambda: ttnn.matmul(OH, S_t, compute_kernel_config=ex, memory_config=L1))
run("mm S lofi", lambda: ttnn.matmul(OH, S_t, compute_kernel_config=lofi, memory_config=L1))
run("mm R lofi", lambda: ttnn.matmul(idx_f, R_t, compute_kernel_config=lofi, memory_config=L1))
run("mm R hifi2", lambda: ttnn.matmul(idx_f, R_t, compute_kernel_config=hifi2, memory_config=L1))
Rb = torch.zeros(1, K, K, E)
for k in range(K):
    Rb[0, k, k] = 1
Rb_t = dev(Rb, ttnn.bfloat16)
Cb_t = dev(torch.arange(E).float().reshape(1, 1, 1, E).expand(1, K, T, E).contiguous(), ttnn.bfloat16)
run("mm batched R (bcast A)", lambda: ttnn.matmul(idx_f, Rb_t, compute_kernel_config=lofi, memory_config=L1))
try:
    Xb = ttnn.matmul(idx_f, Rb_t, compute_kernel_config=lofi, memory_config=L1)
    OHb = ttnn.eq(Xb, Cb_t, memory_config=L1)
    run("eq batched", lambda: ttnn.eq(Xb, Cb_t, memory_config=L1))
    run("fast_reduce_nc OHb", lambda: ttnn.experimental.fast_reduce_nc(OHb, dims=[1], memory_config=L1))
    run("sum dim1 OHb", lambda: ttnn.sum(OHb, dim=1, keepdim=True, memory_config=L1))
except Exception as e:
    print("batched ERR", str(e)[:200])
run("transpose scores fp32", lambda: ttnn.transpose(scores, -2, -1, memory_config=L1))
run("transpose idx_f bf16", lambda: ttnn.transpose(idx_f, -2, -1, memory_config=L1))
run("transpose active bf16", lambda: ttnn.transpose(active, -2, -1, memory_config=L1))
run("typecast idx->bf16", lambda: ttnn.typecast(idx, ttnn.bfloat16, memory_config=L1))
selT = ttnn.transpose(sel, -2, -1, memory_config=L1)
run("sum dim-2 selT", lambda: ttnn.sum(selT, dim=-2, keepdim=True, memory_config=L1))
denT = ttnn.add(ttnn.sum(selT, dim=-2, keepdim=True, memory_config=L1), 1e-6, memory_config=L1)
run("div row-bcast selT/denT", lambda: ttnn.div(selT, denT, memory_config=L1))
run("div row-bcast out bf16", lambda: ttnn.div(selT, denT, dtype=ttnn.bfloat16, memory_config=L1))
activeT = ttnn.transpose(active, -2, -1, memory_config=L1)
run("mul row-bcast maskT x activeT", lambda: ttnn.multiply(selT, activeT, memory_config=L1))
idx_fT = ttnn.transpose(idx_f, -2, -1, memory_config=L1)
RT_t, ST_t, CT_t = (
    dev(R.T.contiguous(), ttnn.bfloat16),
    dev(S.T.contiguous(), ttnn.bfloat16),
    dev(C.T.contiguous(), ttnn.bfloat16),
)
run("mm R^T @ idx_fT", lambda: ttnn.matmul(RT_t, idx_fT, compute_kernel_config=lofi, memory_config=L1))
XT = ttnn.matmul(RT_t, idx_fT, compute_kernel_config=lofi, memory_config=L1)
run("eq CT", lambda: ttnn.eq(XT, CT_t, memory_config=L1))
OHT = ttnn.eq(XT, CT_t, memory_config=L1)
run("mm S^T @ OHT lofi", lambda: ttnn.matmul(ST_t, OHT, compute_kernel_config=lofi, memory_config=L1))
wc = dev(torch.rand(1, 1, T, 4), ttnn.bfloat16)
wcT = dev(torch.rand(1, 1, 4, T), ttnn.bfloat16)
run("permute wc [1,1,T,n]->[1,n,T,1]", lambda: ttnn.permute(wc, (0, 3, 2, 1), memory_config=L1))
run("permute wcT [1,1,n,T]->[1,n,T,1]", lambda: ttnn.permute(wcT, (0, 2, 3, 1), memory_config=L1))
rT = dev(torch.rand(1, 1, E, T), ttnn.bfloat16)
r = dev(torch.rand(1, 1, T, E), ttnn.bfloat16)
run("permute routing [1,1,T,E]->[1,E,T,1]", lambda: ttnn.permute(r, (0, 3, 2, 1), memory_config=L1))
run("permute routingT [1,1,E,T]->[1,E,T,1]", lambda: ttnn.permute(rT, (0, 2, 3, 1), memory_config=L1))
run("reshape routingT -> [1,E,T,1]", lambda: ttnn.reshape(rT, (1, E, T, 1)))
run(
    "max dim2 + to RM (current sparsity)",
    lambda: ttnn.to_layout(ttnn.max(r, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT),
)
run("max dim-1 on routingT + to RM", lambda: ttnn.to_layout(ttnn.max(rT, dim=-1, keepdim=True), ttnn.ROW_MAJOR_LAYOUT))
run("where bcast [T,1] cur (routing bf16)", lambda: ttnn.where(active, r, 0.0, memory_config=L1))
dense = dev(torch.rand(1, 1, T, E), ttnn.bfloat16)
Rep = torch.zeros(E, K * E)
for k in range(K):
    Rep[:, k * E : (k + 1) * E] = torch.eye(E)
Rep_t = dev(Rep, ttnn.bfloat16)
run("mm dense @ Rep lofi", lambda: ttnn.matmul(dense, Rep_t, compute_kernel_config=lofi, memory_config=L1))
run("mm dense @ Rep hifi4", lambda: ttnn.matmul(dense, Rep_t, compute_kernel_config=ex, memory_config=L1))
DR = ttnn.matmul(dense, Rep_t, compute_kernel_config=ex, memory_config=L1)
run("mul OH x denseRep", lambda: ttnn.multiply(OH, DR, memory_config=L1))
Sp = torch.zeros(K * E, K)
for k in range(K):
    Sp[k * E : (k + 1) * E, k] = 1
Sp_t = dev(Sp, ttnn.bfloat16)
P = ttnn.multiply(OH, DR, memory_config=L1)
run("mm P @ S' hifi4", lambda: ttnn.matmul(P, Sp_t, compute_kernel_config=ex, memory_config=L1))
run("mm P @ S' lofi", lambda: ttnn.matmul(P, Sp_t, compute_kernel_config=lofi, memory_config=L1))
onehot = dev((torch.rand(E, 4) > 0.9).float(), ttnn.bfloat16)
run("mm dense @ onehot (cur)", lambda: ttnn.matmul(dense, onehot, compute_kernel_config=ex, memory_config=L1))
# exactness
st = lambda t: ttnn.to_torch(t).float()
print(
    "CHECK lofi mmS exact:",
    torch.equal(
        st(ttnn.matmul(OH, S_t, compute_kernel_config=lofi, memory_config=L1)),
        st(ttnn.matmul(OH, S_t, compute_kernel_config=ex, memory_config=L1)),
    ),
)
print(
    "CHECK lofi mmR exact:",
    torch.equal(st(ttnn.matmul(idx_f, R_t, compute_kernel_config=lofi, memory_config=L1)), st(X)),
)
print(
    "CHECK hifi2 mmR exact:",
    torch.equal(st(ttnn.matmul(idx_f, R_t, compute_kernel_config=hifi2, memory_config=L1)), st(X)),
)
print(
    "CHECK lofi denseRep exact:",
    torch.equal(st(ttnn.matmul(dense, Rep_t, compute_kernel_config=lofi, memory_config=L1)), st(DR)),
    torch.equal(st(DR)[..., :E], st(dense)),
)
print(
    "CHECK lofi P@S' exact:",
    torch.equal(
        st(ttnn.matmul(P, Sp_t, compute_kernel_config=lofi, memory_config=L1)),
        st(ttnn.matmul(P, Sp_t, compute_kernel_config=ex, memory_config=L1)),
    ),
)
a = st(ttnn.div(sel, den, memory_config=L1))
b = st(ttnn.typecast(ttnn.div(sel, den, memory_config=L1), ttnn.bfloat16))
c = st(ttnn.div(sel, den, dtype=ttnn.bfloat16, memory_config=L1))
print("CHECK div->bf16 fused == div+typecast:", torch.equal(b, c), (b - c).abs().max().item())
dT = st(ttnn.div(selT, denT, memory_config=L1)).transpose(-1, -2)
print("CHECK transposed div == div bcast:", torch.equal(dT, a), (dT - a).abs().max().item())
ttnn.close_mesh_device(md)
