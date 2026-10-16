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


X = ttnn.matmul(idx_f, R_t, compute_kernel_config=ex, memory_config=L1)
OH = ttnn.eq(X, C_t, memory_config=L1)
ref_mask = ttnn.to_torch(ttnn.matmul(OH, S_t, compute_kernel_config=ex, memory_config=L1)).float()


def cfg_reuse(gx, gy, ibw, pm, pn):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=pn,
        per_core_M=pm,
        per_core_N=pn,
    )


for ibw in (1, 2, 4, 8):
    for pn in (1, 2):
        c = cfg_reuse(8, 8, ibw, 1, pn)
        run(
            f"mm S reuse ibw{ibw} pn{pn}",
            lambda c=c: ttnn.matmul(OH, S_t, compute_kernel_config=ex, program_config=c, memory_config=L1),
        )
        try:
            ok = torch.equal(
                ttnn.to_torch(
                    ttnn.matmul(OH, S_t, compute_kernel_config=ex, program_config=c, memory_config=L1)
                ).float(),
                ref_mask,
            )
            print(f"CHECK mm S reuse ibw{ibw} pn{pn} exact {ok}")
        except Exception as e:
            print("CHECK ERR", str(e)[:150])
for ibw in (2, 4, 8):
    c = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    run(
        f"mm S 1d ibw{ibw}",
        lambda c=c: ttnn.matmul(OH, S_t, compute_kernel_config=ex, program_config=c, memory_config=L1),
    )
# batched broadcast-A via reuse config
Rb = torch.zeros(1, K, K, E)
for k in range(K):
    Rb[0, k, k] = 1
Rb_t = dev(Rb, ttnn.bfloat16)
Cb_t = dev(torch.arange(E).float().reshape(1, 1, 1, E).expand(1, K, T, E).contiguous(), ttnn.bfloat16)
cb = cfg_reuse(8, 8, 1, 1, 2)
run(
    "mm batched R reuse",
    lambda: ttnn.matmul(idx_f, Rb_t, compute_kernel_config=ex, program_config=cb, memory_config=L1),
)
try:
    Xb = ttnn.matmul(idx_f, Rb_t, compute_kernel_config=ex, program_config=cb, memory_config=L1)
    OHb = ttnn.eq(Xb, Cb_t, memory_config=L1)
    run("eq batched", lambda: ttnn.eq(Xb, Cb_t, memory_config=L1))
    run("fast_reduce_nc OHb", lambda: ttnn.experimental.fast_reduce_nc(OHb, dims=[1], memory_config=L1))
    run("sum dim1 OHb", lambda: ttnn.sum(OHb, dim=1, keepdim=True, memory_config=L1))
    print(
        "CHECK batched mask exact:",
        torch.equal(ttnn.to_torch(ttnn.experimental.fast_reduce_nc(OHb, dims=[1], memory_config=L1)).float(), ref_mask),
    )
except Exception as e:
    print("batched ERR", str(e)[:300])
# scatter in tile layout
z = dev(torch.zeros(1, 1, T, E), ttnn.bfloat16)
onesk = dev(torch.ones(1, 1, T, K), ttnn.bfloat16)
try:
    run("scatter tile bf16", lambda: ttnn.scatter(z, -1, idx, onesk))
    print("CHECK scatter mask exact:", torch.equal(ttnn.to_torch(ttnn.scatter(z, -1, idx, onesk)).float(), ref_mask))
except Exception as e:
    print("scatter ERR", str(e)[:300])
ttnn.close_mesh_device(md)
