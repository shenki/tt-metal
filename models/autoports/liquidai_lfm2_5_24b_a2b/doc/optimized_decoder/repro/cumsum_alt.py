import time

import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
P, E = 8192, 64
G = P // 32
g = torch.Generator().manual_seed(0)
eids = torch.randint(0, E, (P,), generator=g)
oh = torch.nn.functional.one_hot(eids, E).float()
ref = oh.cumsum(0)
tt = lambda a, dt=ttnn.float32: ttnn.from_torch(a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=md)
toh = tt(oh)
L32 = tt(torch.tril(torch.ones(32, 32)).reshape(1, 1, 32, 32).expand(1, G, 32, 32).contiguous(), ttnn.bfloat16)
Ls = tt(torch.tril(torch.ones(G, G), -1), ttnn.bfloat16)  # strict lower: exclusive block prefix
hifi4 = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)
lofi = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def a():
    return ttnn.cumsum(toh, dim=0)


def b():
    c1 = ttnn.cumsum(ttnn.reshape(toh, (G, 32, E)), dim=1)
    tot = ttnn.reshape(ttnn.slice(c1, (0, 31, 0), (G, 32, E)), (G, E))
    pre = ttnn.subtract(ttnn.cumsum(tot, dim=0), tot)
    return ttnn.reshape(ttnn.add(c1, ttnn.reshape(pre, (G, 1, E))), (P, E))


def c(ckc):
    def f():
        oh_b = ttnn.typecast(ttnn.reshape(toh, (1, G, 32, E)), ttnn.bfloat16)
        local = ttnn.matmul(L32, oh_b, compute_kernel_config=ckc, dtype=ttnn.float32)  # [1,G,32,E]
        tot = ttnn.reshape(ttnn.slice(local, (0, 0, 31, 0), (1, G, 32, E)), (G, E))
        pre = ttnn.matmul(
            Ls, ttnn.typecast(tot, ttnn.bfloat16), compute_kernel_config=ckc, dtype=ttnn.float32
        )  # exact: tot <= 32
        return ttnn.reshape(ttnn.add(local, ttnn.reshape(pre, (1, G, 1, E))), (P, E))

    return f


def timeit(f, n=3, reps=3):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(n):
            ttnn.deallocate(f())
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    try:
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / (n * reps) * 1e3
    finally:
        ttnn.release_trace(md, tid)


for name, f in [
    ("a cumsum dim0", a),
    ("b two-level cumsum", b),
    ("c matmul scan hifi4", c(hifi4)),
    ("c matmul scan lofi", c(lofi)),
]:
    try:
        out = ttnn.to_torch(f()).float()
        ok = torch.equal(out, ref)
        print(f"CUMSUM {name:22s}: exact={ok} max|err|={(out-ref).abs().max():.1f}  {timeit(f):7.3f} ms", flush=True)
    except Exception as e:
        print(f"CUMSUM {name:22s}: FAIL {str(e).splitlines()[0][:160]}", flush=True)
ttnn.close_mesh_device(md)
