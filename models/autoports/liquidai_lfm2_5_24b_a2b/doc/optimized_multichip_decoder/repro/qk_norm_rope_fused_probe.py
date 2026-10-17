# Stage 5 review item 3 (audit R2/B3): fused per-head q/k RMSNorm (+ RoPE) for decode vs the current chain.
#   current (per q / k): nlp_create_qkv_heads_decode gives height-sharded [1, 32 users, 32(pad heads), 64]; S2I -> rms_norm(weight)
#            -> I2S -> rotary_embedding_hf (3 + 1 ops, x2 for q and k)
#   fused:   dit_fused_distributed_rmsnorm(cluster_axis=None, per_head_norm=True, num_heads_per_device=8, weight) on the q
#            block of the qkv row [1, 1, 32, 512] (TILE interleaved) -- 1 op for 8 heads; RoPE variant with rope_cos/sin +
#            transformation_mat (the op's RoPE is the llama trans-mat convention: needs head-dim-permuted q/k weights and
#            cache, see work_log).  Single chip, traced timing, correctness of the norm vs torch.
#   TT_VISIBLE_DEVICES=0 python qk_norm_rope_fused_probe.py
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import decode_head_mem
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import _ckc

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
L1 = ttnn.L1_MEMORY_CONFIG
U, NH, HD, eps = 32, 8, 64, 1e-5
g = torch.Generator().manual_seed(0)
q_t = torch.randn(1, 1, U, NH * HD, generator=g)
w_t = torch.rand(HD, generator=g) + 0.5
ref = (q_t.reshape(U, NH, HD) * torch.rsqrt(q_t.reshape(U, NH, HD).pow(2).mean(-1, keepdim=True) + eps) * w_t).reshape(
    1, 1, U, NH * HD
)
ckc4 = _ckc(md, "hifi4")


def timed(f, reps=16):
    o = f()
    ttnn.synchronize_device(md)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        for _ in range(reps):
            ttnn.deallocate(f())
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(5):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t0) / (5 * reps) * 1e6
    ttnn.release_trace(md, tid)
    return us


# current: heads height-sharded [1, U, 32(pad), 64] -> S2I -> rms_norm(weight) -> I2S
head_mem = decode_head_mem(md, U, HD)
qh_t = torch.zeros(1, U, 32, HD)
qh_t[0, :, :NH] = q_t.reshape(U, NH, HD)
qh = ttnn.from_torch(qh_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=head_mem)
w_rm = ttnn.from_torch(w_t.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)


def current():
    a = ttnn.to_memory_config(qh, L1)
    b = ttnn.rms_norm(a, epsilon=eps, weight=w_rm, memory_config=L1, compute_kernel_config=ckc4)
    ttnn.deallocate(a)
    c = ttnn.to_memory_config(b, head_mem)
    ttnn.deallocate(b)
    return c


o = current()
got = ttnn.to_torch(o).float()[0, :, :NH].reshape(1, 1, U, NH * HD)
print(
    f"QKFUSE current (S2I + rms_norm + I2S, one of q/k): {timed(current):6.2f} us  pcc {torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]:.6f}",
    flush=True,
)

# fused per-head norm on the [1, 1, U, NH*HD] row block
qr = ttnn.from_torch(q_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=L1)
w_tile = ttnn.from_torch(
    w_t.repeat(NH).reshape(1, 1, 1, NH * HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md
)
for label, kw in [("per-head norm", {}), ("per-head norm, weight per head", {"weight": w_tile})]:
    try:
        f = lambda kw=kw: ttnn.experimental.dit_fused_distributed_rmsnorm(
            qr,
            None,
            md,
            [],
            epsilon=eps,
            num_heads_per_device=NH,
            per_head_norm=True,
            memory_config=L1,
            compute_kernel_config=ckc4,
            **kw,
        )
        o = f()
        got = ttnn.to_torch(o).float()
        print(f"QKFUSE fused output shape {tuple(o.shape)}", flush=True)
        if got.shape[1] == NH:  # head-split output [1, NH, U, HD] -> rows of the [U, NH*HD] reference
            got = got[0].permute(1, 0, 2).reshape(1, 1, U, NH * HD)
        r = ref if "weight" in kw else (ref / w_t.repeat(NH))
        print(
            f"QKFUSE fused {label:32s}: {timed(f):6.2f} us  pcc {torch.corrcoef(torch.stack([got.flatten(), r.flatten()]))[0, 1]:.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"QKFUSE fused {label:32s}: ERR {str(e).splitlines()[0][:200]}", flush=True)
ttnn.close_mesh_device(md)
