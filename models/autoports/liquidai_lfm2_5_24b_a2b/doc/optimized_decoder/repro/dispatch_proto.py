"""Prototype: token-dispatched routed-expert MoE on one Blackhole chip, fabric-free.

route (fp32, on host for this prototype) -> per-pair rank / region offsets on device (fp32 cumsum)
-> expert-sorted buffer via row gather (embedding) -> unified_routed_expert_moe -> gather back + weighted sum.
Compares against torch (real layer weights) and times the device part.
"""
import sys
import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import layer_inputs, local_model_path
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.weights import (
    load_layer_state_dict,
    load_tensors,
    stacked_expert_weights,
)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ACTIVE = int(sys.argv[2]) if len(sys.argv) > 2 else T
E, K, H, I = 64, 4, 2048, 1536
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
DR = ttnn.DRAM_MEMORY_CONFIG
p = local_model_path()
sd = load_layer_state_dict(p, 3)
emb = load_tensors(p, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
gate, up, down = stacked_expert_weights(sd, E)  # [E,I,H],[E,I,H],[E,H,I]

# inputs ~ post-ffn_norm activations
x = layer_inputs(T, H, 3, emb).float()
x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * sd["ffn_norm.weight"].float()).bfloat16().float()
s = torch.sigmoid(x @ sd["feed_forward.gate.weight"].float().T)
idx = (s + sd["feed_forward.expert_bias"].float()).topk(K).indices
w = s.gather(-1, idx)
w = w / (w.sum(-1, keepdim=True) + 1e-6)
idx[ACTIVE:] = E
w[ACTIVE:] = 0.0
# torch reference (bf16-rounded weights like the TT bf16->bfp8 path is not modelled: fp32 ref)
ref = torch.zeros(T, H)
for e in range(E):
    rows, slot = (idx == e).nonzero(as_tuple=True)  # sentinel E never matches
    if len(rows) == 0:
        continue
    xe = x[rows]
    ye = (torch.nn.functional.silu(xe @ gate[e].float().T) * (xe @ up[e].float().T)) @ down[e].float().T
    ref.index_add_(0, rows, ye * w[rows, slot, None])

t = lambda a, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
    a.contiguous(), dtype=dt, layout=lay, device=md, memory_config=DR
)
wdt = ttnn.bfloat8_b
gates = [t(gate[e].T, wdt) for e in range(E)]
ups = [t(up[e].T, wdt) for e in range(E)]
downs = [t(down[e].T, wdt) for e in range(E)]
gidx = t(torch.arange(E, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
arange_e = t(torch.arange(E, dtype=torch.float32).reshape(1, E), ttnn.float32)
P = T * K
MAXBUF = ((P + E * 31) + 31) // 32 * 32 + 32  # + trash tile for idle pairs
TRASH = MAXBUF - 1

tx = t(x.reshape(1, 1, T, H))
tidx = t(idx.to(torch.int32).reshape(T, K), ttnn.uint16)  # like the router's topk output
tw = t(w.reshape(1, 1, T, K), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
tvals = t(torch.arange(P, dtype=torch.float32).div(K).floor().reshape(1, P), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
zero_row = t(torch.zeros(1, 1, 1, H), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
src_init = t(torch.full((1, MAXBUF), float(T)), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


def moe():
    rm = lambda a: ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
    # pair -> expert one-hot [P, E] (fp32)
    e_col = ttnn.reshape(ttnn.typecast(tidx, ttnn.float32), (P, 1))
    onehot = ttnn.eq(e_col, arange_e)  # [P, E]
    onehot = ttnn.typecast(onehot, ttnn.float32) if onehot.dtype != ttnn.float32 else onehot
    cum = ttnn.cumsum(onehot, dim=0)  # inclusive running count per expert
    counts = ttnn.slice(cum, (P - 1, 0), (P, E))  # [1, E]
    aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(counts, 31.0), 1.0 / 32)), 32.0)
    region = ttnn.subtract(ttnn.cumsum(aligned, dim=-1), aligned)  # exclusive
    # dest row of each pair = region[e] + rank
    dest = ttnn.sum(ttnn.multiply(onehot, ttnn.add(ttnn.add(cum, region), -1.0)), dim=-1, keepdim=True)  # [P, 1]
    idle = ttnn.rsub(ttnn.sum(onehot, dim=-1, keepdim=True), 1.0)  # 1 for sentinel pairs
    dest = ttnn.mac(idle, float(TRASH), dest)
    dest_u = ttnn.typecast(rm(ttnn.reshape(dest, (1, P))), ttnn.uint32)
    # destination -> source token (pad rows -> zero row T)
    src = ttnn.scatter(src_init, -1, dest_u, tvals)
    src_u = ttnn.typecast(src, ttnn.uint32)
    table = ttnn.concat([rm(tx), zero_row], dim=2)  # [1,1,T+1,H]
    table = ttnn.reshape(table, (T + 1, H))
    buf = ttnn.embedding(src_u, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)  # [1, MAXBUF, H]
    buf = ttnn.reshape(buf, (MAXBUF, H))
    region_u = ttnn.typecast(rm(ttnn.reshape(region, (1, E))), ttnn.uint32)
    counts_u = ttnn.typecast(rm(ttnn.reshape(counts, (1, E))), ttnn.uint32)
    region_u = ttnn.reshape(region_u, (E,))
    counts_u = ttnn.reshape(counts_u, (E,))
    out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        buf, region_u, counts_u, gidx, gates, ups, downs, T, compute_kernel_config=ckc
    )
    out_rm = ttnn.to_layout(ttnn.typecast(out, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    comb = ttnn.embedding(dest_u, out_rm, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)  # [1, P, H]
    comb = ttnn.reshape(comb, (1, 1, T, K, H))
    y = ttnn.experimental.deepseek_prefill.post_combine_reduce(
        comb, ttnn.reshape(tw, (1, 1, T, K, 1)), expert_dim=3, output_memory_config=DR
    )
    return y


y = moe()
yt = ttnn.to_torch(y).float().reshape(T, H)
pcc = torch.corrcoef(torch.stack([yt.flatten(), ref.flatten()]))[0, 1].item()
print(
    f"PROTO T={T} active={ACTIVE} PCC vs torch fp32 {pcc:.5f} max|err| {(yt-ref).abs().max():.4f} ref absmax {ref.abs().max():.3f}",
    flush=True,
)
ttnn.synchronize_device(md)
t0 = time.perf_counter()
for _ in range(5):
    ttnn.deallocate(moe())
ttnn.synchronize_device(md)
print(f"PROTO T={T} active={ACTIVE} eager wall per call {(time.perf_counter()-t0)/5*1e3:.2f} ms", flush=True)
tid = ttnn.begin_trace_capture(md, cq_id=0)
try:
    yy = moe()
finally:
    ttnn.end_trace_capture(md, tid, cq_id=0)
ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
t0 = time.perf_counter()
for _ in range(10):
    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
ttnn.synchronize_device(md)
print(f"PROTO T={T} active={ACTIVE} traced per call {(time.perf_counter()-t0)/10*1e3:.3f} ms", flush=True)
ttnn.release_trace(md, tid)
ttnn.close_mesh_device(md)
