import time

import torch

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_inputs, local_model_path
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.fused_decoder import DEFAULT_FUSIONS
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import Lfm2MoeArgs
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DEFAULT_POLICY, OptimizedMoE, replace
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.weights import load_layer_state_dict, load_tensors

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
p = local_model_path()
args = Lfm2MoeArgs.from_hf_config(hf_config())
sd = load_layer_state_dict(p, 3)
emb = load_tensors(p, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
moe = OptimizedMoE(md, args, sd, replace(DEFAULT_POLICY, moe_dispatch_min_rows=64), DEFAULT_FUSIONS)


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


for T in (64, 96, 128, 192, 256, 512, 1024, 2048):
    x = layer_inputs(T, 2048, T, emb).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * sd["ffn_norm.weight"].float()
    tx = ttnn.from_torch(x.reshape(1, 1, T, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
    sp = timeit(lambda: super(OptimizedMoE, moe).forward(tx))  # FusedMoE/MoE sparse prefill path
    dp = timeit(lambda: moe._dispatch_forward(tx))
    a = ttnn.to_torch(super(OptimizedMoE, moe).forward(tx)).float().flatten()
    b = ttnn.to_torch(moe._dispatch_forward(tx)).float().flatten()
    print(
        f"XOVER T={T:5d}: sparse {sp:8.3f} ms  dispatch {dp:8.3f} ms  pcc(sparse,dispatch) {torch.corrcoef(torch.stack([a,b]))[0,1]:.5f}",
        flush=True,
    )
    ttnn.deallocate(tx)
ttnn.close_mesh_device(md)
