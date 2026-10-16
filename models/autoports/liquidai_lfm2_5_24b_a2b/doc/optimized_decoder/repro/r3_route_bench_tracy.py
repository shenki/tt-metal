# Tracy (run via r3_route_tracy.sh): traced device time of the decode routing chains, current vs
# Policy.moe_route_v3, on a real 32-token batch of layer 3 (conv_moe) router inputs.
import json

import torch
from tracy import signpost

import ttnn
from models.autoports.lfm2_r3_route.tests.harness import hf_config, layer_state_dict
from models.autoports.lfm2_r3_route.tt.fused_decoder import DEFAULT_FUSIONS
from models.autoports.lfm2_r3_route.tt.model_config import Lfm2MoeArgs
from models.autoports.lfm2_r3_route.tt.optimized_decoder import DEFAULT_POLICY, OptimizedMoE, ShardedNorm, _ckc, replace

L1 = ttnn.L1_MEMORY_CONFIG
import os

extra = json.loads(os.environ.get("R3_EXTRA", "{}"))
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
args = Lfm2MoeArgs.from_hf_config(hf_config())
moe = OptimizedMoE(
    md, args, layer_state_dict(3, True), replace(DEFAULT_POLICY, moe_route_v3=True, **extra), DEFAULT_FUSIONS
)
norm = ShardedNorm(md, moe.H, 8, 1e-5, _ckc(md, "hifi4"))
T = 32
x = ttnn.from_torch(
    torch.load("generated/r3_route_hn.pt")[3][:T].reshape(1, 1, T, -1),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    memory_config=L1,
)
moe._x_sharded = ttnn.to_memory_config(x, norm.mem)
a = torch.ones(1, 1, T, 1)
sa = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, memory_config=L1)
REPS = 5


def run(name, f):
    def g():
        o = f()
        outs = o if isinstance(o, (list, tuple)) else [o]
        for t in outs:
            ttnn.deallocate(t)

    g()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        g()
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


run("router_scores_only", lambda: moe._router_scores(x))
cur_pol, new_pol = replace(moe.policy, router_ds_fused=False), moe.policy


def cur(f):
    def g():
        moe.policy = cur_pol
        try:
            return f()
        finally:
            moe.policy = new_pol

    return g


run("cur_wide (route + where(slot_active))", cur(lambda: ttnn.where(sa, moe.route(x), 0.0)))
run("v3_wide", lambda: moe._route_v3_wide(x, sa))
for b in (1, 2, 4):

    def cur_c(b=b):
        ids, wc = moe._compact_route(x, sa, b)
        return [ids, ttnn.permute(ttnn.reshape(wc, (1, 1, T, b * moe.K)), (0, 3, 2, 1), memory_config=L1)]

    run(f"cur_compact_b{b} (+wt permute)", cur(cur_c))

    def v3_c(b=b):
        ids, wc = moe._compact_route_v3(x, b)
        if True:
            wc = ttnn.permute(ttnn.reshape(wc, (1, 1, T, b * moe.K)), (0, 3, 2, 1), memory_config=L1)
        return [ids, wc]

    run(f"v3_compact_b{b} (wt layout)", v3_c)
ttnn.close_mesh_device(md)
