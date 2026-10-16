# Review P2 #4: single-chunk prefill (fresh request, position 0) at short / boundary lengths.
# Real weights; output + update PCC vs HF; device time = traced replay of prefill_forward (wall / N).
#   TT_VISIBLE_DEVICES=3 python .../repro/r2_prefill_len_sweep.py <kind> <baseline|default|candidate> [lens...]
#   candidate = DEFAULT_POLICY + json.loads($LFM2_CAND)
import json
import math
import os
import sys
import time

import torch

import ttnn
from models.autoports.lfm2_s3snap_r2pf.reference.hf_layer import build_hf_layer, hf_layer_forward
from models.autoports.lfm2_s3snap_r2pf.tests.harness import (
    PagedLayout,
    User,
    from_dev,
    hf_config,
    layer_inputs,
    layer_state_dict,
    pad_page_table_row,
    pcc,
    to_dev,
)
from models.autoports.lfm2_s3snap_r2pf.tests.test_functional_decoder import LAYERS, _real_embed
from models.autoports.lfm2_s3snap_r2pf.tt.functional_decoder import ConvPrefillInputs
from models.autoports.lfm2_s3snap_r2pf.tt.model_config import TILE, Lfm2MoeArgs
from models.autoports.lfm2_s3snap_r2pf.tt.optimized_decoder import (
    BASELINE_POLICY,
    DEFAULT_POLICY,
    make_optimized,
    replace,
)

kind, pname = sys.argv[1], sys.argv[2]
lens = [int(v) for v in sys.argv[3:]] or [1, 31, 65, 95, 96, 97, 127, 128, 129, 300, 2047, 2048]
policy = {"baseline": BASELINE_POLICY, "default": DEFAULT_POLICY}.get(pname)
if policy is None:
    policy = replace(DEFAULT_POLICY, **json.loads(os.environ.get("LFM2_CAND", "{}")))
out_path = os.environ.get("LFM2_SWEEP_OUT", "/dev/null")
N = int(os.environ.get("LFM2_SWEEP_ITERS", "10"))

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200 << 20)
layer_idx = LAYERS[kind]
cfg = hf_config()
args = Lfm2MoeArgs.from_hf_config(cfg)
H = args.hidden_size
sd = layer_state_dict(layer_idx, True)
embed = _real_embed()
max_seq = 2048 + 2 * TILE
dec, rope = make_optimized(sd, cfg, args, layer_idx, md, max_seq, policy=policy)
hf_layer, hf_rope = build_hf_layer(cfg, layer_idx, sd)
users = [User(slot=s, prompt_len=2048, decode_steps=1, seed=900 + s) for s in range(TILE)]
layout = PagedLayout(users, max_seq)
if dec.is_attention:
    dec.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
u = users[3]

for L in lens:
    xs = layer_inputs(L, H, 4242 + L, embed).to(torch.bfloat16).float()
    with torch.no_grad():
        ref = hf_layer_forward(hf_layer, hf_rope, xs.unsqueeze(0))[0]
    s_pad = math.ceil(L / TILE) * TILE
    x = torch.zeros(1, 1, s_pad, H)
    x[0, 0, :L] = xs
    tt_x = to_dev(x, md)
    if dec.is_attention:
        nblk = math.ceil(s_pad / layout.block_size)
        kw = dict(
            rope=rope,
            chunk_start=0,
            page_table=to_dev(layout.user_table(u), md, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            chunk_page_table=to_dev(
                pad_page_table_row(u.blocks[:nblk]).reshape(1, -1), md, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
            ),
            chunk_start_tensor=to_dev(torch.tensor([0], dtype=torch.int32), md, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        )
    else:
        kw = dict(
            conv_inputs=ConvPrefillInputs.create(
                md,
                slot=u.slot,
                valid_len=L,
                padded_len=s_pad,
                continuation=False,
                windowed=getattr(dec, "conv_prefill_windowed", False),
            )
        )
    row = {
        "kind": kind,
        "policy": pname,
        "L": L,
        "cand": os.environ.get("LFM2_CAND", "") if pname not in ("baseline", "default") else "",
    }
    try:
        out = dec.prefill_forward(tt_x, **kw)
        got = from_dev(out, md)[0, 0, :L]
        ttnn.deallocate(out)
        row["pcc"] = pcc(got, ref)
        row["delta_pcc"] = pcc(got - xs, ref - xs) if L > 1 else float("nan")
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        tout = dec.prefill_forward(tt_x, **kw)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        reps = []
        for _ in range(int(os.environ.get("LFM2_SWEEP_REPS", "5"))):
            t0 = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            reps.append((time.perf_counter() - t0) / N * 1e6)
        row["us"] = min(reps)  # min over repeats of N-replay means (host noise only adds)
        row["us_median"] = sorted(reps)[len(reps) // 2]
        ttnn.release_trace(md, tid)
        ttnn.deallocate(tout)
    except Exception as e:
        row["error"] = str(e).strip().splitlines()[0][:200]
    print("SWEEP " + json.dumps(row), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    ttnn.deallocate(tt_x)
    del kw

ttnn.close_mesh_device(md)
