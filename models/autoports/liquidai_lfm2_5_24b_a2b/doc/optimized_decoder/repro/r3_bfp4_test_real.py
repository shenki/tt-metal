# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round-3 BFP4 / LoFi precision matrix on REAL (in-distribution) layer inputs.

Per layer (env R3_LAYERS, e.g. "2,3,6"): real weights, inputs = hidden state entering that layer
(generated/lfm2_real_hidden.pt; layer 0 = embedding rows of the same prompts), HF references
computed once, then for every applicable policy tag (env R3_TAGS, default all; R3_EXTRA='{"tag":
{override}}' adds tags) one OptimizedDecoder (DEFAULT_POLICY + override):
  * phase A: the test_optimized_deep_layers_real_hidden case (4 prompts 396/300/129/61 in slots
    1/5/9/13, chunk 128, 4 traced decode steps, full 32-slot decode)
  * phase B (MoE layers): bucket-1 compact decode -- prompt 0 (396) in slot 0, active_slots=1
Writes one JSON line per (layer, tag) to R3_OUT with min output/update PCC prefill/decode.
No assertion: this is a measurement matrix (bars: out >= 0.995, update >= 0.99).
"""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ..reference.hf_layer import build_hf_layer, hf_layer_forward, near_tie_alternatives
from ..tt.model_config import TILE, Lfm2MoeArgs
from ..tt.optimized_decoder import DEFAULT_POLICY, make_optimized, replace
from .harness import (
    DecodeInputs,
    PagedLayout,
    User,
    decode_step,
    from_dev,
    hf_config,
    layer_state_dict,
    pcc,
    run_prefill,
)
from .test_functional_decoder import DELTA_PCC_BAR, PCC_BAR, _real_embed, _restore_state, _snapshot_state, delta_pccs

# tag -> (applies-to group, override)
TAGS = {
    "default": ("all", {}),
    # experts: stacked (decode + <96-row prefill) copy only
    "e_gu4_dec_hifi2": ("moe", {"expert_gate_up_decode": "bfp4"}),
    "e_gu4_dec_lofi": ("moe", {"expert_gate_up_decode": "bfp4", "expert_fidelity": "lofi"}),
    "e_both4_dec_hifi2": ("moe", {"expert_gate_up_decode": "bfp4", "expert_down_decode": "bfp4"}),
    "e_both4_dec_lofi": (
        "moe",
        {"expert_gate_up_decode": "bfp4", "expert_down_decode": "bfp4", "expert_fidelity": "lofi"},
    ),
    # experts: both copies (prefill dispatch copy too; dispatch fidelity is already LoFi in DEFAULT)
    "e_both4_all_hifi2": ("moe", {"expert_gate_up": "bfp4", "expert_down": "bfp4"}),
    "e_both4_all_lofi": ("moe", {"expert_gate_up": "bfp4", "expert_down": "bfp4", "expert_fidelity": "lofi"}),
    # attention projections
    "a_qkv4": ("attn", {"attn_qkv": "bfp4"}),
    "a_out4": ("attn", {"attn_out": "bfp4"}),
    "a_both4": ("attn", {"attn_qkv": "bfp4", "attn_out": "bfp4"}),
    "a_bfp8_lofi": ("attn", {"attn_fidelity": "lofi"}),
    # conv / dense (layers 0/1 and conv_moe layers)
    "c_in4": ("conv", {"conv_in": "bfp4"}),
    "c_out4": ("conv", {"conv_out": "bfp4"}),
    "c_bfp8_lofi": ("conv", {"conv_fidelity": "lofi"}),
    "d_w13_4": ("dense", {"dense_w13": "bfp4"}),
    "d_w2_4": ("dense", {"dense_w2": "bfp4"}),
    "d_both4": ("dense", {"dense_w13": "bfp4", "dense_w2": "bfp4"}),
    "d_both4_lofi": ("dense", {"dense_w13": "bfp4", "dense_w2": "bfp4", "dense_fidelity": "lofi"}),
    "d_bfp8_lofi": ("dense", {"dense_fidelity": "lofi"}),
}
PROMPT_LENS = (396, 300, 129, 61)
DECODE_STEPS = 4
CHUNK = 128


def _groups(kind):
    g = {"all"}
    g |= {"attn"} if kind.startswith("attn") else {"conv"}
    g |= {"moe"} if kind.endswith("moe") else {"dense"}
    return g


def _layer_inputs(layer):
    d = torch.load("generated/lfm2_real_hidden.pt")
    if layer == 0:  # the decoder input of layer 0 is the embedding of the prompt tokens
        embed = _real_embed()
        return torch.stack([embed[torch.tensor(p)].float() for p in d["prompts"]])
    return d["hidden"][layer].float()


def _decode_phase(decoder, rope, mesh_device, layout, users, active_slots):
    H = users[0].x.shape[-1]
    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq, active_slots=active_slots)
    table = layout.decode_table(users)
    garbage = torch.Generator().manual_seed(1234)
    trace_id = None
    for t in range(max(u.decode_steps for u in users)):
        x = torch.randn(1, 1, TILE, H, generator=garbage) * 0.5
        cur = torch.full((TILE,), -1, dtype=torch.int32)
        for u in users:
            if t < u.decode_steps:
                x[0, 0, u.slot] = u.x[u.prompt_len + t]
                cur[u.slot] = u.prompt_len + t
        inp.update(x, cur, table)
        if trace_id is None:
            snap = _snapshot_state(decoder)
            ttnn.deallocate(decode_step(decoder, rope, inp))
            _restore_state(decoder, snap)
            ttnn.synchronize_device(mesh_device)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            trace_out = decode_step(decoder, rope, inp)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            _restore_state(decoder, snap)
            del snap
            inp.update(x, cur, table)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        host = from_dev(trace_out, mesh_device)[0, 0]
        for u in users:
            if t < u.decode_steps:
                u.tt_decode.append(host[u.slot])
    ttnn.release_trace(mesh_device, trace_id)
    del inp


def _score(users):
    out_p, out_d, up_p, up_d, ties = [], [], [], [], 0
    for u in users:
        out_p.append(pcc(u.tt_prefill, u.ref[: u.prompt_len]))
        out_d.append(pcc(torch.stack(u.tt_decode), u.ref[u.prompt_len : u.prompt_len + u.decode_steps]))
        dp, dd = delta_pccs(u)
        up_p.append(dp)
        up_d.append(dd)
        ties += hasattr(u, "near_tie")
    r = dict(out_p=min(out_p), out_d=min(out_d), upd_p=min(up_p), upd_d=min(up_d), near_ties=ties)
    r["per_user"] = [
        [round(a, 6), round(b, 6), round(c, 6), round(d, 6)] for a, b, c, d in zip(out_p, out_d, up_p, up_d)
    ]
    r["pass"] = (
        r["out_p"] >= PCC_BAR and r["out_d"] >= PCC_BAR and r["upd_p"] >= DELTA_PCC_BAR and r["upd_d"] >= DELTA_PCC_BAR
    )
    return r


def _fresh(u):
    v = User(slot=u.slot, prompt_len=u.prompt_len, decode_steps=u.decode_steps, seed=u.seed)
    v.x, v.ref, v.ref_alts, v.blocks = u.x, u.ref, u.ref_alts, u.blocks
    return v


def _layers():
    return [int(s) for s in os.environ.get("R3_LAYERS", "2,3").split(",")]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer", _layers())
def test_r3_bfp4_real(mesh_device, layer):
    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    kind = args.layer_kind(layer)
    tags = dict(TAGS)
    tags.update({k: ("all", v) for k, v in json.loads(os.environ.get("R3_EXTRA", "{}")).items()})
    want = os.environ.get("R3_TAGS")
    names = [t for t, (g, _) in tags.items() if g in _groups(kind) and (not want or t in want.split(","))]
    out_path = Path(os.environ.get("R3_OUT", "generated/r3_bfp4_real.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sd = layer_state_dict(layer, True)
    hidden = _layer_inputs(layer)
    users = [User(slot=4 * p + 1, prompt_len=n, decode_steps=DECODE_STEPS, seed=p) for p, n in enumerate(PROMPT_LENS)]
    moe = kind.endswith("moe")
    b1 = [User(slot=0, prompt_len=PROMPT_LENS[0], decode_steps=DECODE_STEPS, seed=0)] if moe else []
    hf_layer, hf_rope = build_hf_layer(cfg, layer, sd)
    for u in users:
        u.x = hidden[u.seed, : u.prompt_len + u.decode_steps].to(torch.bfloat16).float()
        u.ref = hf_layer_forward(hf_layer, hf_rope, u.x.unsqueeze(0))[0]
        u.ref_alts = near_tie_alternatives(hf_layer, hf_rope, u.x, range(u.prompt_len, u.prompt_len + u.decode_steps))
    for u in b1:
        u.x, u.ref, u.ref_alts = users[0].x, users[0].ref, users[0].ref_alts
    del hf_layer, hidden
    max_seq = max(TILE, max(PROMPT_LENS) + DECODE_STEPS)
    layout = PagedLayout(users + b1, max_seq)

    for tag in names:
        policy = replace(DEFAULT_POLICY, **tags[tag][1])
        decoder, rope = make_optimized(sd, cfg, args, layer, mesh_device, max_seq, policy=policy)
        if decoder.is_attention:
            decoder.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
        ua = [_fresh(u) for u in users]
        for u in ua:
            u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, CHUNK)
        _decode_phase(decoder, rope, mesh_device, layout, ua, None)
        row = {"layer": layer, "kind": kind, "tag": tag, "override": tags[tag][1], **_score(ua)}
        if b1:
            ub = [_fresh(u) for u in b1]
            for u in ub:
                u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, CHUNK)
            _decode_phase(decoder, rope, mesh_device, layout, ub, 1)
            sb = _score(ub)
            row.update(b1_out_d=sb["out_d"], b1_upd_d=sb["upd_d"], b1_out_p=sb["out_p"], b1_upd_p=sb["upd_p"])
            row["pass"] = row["pass"] and sb["out_d"] >= PCC_BAR and sb["upd_d"] >= DELTA_PCC_BAR
        logger.info(f"R3ROW {json.dumps(row)}")
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        del decoder, rope, ua
        gc.collect()
        ttnn.synchronize_device(mesh_device)
