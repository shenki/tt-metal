# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The precision config is consumed by the default construction path (stage 8).

Builds the reduced probe (layers 0, 2, 3 + embedding / final norm / LM head) through ``build_generator`` WITHOUT any
precision argument (``tt.precision.resolve()``: ``$LFM2_PRECISION_CONFIG`` if set, else
``doc/datatype_sweep/selected_precision_config.json``) and checks that every field of the resolved config reached the
built device objects: weight dtypes per group (decode and prefill expert copies, attention, ShortConv, dense, LM head),
compute fidelities per module, KV-cache dtype, decode all-reduce payload dtypes (partials + scratch), layer exceptions;
plus the escape hatch ``LFM2_PRECISION_CONFIG=baseline``."""

import os

import ttnn

from ..tt.generator import build_generator, open_mesh
from ..tt.optimized_decoder import DTYPES
from ..tt.precision import SELECTED_PATH, resolve

PROBE = (0, 2, 3)
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}


def _fid(ckc):
    return ckc.math_fidelity


def check(gen, pc):
    m = gen.model
    assert m.precision.config_id == pc.config_id and m.precision_source == pc.source, (m.precision_source, pc.source)
    assert m.head.weight.dtype == DTYPES[pc.head["dtype"]] and _fid(m.head_ckc) == FID[pc.head["fidelity"]]
    assert tuple(m.head_cfg) == tuple(pc.head["cfg"])
    checked = 0
    for l in m.layers:
        p = pc.policy_for_layer(l.layer_idx)
        mix, ffn = l.mixer, l.ffn
        if l.is_attention:
            assert mix.ds_qkv.weight.dtype == DTYPES[p.attn_qkv] and mix.ds_wo.weight.dtype == DTYPES[p.attn_out]
            assert _fid(mix.ckc) == FID[p.attn_fidelity]
            assert _fid(mix.ckc_sdpa_decode) == FID[p.sdpa_decode_fidelity]
            assert _fid(mix.ckc_sdpa_prefill) == FID[p.sdpa_prefill_fidelity]
            assert mix.k_cache.dtype == DTYPES[p.kv_cache] and mix.v_cache.dtype == DTYPES[p.kv_cache]
            part = mix.ds_wo
        else:
            assert mix.ds_in.weight.dtype == DTYPES[p.conv_in] and mix.ds_out.weight.dtype == DTYPES[p.conv_out]
            assert _fid(mix.ckc) == FID[p.conv_fidelity], (l.layer_idx, _fid(mix.ckc), p.conv_fidelity)
            part = mix.ds_out
        ccl = None if pc.plan.ccl_dtype == "bf16" else DTYPES[pc.plan.ccl_dtype]
        assert getattr(part, "partial_dtype", None) == ccl
        if hasattr(ffn, "w_up_gate"):  # MoE
            assert ffn.w_up_gate.dtype == DTYPES[p.expert_gate_up_decode or p.expert_gate_up]
            assert ffn.w_down.dtype == DTYPES[p.expert_down_decode or p.expert_down]
            assert ffn.d_gates[1].dtype == DTYPES[p.expert_gate_up] and ffn.d_downs[1].dtype == DTYPES[p.expert_down]
            assert _fid(ffn.ckc) == FID[p.expert_fidelity] and _fid(ffn.d_ckc) == FID[p.dispatch_fidelity]
            assert getattr(ffn, "partial_dtype", None) == (ccl if pc.plan.ccl_moe else None)
        else:
            w13 = ffn.ds_w13.weight if ffn.ds_w13 is not None else ffn.ds_w1.weight
            assert w13.dtype == DTYPES[p.dense_w13] and ffn.ds_w2_dec.weight.dtype == DTYPES[p.dense_w2]
            assert _fid(ffn.ckc) == FID[p.dense_fidelity]
            assert getattr(ffn.ds_w2_dec, "partial_dtype", None) == ccl
        scratch_dtypes = {str(k) for k in l.reduce.scratch}
        assert str(DTYPES[pc.plan.ccl_dtype]) in scratch_dtypes
        checked += 1
    return checked


def test_default_path_consumes_selected_config():
    env = os.environ.get("LFM2_PRECISION_CONFIG")
    expected = resolve()  # what the default path must use
    assert env or SELECTED_PATH.exists(), "no selected config and no LFM2_PRECISION_CONFIG"
    mesh = open_mesh(256 << 20)
    try:
        gen = build_generator(mesh_device=mesh, layer_indices=PROBE, max_seq_len=8192, prepare=False)
        assert check(gen, expected) == len(PROBE)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


def test_baseline_escape_hatch(monkeypatch):
    monkeypatch.setenv("LFM2_PRECISION_CONFIG", "baseline")
    base = resolve()
    assert base.config_id == "baseline"
    mesh = open_mesh(256 << 20)
    try:
        gen = build_generator(mesh_device=mesh, layer_indices=PROBE, max_seq_len=8192, prepare=False)
        assert check(gen, base) == len(PROBE)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)
