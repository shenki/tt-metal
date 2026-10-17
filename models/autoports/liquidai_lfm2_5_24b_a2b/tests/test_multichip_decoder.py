# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multichip-decoder correctness on the 1x4 Blackhole ring (TP = 4): the stage-3 test matrix vs HF
through ``MultichipDecoder`` (``MC_POLICY`` + ``DEFAULT_MESH_PLAN``)."""

import json
import os

import pytest
from loguru import logger

from ..tt.fused_decoder import DEFAULT_FUSIONS
from ..tt.model_config import TILE, Lfm2MoeArgs
from ..tt.multichip_decoder import DEFAULT_MESH_PLAN, MC_POLICY, make_multichip, mesh_device_params, replace
from .harness import User
from .test_functional_decoder import LAYERS, _assert, hf_config, run_case, run_interleaved

MESH = (1, 4)
DEVICE_PARAMS = mesh_device_params()


def multichip_factory(policy=MC_POLICY, fusions=DEFAULT_FUSIONS, plan=DEFAULT_MESH_PLAN):
    def factory(sd, cfg, args, layer_idx, mesh_device, max_seq_len):
        dec, rope = make_multichip(
            sd, cfg, args, layer_idx, mesh_device, max_seq_len, policy=policy, fusions=fusions, plan=plan
        )
        return dec, rope

    return factory


def _policy():
    """MC_POLICY, optionally overridden for a diagnostic run via LFM2_POLICY_OVERRIDE (JSON)."""
    override = os.environ.get("LFM2_POLICY_OVERRIDE", "{}")
    logger.info(f"LFM2_POLICY_OVERRIDE={override}")
    return replace(MC_POLICY, **json.loads(override))


def _plan():
    """DEFAULT_MESH_PLAN, optionally overridden via LFM2_PLAN_OVERRIDE (JSON; lists become tuples)."""
    override = os.environ.get("LFM2_PLAN_OVERRIDE", "{}")
    logger.info(f"LFM2_PLAN_OVERRIDE={override}")
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(override).items()}
    return replace(DEFAULT_MESH_PLAN, **kw)


MC = multichip_factory(_policy(), plan=_plan())

# Router near-tie window for the multichip tests.  Decode router scores of BOTH TT decoders differ from HF's fp32
# scores by more than the stage-3 window (1e-3): single chip up to 4.9e-3, multichip attention layer up to 6.9e-3
# (repro/router_score_noise.py -> logs/router_score_noise_L*.log); a top-k flip at a gap below |e_k| + |e_k+1| is
# therefore numerics, not a bug.  Acceptance stays strict: ONE swap at that position must explain the decode
# update at PCC >= 0.99 (test_functional_decoder.delta_pccs), and every acceptance is logged.
MC_ROUTER_TIE_TAU = 1e-2


@pytest.fixture(autouse=True)
def _multichip_router_tie_window():
    from ..reference import hf_layer

    old = hf_layer.ROUTER_TIE_TAU
    hf_layer.ROUTER_TIE_TAU = MC_ROUTER_TIE_TAU
    yield
    hf_layer.ROUTER_TIE_TAU = old


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize(
    "prompt_len, chunk",
    [
        (1, 2048),
        (31, 2048),
        (32, 2048),
        (64, 2048),
        (65, 2048),
        (96, 2048),
        (97, 2048),
        (300, 128),
        (300, 2048),
        (2049, 2048),
    ],
)
def test_multichip_single_user(mesh_device, kind, prompt_len, chunk):
    users = [User(slot=5, prompt_len=prompt_len, decode_steps=4, seed=prompt_len)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=chunk, decoder_factory=MC))


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("active, bucket", [(32, None), (1, None), (1, 1), (4, 4)])
def test_multichip_perf_determinism_no_fallback(mesh_device, kind, active, bucket):
    from .test_perf_determinism import run_perf_determinism

    # LFM2_MC_DECODE_ITERS: replays in the timed window (default 5 = fits the device-profiler buffer;
    # 50 for wall-clock tables)
    iters = int(os.environ.get("LFM2_MC_DECODE_ITERS", "5"))
    r = run_perf_determinism(
        mesh_device, kind, decoder_factory=MC, active_users=active, active_slots=bucket, decode_iters=iters
    )
    logger.info(f"PERF {kind} active {active} bucket {bucket}: {r}")


# ------------------------------------------------------------------------------------------------
# multichip vs the single-chip TTNN baseline: the stage-3 OptimizedDecoder with every tensor replicated on
# the same 1x4 mesh (each chip computes the whole single-chip layer, no collectives; a 1x1 submesh of the
# open mesh hung the next all-reduce: per-device allocator state diverges, doc/multichip_decoder/triage/):
# identical weights, inputs, page tables, positions; isolates sharding / collective bugs from
# HF-vs-TTNN numerics.  Also checks the per-chip KV-cache heads / ShortConv state slices and that the
# replicated residual is bitwise identical on all chips.
# ------------------------------------------------------------------------------------------------
MC_VS_SC_OUT_BAR = 0.999  # output PCC multichip vs single-chip
MC_VS_SC_UPD_BAR = 0.995  # update (out - x) PCC multichip vs single-chip


def _sc_baseline():
    """Stage-3 DEFAULT_POLICY with the multichip expert-FFN op choice (unified, HiFi2): the hybrid op cannot
    run next to the CCL L1_SMALL region (work_log M-3), and the same op on both sides keeps the comparison
    about sharding / collectives."""
    from ..tt.optimized_decoder import DEFAULT_POLICY

    return replace(DEFAULT_POLICY, dispatch_ffn=MC_POLICY.dispatch_ffn, dispatch_fidelity=MC_POLICY.dispatch_fidelity)


SC_BASELINE = _sc_baseline()


def _all_devices(t):
    import ttnn

    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def run_mc_vs_single_chip(mesh_device, layer_idx, users, *, chunk_size, active_slots=None):
    import torch

    import ttnn

    from ..tt.optimized_decoder import make_optimized
    from .harness import DecodeInputs, PagedLayout, decode_step, layer_inputs, layer_state_dict, pcc, run_prefill
    from .test_functional_decoder import _real_embed, _use_real

    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    H = args.hidden_size
    sd = layer_state_dict(layer_idx, _use_real())
    embed = _real_embed()
    for u in users:
        u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
    max_seq = max(max(u.prompt_len + u.decode_steps for u in users), TILE)
    runs = {}
    for name, md, make in [
        (
            "single",
            mesh_device,
            lambda: make_optimized(sd, cfg, args, layer_idx, mesh_device, max_seq, policy=SC_BASELINE),
        ),
        (
            "multi",
            mesh_device,
            lambda: make_multichip(sd, cfg, args, layer_idx, mesh_device, max_seq, policy=_policy(), plan=_plan()),
        ),
    ]:
        dec, rope = make()
        layout = PagedLayout(users, max_seq)
        if dec.is_attention:
            dec.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
        pre = {u.slot: run_prefill(dec, rope, md, layout, u, chunk_size) for u in users}
        inp = DecodeInputs(
            md,
            H,
            layout.max_blocks_per_seq,
            active_slots=active_slots,
            slot_mask_width=getattr(dec, "slot_mask_width", None),
        )
        table = layout.decode_table(users)
        garbage = torch.Generator().manual_seed(77)
        dec_rows = {u.slot: [] for u in users}
        replicas_equal = True
        for t in range(max(u.decode_steps for u in users)):
            x = torch.randn(1, 1, TILE, H, generator=garbage) * 0.5
            cur = torch.full((TILE,), -1, dtype=torch.int32)
            for u in users:
                if t < u.decode_steps:
                    x[0, 0, u.slot] = u.x[u.prompt_len + t]
                    cur[u.slot] = u.prompt_len + t
            inp.update(x, cur, table)
            out = decode_step(dec, rope, inp)
            devs = _all_devices(out)
            replicas_equal &= all(torch.equal(devs[0], d) for d in devs[1:])
            for u in users:
                if t < u.decode_steps:
                    dec_rows[u.slot].append(devs[0][0, 0, u.slot])
            ttnn.deallocate(out)
        m = dec.mixer
        state = (
            [_all_devices(m.k_cache), _all_devices(m.v_cache)]
            if dec.is_attention
            else [_all_devices(s) for s in m.state]
        )
        if name == "single":
            state = [devs[:1] for devs in state]  # replicated: every chip holds the full single-chip state
        runs[name] = dict(pre=pre, dec=dec_rows, state=state, replicas_equal=replicas_equal, attn=dec.is_attention)
        del dec, rope, inp
    sc, mc = runs["single"], runs["multi"]
    assert mc["replicas_equal"], "multichip decode output differs between chips (replicated residual contract)"
    tp = mesh_device.get_num_devices()
    results = []
    for u in users:
        n = u.prompt_len
        xp, xd = u.x[:n], u.x[n : n + u.decode_steps]
        pp = pcc(mc["pre"][u.slot], sc["pre"][u.slot])
        pu = pcc(mc["pre"][u.slot] - xp, sc["pre"][u.slot] - xp)
        dm, ds = torch.stack(mc["dec"][u.slot]), torch.stack(sc["dec"][u.slot])
        dp, du = pcc(dm, ds), pcc(dm - xd, ds - xd)
        logger.info(
            f"MC-vs-SC layer {layer_idx} slot {u.slot} prompt {n}: prefill PCC {pp:.6f} upd {pu:.6f} | "
            f"decode PCC {dp:.6f} upd {du:.6f}"
        )
        results.append((pp, pu, dp, du))
    # per-chip state = the single-chip state's slice: KV heads 2d, 2d+1 (dim 1) / ShortConv channels (last dim)
    for i, (s_full, m_devs) in enumerate(zip(sc["state"], mc["state"])):
        s_full = s_full[0]
        for d, m_d in enumerate(m_devs):
            if mc["attn"]:
                w = s_full.shape[1] // tp
                ref = s_full[:, d * w : (d + 1) * w]
            else:
                w = s_full.shape[-1] // tp
                ref = s_full[..., d * w : (d + 1) * w]
            p = pcc(m_d, ref) if ref.abs().max() > 0 else float(torch.equal(m_d, ref))
            logger.info(f"MC-vs-SC layer {layer_idx} state {i} chip {d}: shape {tuple(m_d.shape)} PCC {p:.6f}")
            assert m_d.shape == ref.shape and p >= MC_VS_SC_OUT_BAR, (i, d, tuple(m_d.shape), tuple(ref.shape), p)
    for pp, pu, dp, du in results:
        assert min(pp, dp) >= MC_VS_SC_OUT_BAR and min(pu, du) >= MC_VS_SC_UPD_BAR, (pp, pu, dp, du)
    return results


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("bucket", [None, 1])
def test_multichip_vs_single_chip(mesh_device, kind, bucket):
    """Several users (non-aligned prompts, multi-chunk, dispatch / sparse / decode-shaped MoE prefill
    chunks) vs the stage-3 single-chip decoder; bucket 1: one user in slot 0."""
    if bucket == 1:
        users = [User(slot=0, prompt_len=300, decode_steps=4, seed=3)]
    else:
        users = [
            User(slot=1, prompt_len=1, decode_steps=4, seed=11),
            User(slot=6, prompt_len=65, decode_steps=4, seed=12),
            User(slot=17, prompt_len=300, decode_steps=3, seed=13),
            User(slot=31, prompt_len=2049, decode_steps=3, seed=14),
        ]
    run_mc_vs_single_chip(mesh_device, LAYERS[kind], users, chunk_size=2048, active_slots=bucket)


# ------------------------------------------------------------------------------------------------
# the stage-3 test matrix on the mesh (vs HF)
# ------------------------------------------------------------------------------------------------
def _mesh_test(f):
    f = pytest.mark.parametrize("mesh_device", [MESH], indirect=True)(f)
    return pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)(f)


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_batch32(mesh_device, kind):
    lens = [1 + (37 * i) % 200 for i in range(TILE)]
    users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=128, decoder_factory=MC))


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_slot_reuse(mesh_device, kind):
    fg = [
        User(slot=7, prompt_len=700, decode_steps=2, seed=1),
        User(slot=7, prompt_len=33, decode_steps=3, seed=2, join_delay=2),
    ]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=256, decoder_factory=MC))


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_increasing_lengths_same_instance(mesh_device, kind):
    lens = [2047, 2048, 2049, 4100, 32, 33]
    fg = [User(slot=9, prompt_len=n, decode_steps=2, seed=500 + n, join_delay=i % 3) for i, n in enumerate(lens)]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=2048, decoder_factory=MC))


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_full_context(mesh_device, kind):
    """131069-token prompt + decode to position 131071 next to a short user; the KV cache reserves pages
    for 32 users x 131072 tokens (2 local KV heads per chip)."""
    ctx = Lfm2MoeArgs.from_hf_config(hf_config()).max_position_embeddings
    users = [
        User(slot=9, prompt_len=ctx - 3, decode_steps=3, seed=7),
        User(slot=30, prompt_len=77, decode_steps=1, seed=8),
    ]
    _assert(
        run_case(
            mesh_device, LAYERS[kind], users, chunk_size=2048, max_seq_len=ctx, reserve_seqs=TILE, decoder_factory=MC
        )
    )


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_bucketed_full_context(mesh_device, kind):
    ctx = Lfm2MoeArgs.from_hf_config(hf_config()).max_position_embeddings
    users = [User(slot=0, prompt_len=ctx - 3, decode_steps=3, seed=7)]
    _assert(
        run_case(
            mesh_device,
            LAYERS[kind],
            users,
            chunk_size=2048,
            max_seq_len=ctx,
            reserve_seqs=TILE,
            decoder_factory=MC,
            active_slots=1,
        )
    )


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("case", ["b1_short", "b1_long", "b2", "b4_gap", "b8", "b16"])
def test_multichip_bucketed_decode(mesh_device, kind, case):
    from .test_optimized_decoder import BUCKET_CASES

    bucket, spec = BUCKET_CASES[case]
    users = [User(slot=s, prompt_len=n, decode_steps=4, seed=1000 + s + n) for s, n in spec]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=2048, decoder_factory=MC, active_slots=bucket))


@_mesh_test
@pytest.mark.parametrize("kind", list(LAYERS))
def test_multichip_multi_bucket_traces(mesh_device, kind):
    """Traces for buckets 1 / 4 / 32 on one decoder (shared inputs, KV cache, conv state, CCL semaphores),
    prefills between replays, users joining / leaving, a hole and slot reuse."""
    from .test_optimized_decoder import run_multi_bucket

    _assert(run_multi_bucket(mesh_device, LAYERS[kind], factory=MC))


@_mesh_test
def test_multichip_router_near_tie_control(mesh_device):
    """The stage-3 near-tie control input (attn_moe, prompt 428, seed 1435; HF's 4th/5th biased scores at the first
    decode position differ by 7.3e-4).  The single-chip decoder lands on the swapped side of that tie; the
    multichip decoder lands on HF's own choice (log: update vs plain HF >= 0.99, vs the swap 0.989).  Either side is
    legitimate; the check is that the decode update is explained by plain HF or by exactly that one swap."""

    import torch

    from ..reference.hf_layer import build_hf_layer, force_expert_swap, hf_layer_forward
    from .harness import layer_state_dict, pcc
    from .test_functional_decoder import DELTA_PCC_BAR, _use_real

    users = [User(slot=7, prompt_len=428, decode_steps=4, seed=1435)]
    res = run_case(mesh_device, LAYERS["attn_moe"], users, chunk_size=2048, decoder_factory=MC)
    u = res[0][0]
    n, m = u.prompt_len, len(u.tt_decode)
    upd = torch.stack(u.tt_decode) - u.x[n : n + m]
    plain = pcc(upd, u.ref[n : n + m] - u.x[n : n + m])
    hl, rope = build_hf_layer(hf_config(), LAYERS["attn_moe"], layer_state_dict(LAYERS["attn_moe"], _use_real()))
    with torch.no_grad():
        swapped = hf_layer_forward(force_expert_swap(hl, 428), rope, u.x.unsqueeze(0))[0]
    swap = pcc(upd, swapped[n : n + m] - u.x[n : n + m])
    logger.info(f"near-tie control: decode update PCC vs plain HF {plain:.6f}, vs HF with the 428 swap {swap:.6f}")
    assert max(plain, swap) >= DELTA_PCC_BAR, (plain, swap)
    _assert(res)


@_mesh_test
@pytest.mark.parametrize("layer", [1, 6, 7, 15, 23, 31, 38, 39])
def test_multichip_deep_layers_real_hidden(mesh_device, layer):
    """Deep layers on real hidden states (tests/gen_real_hidden.py), as the stage-3 test."""
    import hashlib

    import torch

    from . import gen_real_hidden

    path = str(gen_real_hidden.OUT)
    if not os.path.exists(path):
        gen_real_hidden.main()
    with open(path, "rb") as f:
        logger.info(f"real hidden states {path} sha256 {hashlib.sha256(f.read()).hexdigest()[:16]}")
    hidden = torch.load(path)["hidden"][layer]
    users = [User(slot=4 * p + 1, prompt_len=n, decode_steps=4, seed=p) for p, n in enumerate((396, 300, 129, 61))]
    _assert(
        run_case(
            mesh_device,
            layer,
            users,
            chunk_size=128,
            decoder_factory=MC,
            user_inputs=lambda u: hidden[u.seed, : u.prompt_len + u.decode_steps],
        )
    )


@_mesh_test
def test_multichip_memory_capacity(mesh_device):
    """Measured per-chip DRAM of one decoder layer of each kind (weights + constants) and of one attention
    layer's KV cache for 32 users x 131072 tokens; extrapolated to the 40-layer stack
    (doc/context_contract.json "multichip_decoder")."""
    import json

    import ttnn

    from .harness import PagedLayout, layer_state_dict
    from .test_functional_decoder import _use_real

    def dram_used():
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return v.num_banks * v.total_bytes_allocated_per_bank, v.num_banks * v.total_bytes_per_bank

    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    ctx = args.max_position_embeddings
    rec = {}
    keep = []
    base, total = dram_used()
    rec["dram_bytes_per_chip"] = total
    for kind, layer in LAYERS.items():
        before, _ = dram_used()
        sd = layer_state_dict(layer, _use_real())
        dec, rope = make_multichip(sd, cfg, args, layer, mesh_device, ctx)
        del sd
        after, _ = dram_used()
        rec[f"{kind}_layer_bytes"] = after - before
        if dec.is_attention:
            users = [User(slot=0, prompt_len=1, decode_steps=0, seed=0)]
            layout = PagedLayout(users, ctx, reserve_seqs=TILE)
            dec.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
            rec["kv_cache_bytes_per_attention_layer_32x131072"] = dram_used()[0] - after
            rec["kv_blocks"] = layout.num_blocks
            rec["rope_tables_included_in_attn_layer"] = True
        keep.append((dec, rope))
    n_attn = sum(args.is_attention_layer(i) for i in range(args.num_layers))
    n_dense = args.num_dense_layers
    n_conv_moe = args.num_layers - n_attn - n_dense
    stack = (
        n_dense * rec["conv_dense_layer_bytes"]
        + n_conv_moe * rec["conv_moe_layer_bytes"]
        + n_attn * (rec["attn_moe_layer_bytes"] + rec["kv_cache_bytes_per_attention_layer_32x131072"])
    )
    rec["layer_counts"] = {"conv_dense": n_dense, "conv_moe": n_conv_moe, "attn_moe": n_attn}
    rec["stack_bytes_per_chip_32x131072"] = stack
    rec["stack_fraction_of_dram"] = stack / total
    rec["baseline_allocated_before_layers"] = base
    for k, v in rec.items():
        logger.info(f"MEMORY {k}: {v / 2**30:.3f} GiB" if isinstance(v, int) and v > 2**20 else f"MEMORY {k}: {v}")
    out = os.path.join(os.path.dirname(__file__), "..", "doc", "multichip_decoder", "sweeps", "memory_capacity.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    assert stack < total, rec


# ------------------------------------------------------------------------------------------------
# stacked-decoder contract: real consecutive layers 1 (conv_dense) -> 2 (attn_moe) -> 3 (conv_moe), one decode
# trace over the chain with the residual width-sharded on resid_mem between layers (converted once at entry)
# ------------------------------------------------------------------------------------------------
STACK_LAYERS = (1, 2, 3)


@_mesh_test
def test_multichip_stacked_layers(mesh_device):
    import torch

    import ttnn

    from ..reference.hf_layer import build_hf_layer, hf_layer_forward
    from .harness import DecodeInputs, PagedLayout, from_dev, layer_inputs, layer_state_dict, pcc, run_prefill
    from .test_functional_decoder import DELTA_PCC_BAR, PCC_BAR, _real_embed, _restore_state, _snapshot_state, _use_real

    logger.info(
        f"TT_METAL_TRACE_ALLOC_TRACKING={os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')} "
        f"TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE={os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE')}"
    )
    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    H = args.hidden_size
    real = _use_real()
    embed = _real_embed()
    users = [
        User(slot=3, prompt_len=65, decode_steps=4, seed=31),
        User(slot=17, prompt_len=300, decode_steps=4, seed=32),
    ]
    max_seq = max(u.prompt_len + u.decode_steps for u in users)
    sds = {i: layer_state_dict(i, real) for i in STACK_LAYERS}
    # HF reference: the chained layers over each user's full sequence
    for u in users:
        u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
        h = u.x.unsqueeze(0)
        for i in STACK_LAYERS:
            hl, hr = build_hf_layer(cfg, i, sds[i])
            with torch.no_grad():
                h = hf_layer_forward(hl, hr, h).to(torch.bfloat16).float()
        u.ref = h[0]
    decs = [
        make_multichip(sds[i], cfg, args, i, mesh_device, max_seq, policy=_policy(), plan=_plan()) for i in STACK_LAYERS
    ]
    layout = PagedLayout(users, max_seq)
    for dec, _ in decs:
        if dec.is_attention:
            dec.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
    # chained prefill: layer i's output rows are layer i+1's input rows
    for u in users:
        cur = u.x
        for dec, rope in decs:
            tmp = User(slot=u.slot, prompt_len=u.prompt_len, decode_steps=0, seed=0, x=cur, blocks=u.blocks)
            out = run_prefill(dec, rope, mesh_device, layout, tmp, 2048)
            cur = torch.cat([out, torch.zeros(u.decode_steps, H)])
        u.tt_prefill = out
    resid_mem = decs[0][0].resid_mem
    assert all(d.resid_mem == resid_mem for d, _ in decs)
    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq, slot_mask_width=decs[0][0].slot_mask_width)
    x_res = ttnn.to_memory_config(inp.x, resid_mem)  # model entry: the only conversion

    def step(x, check_layout=False):
        cos = sin = None
        for dec, rope in decs:
            kw = dict(slot_active=inp.slot_active, slot_mask=inp.slot_mask)
            if dec.is_attention:
                cos, sin = rope.decode_cos_sin(inp.rope_pos)
                kw.update(rope=rope, rope_cos=cos, rope_sin=sin, cur_pos=inp.cur_pos, page_table=inp.page_table)
            y = dec.decode_forward(x, **kw)
            if check_layout:
                assert y.memory_config() == resid_mem, (dec.kind, y.memory_config())
            if x is not x_res and x is not inp.x:
                ttnn.deallocate(x)
            x = y
        return x

    table = layout.decode_table(users)

    def load(t):
        x = torch.zeros(1, 1, TILE, H)
        cur = torch.full((TILE,), -1, dtype=torch.int32)
        for u in users:
            x[0, 0, u.slot] = u.x[u.prompt_len + t]
            cur[u.slot] = u.prompt_len + t
        inp.update(x, cur, table)
        ttnn.copy(ttnn.to_memory_config(inp.x, resid_mem), x_res)

    load(0)
    snaps = [_snapshot_state(d) for d, _ in decs]
    # eager: sharded entry vs DRAM-interleaved entry (one conversion inside layer 1) must agree bitwise
    o_sh = from_dev(step(x_res, check_layout=True), mesh_device)
    for d, s in zip(decs, snaps):
        _restore_state(d, s)
    o_dram = from_dev(step(inp.x), mesh_device)
    for d, s in zip(decs, snaps):
        _restore_state(d, s)
    o_sh2 = from_dev(step(x_res), mesh_device)
    for d, s in zip(decs, snaps):
        _restore_state(d, s)
    rows = [u.slot for u in users]
    logger.info(
        f"stacked layers entry check (active rows): sharded vs sharded equal {torch.equal(o_sh[..., rows, :], o_sh2[..., rows, :])}, "
        f"sharded vs DRAM max|diff| {(o_sh - o_dram)[..., rows, :].abs().max().item():.3e} "
        f"(all rows {(o_sh - o_dram).abs().max().item():.3e})"
    )
    assert torch.equal(o_sh[..., rows, :], o_dram[..., rows, :]), "sharded-residual entry differs from DRAM entry"
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    t_out = step(x_res)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for d, s in zip(decs, snaps):
        _restore_state(d, s)
    del snaps
    for t in range(max(u.decode_steps for u in users)):
        load(t)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        host = from_dev(t_out, mesh_device)[0, 0]
        for u in users:
            u.tt_decode.append(host[u.slot])
    ttnn.release_trace(mesh_device, tid)
    for u in users:
        n = u.prompt_len
        p = pcc(u.tt_prefill, u.ref[:n])
        d = pcc(torch.stack(u.tt_decode), u.ref[n : n + u.decode_steps])
        dp = pcc(u.tt_prefill - u.x[:n], u.ref[:n] - u.x[:n])
        dd = pcc(
            torch.stack(u.tt_decode) - u.x[n : n + u.decode_steps],
            u.ref[n : n + u.decode_steps] - u.x[n : n + u.decode_steps],
        )
        logger.info(
            f"stacked layers {STACK_LAYERS} slot {u.slot} prompt {n}: prefill PCC {p:.6f} decode PCC {d:.6f} "
            f"stack-update PCC {dp:.6f}/{dd:.6f}"
        )
        assert min(p, d) >= PCC_BAR and (not real or min(dp, dd) >= DELTA_PCC_BAR), (p, d, dp, dd)
    logger.info(f"stacked layers: every layer output on resid_mem {resid_mem}; sharded entry == DRAM entry bitwise")
