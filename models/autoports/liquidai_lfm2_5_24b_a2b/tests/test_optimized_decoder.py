# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder correctness: the functional/fused test matrix run through ``OptimizedDecoder``
with ``DEFAULT_POLICY`` + ``DEFAULT_FUSIONS`` (no fallback), plus perf / determinism /
no-host-fallback at batch 32 and batch 1.

Covers all MoE prefill paths: chunks padded to >= ``moe_dispatch_min_rows`` (96) rows use token dispatch
(prompt 65 pads to 96), 33..64-row chunks the sparse_matmul path, <= 32 rows the decode MoE path.
"""

import pytest

from ..tt.fused_decoder import DEFAULT_FUSIONS
from ..tt.model_config import TILE, Lfm2MoeArgs
from ..tt.optimized_decoder import DEFAULT_POLICY, make_optimized
from .harness import User
from .test_functional_decoder import LAYERS, _assert, hf_config, run_case, run_interleaved


def optimized_factory(policy=DEFAULT_POLICY, fusions=DEFAULT_FUSIONS):
    def factory(sd, cfg, args, layer_idx, mesh_device, max_seq_len):
        dec, rope = make_optimized(sd, cfg, args, layer_idx, mesh_device, max_seq_len, policy=policy, fusions=fusions)
        assert dec.policy == policy and dec.fusions == frozenset(fusions)
        return dec, rope

    return factory


def _policy():
    """DEFAULT_POLICY, optionally overridden for a diagnostic run via LFM2_POLICY_OVERRIDE (JSON), e.g.
    the watcher run swaps the hybrid routed-expert kernel (too large for the kernel config buffer
    once watcher-instrumented) for the unified one."""
    import json
    import os

    from loguru import logger

    from ..tt.optimized_decoder import replace

    override = os.environ.get("LFM2_POLICY_OVERRIDE", "{}")
    logger.info(f"LFM2_POLICY_OVERRIDE={override}")
    return replace(DEFAULT_POLICY, **json.loads(override))


OPT = optimized_factory(_policy())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
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
def test_optimized_single_user(mesh_device, kind, prompt_len, chunk):
    users = [User(slot=5, prompt_len=prompt_len, decode_steps=4, seed=prompt_len)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=chunk, decoder_factory=OPT))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_batch32(mesh_device, kind):
    lens = [1 + (37 * i) % 200 for i in range(TILE)]
    users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=128, decoder_factory=OPT))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_slot_reuse(mesh_device, kind):
    fg = [
        User(slot=7, prompt_len=700, decode_steps=2, seed=1),
        User(slot=7, prompt_len=33, decode_steps=3, seed=2, join_delay=2),
    ]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=256, decoder_factory=OPT))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_increasing_lengths_same_instance(mesh_device, kind):
    lens = [2047, 2048, 2049, 4100, 32, 33]
    fg = [User(slot=9, prompt_len=n, decode_steps=2, seed=500 + n, join_delay=i % 3) for i, n in enumerate(lens)]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=2048, decoder_factory=OPT))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_full_context(mesh_device, kind):
    ctx = Lfm2MoeArgs.from_hf_config(hf_config()).max_position_embeddings
    users = [
        User(slot=9, prompt_len=ctx - 3, decode_steps=3, seed=7),
        User(slot=30, prompt_len=77, decode_steps=1, seed=8),
    ]
    _assert(
        run_case(
            mesh_device, LAYERS[kind], users, chunk_size=2048, max_seq_len=ctx, reserve_seqs=TILE, decoder_factory=OPT
        )
    )


# Bucketed decode (``active_slots`` = b, one trace per bucket): users in slots 0..b-1 only; an idle
# slot inside the bucket (bucket 4, slot 2) and the largest prompts exercise the SDPA row slicing.
BUCKET_CASES = {
    "b1_short": (1, [(0, 65)]),
    "b1_long": (1, [(0, 2049)]),
    "b2": (2, [(0, 300), (1, 1)]),
    "b4_gap": (4, [(0, 31), (1, 2049), (3, 300)]),
    "b8": (8, [(s, 1 + 97 * s) for s in range(8)]),
    # slot 13 (conv_moe) hits a router near-tie, accepted via the logged near-tie alternative;
    # slot 7 of 1 + 61 * s (prompt 428, seed 1435) is the explicit control test_router_near_tie_control
    "b16": (16, [(s, 1 + 59 * s) for s in range(16)]),
}


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("case", list(BUCKET_CASES))
def test_optimized_bucketed_decode(mesh_device, kind, case):
    bucket, spec = BUCKET_CASES[case]
    users = [User(slot=s, prompt_len=n, decode_steps=4, seed=1000 + s + n) for s, n in spec]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=2048, decoder_factory=OPT, active_slots=bucket))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("variant", ["functional", "optimized"])
def test_router_near_tie_control(mesh_device, variant):
    """attn_moe, prompt 428 (seed 1435): at the first decode token (position 428) the HF fp32 router
    scores of the 4th/5th experts (44 / 19) differ by 7.3e-4; TT (stage-1 functional and the
    optimized decoder alike) selects 19, giving decode update PCC 0.989 vs plain HF.  Against HF with
    exactly that swap forced the update PCC is >= 0.99: the deviation is one top-k flip, not drift."""
    from functools import partial

    from ..reference.hf_layer import force_expert_swap

    users = [User(slot=7, prompt_len=428, decode_steps=4, seed=1435)]
    factory = None if variant == "functional" else OPT
    _assert(
        run_case(
            mesh_device,
            LAYERS["attn_moe"],
            users,
            chunk_size=2048,
            decoder_factory=factory,
            hf_patch=partial(force_expert_swap, pos=428),
        )
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_bucketed_full_context(mesh_device, kind):
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
            decoder_factory=OPT,
            active_slots=1,
        )
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("active, bucket", [(32, None), (1, None), (1, 1), (4, 4)])
def test_optimized_perf_determinism_no_fallback(mesh_device, kind, active, bucket):
    from .test_perf_determinism import run_perf_determinism

    run_perf_determinism(mesh_device, kind, decoder_factory=OPT, active_users=active, active_slots=bucket)


# Deep layers on REAL hidden states (tests/gen_real_hidden.py: HF layers 0..L-1 over book text).
# Embedding-row inputs are out of distribution there (stage-1 functional decoder: layer-39 prefill
# output PCC 0.958 on them, doc/optimized_decoder/work_log.md round 2).
DEEP_LAYERS = [1, 6, 7, 15, 23, 31, 38, 39]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer", DEEP_LAYERS)
@pytest.mark.parametrize("variant", ["optimized", "functional"])
def test_optimized_deep_layers_real_hidden(mesh_device, layer, variant):
    import hashlib
    import os

    import torch
    from loguru import logger

    from . import gen_real_hidden

    path = str(gen_real_hidden.OUT)
    if not os.path.exists(path):  # deterministic from the in-repo corpus + pinned weights (CPU, ~2 min)
        gen_real_hidden.main()
    with open(path, "rb") as f:
        logger.info(f"real hidden states {path} sha256 {hashlib.sha256(f.read()).hexdigest()[:16]}")
    hidden = torch.load(path)["hidden"][layer]  # [P, S, H]
    users = [User(slot=4 * p + 1, prompt_len=n, decode_steps=4, seed=p) for p, n in enumerate((396, 300, 129, 61))]
    factory = OPT if variant == "optimized" else None
    res = run_case(
        mesh_device,
        layer,
        users,
        chunk_size=128,
        decoder_factory=factory,
        user_inputs=lambda u: hidden[u.seed, : u.prompt_len + u.decode_steps],
    )
    _assert(res)


# ---------------- several decode traces (one per bucket) on one decoder instance ----------------
MULTI_BUCKETS = (1, 4, TILE)


def _bucket_for(active):
    """Smallest bucket covering the highest active slot (new users take the lowest free slot, so no
    per-slot state ever moves; holes below the highest active slot are allowed)."""
    from ..tt.optimized_decoder import decode_bucket

    return decode_bucket([u.slot for u in active], MULTI_BUCKETS) or TILE


def run_multi_bucket(mesh_device, layer_idx, chunk_size=128):
    """One decoder, one trace per bucket in MULTI_BUCKETS sharing the persistent decode inputs, KV cache
    and ShortConv state.  Every bucket's decode and every prefill shape are compiled before the first
    capture; the schedule then alternates buckets as users join (prefill between replays) and leave,
    including a hole (slot 0 freed while slots 1-2 decode, then reused by a new request)."""
    import torch
    from loguru import logger
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    from ..reference.hf_layer import build_hf_layer, hf_layer_forward, near_tie_alternatives
    from .harness import (
        DecodeInputs,
        PagedLayout,
        decode_step,
        from_dev,
        hf_config,
        layer_inputs,
        layer_state_dict,
        run_prefill,
    )
    from .test_functional_decoder import (
        SCRATCH_SLOT,
        _real_embed,
        _restore_state,
        _snapshot_state,
        _use_real,
        delta_pccs,
        make_decoder,
        pcc,
    )

    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    H = args.hidden_size
    sd = layer_state_dict(layer_idx, _use_real())
    embed = _real_embed()
    # (prompt_len, decode_steps): steps follow the schedule below
    U = [
        User(slot=0, prompt_len=300, decode_steps=9, seed=11),
        User(slot=1, prompt_len=65, decode_steps=10, seed=12),
        User(slot=2, prompt_len=129, decode_steps=9, seed=13),
    ]
    U += [User(slot=s, prompt_len=20 + 37 * s, decode_steps=2, seed=20 + s) for s in range(3, 10)]
    U += [User(slot=0, prompt_len=97, decode_steps=4, seed=31)]  # reuses slot 0 after U[0] leaves
    late = U[-1]
    scratch = User(slot=SCRATCH_SLOT, prompt_len=max(u.prompt_len for u in U), decode_steps=0, seed=0)
    max_seq = max(u.prompt_len + u.decode_steps for u in U)

    hf_layer, hf_rope = build_hf_layer(cfg, layer_idx, sd)
    for u in U:
        u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
        u.ref = hf_layer_forward(hf_layer, hf_rope, u.x.unsqueeze(0))[0]
        u.ref_alts = near_tie_alternatives(hf_layer, hf_rope, u.x, range(u.prompt_len, u.prompt_len + u.decode_steps))
    del hf_layer

    decoder, rope = make_decoder(OPT, sd, cfg, args, layer_idx, mesh_device, max_seq)
    layout = PagedLayout(U + [scratch], max_seq)
    if decoder.is_attention:
        decoder.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
    for u in U:  # compile every prefill shape/offset before any capture (scratch slot + pages)
        tmp = User(slot=SCRATCH_SLOT, prompt_len=u.prompt_len, decode_steps=0, seed=0, x=u.x, blocks=scratch.blocks)
        run_prefill(decoder, rope, mesh_device, layout, tmp, chunk_size)

    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq)
    garbage = torch.Generator().manual_seed(7)
    active, step = [], {id(u): 0 for u in U}

    def load_inputs(b):
        inp.active_slots = None if b == TILE else b
        x = torch.randn(1, 1, TILE, H, generator=garbage) * 0.5
        cur = torch.full((TILE,), -1, dtype=torch.int32)
        for u in active:
            x[0, 0, u.slot] = u.x[u.prompt_len + step[id(u)]]
            cur[u.slot] = u.prompt_len + step[id(u)]
        inp.update(x, cur, layout.decode_table(active))

    # warm every bucket, then capture every bucket (state restored after each)
    snapshot = _snapshot_state(decoder)
    for b in MULTI_BUCKETS:
        load_inputs(b)
        ttnn.deallocate(decode_step(decoder, rope, inp))
        _restore_state(decoder, snapshot)
    ttnn.synchronize_device(mesh_device)
    traces = {}
    for b in MULTI_BUCKETS:
        load_inputs(b)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = decode_step(decoder, rope, inp)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        # Lifetime invariant (reviewed): a bucket's trace output is read right after that bucket's
        # replay and before any other replay or prefill; it was allocated after every persistent
        # buffer (KV cache, conv state, decode inputs, weights), so only the OTHER traces'
        # intermediates (whose addresses were free at their capture) can overwrite it, and only while
        # it is dead.  Acknowledge exactly these output tensors.
        trace_allocation_tracker.acknowledge_corruptible(out)
        traces[b] = (tid, out)
        _restore_state(decoder, snapshot)
    del snapshot
    counts = {b: 0 for b in MULTI_BUCKETS}

    def replay(n=1):
        for _ in range(n):
            b = _bucket_for(active)
            load_inputs(b)
            tid, out = traces[b]
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            host = from_dev(out, mesh_device)[0, 0]
            for u in list(active):
                u.tt_decode.append(host[u.slot])
                step[id(u)] += 1
                if step[id(u)] == u.decode_steps:
                    active.remove(u)
            counts[b] += 1

    def join(u):
        u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, chunk_size)
        active.append(u)

    join(U[0])
    replay(2)  # bucket 1
    join(U[1])
    replay(1)  # bucket 4 (prefill between replays)
    join(U[2])
    replay(2)  # bucket 4
    for u in U[3:10]:
        join(u)
    replay(2)  # bucket 32; U[3..9] leave
    replay(2)  # bucket 4; U[0] leaves -> hole at slot 0
    replay(1)  # bucket 4 with slot 0 idle
    join(late)
    replay(2)  # bucket 4, slot 0 reused; U[1], U[2] leave
    replay(2)  # bucket 1, late user alone
    assert not active, [u.slot for u in active]
    for b, (tid, _) in traces.items():
        ttnn.release_trace(mesh_device, tid)
    logger.info(f"multi-bucket: trace ids {[(b, t[0]) for b, t in traces.items()]}, replays per bucket {counts}")
    assert all(counts[b] > 0 for b in MULTI_BUCKETS), counts

    results = []
    for u in U:
        p = pcc(u.tt_prefill, u.ref[: u.prompt_len])
        d = pcc(torch.stack(u.tt_decode), u.ref[u.prompt_len : u.prompt_len + u.decode_steps])
        dp, dd = delta_pccs(u)
        logger.info(
            f"multi-bucket layer {layer_idx} slot {u.slot} prompt {u.prompt_len} decode {u.decode_steps}: "
            f"prefill PCC {p:.6f} decode PCC {d:.6f} delta PCC {dp:.6f}/{dd:.6f}"
        )
        results.append((u, p, d))
    return results


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_optimized_multi_bucket_traces(mesh_device, kind):
    _assert(run_multi_bucket(mesh_device, LAYERS[kind]))
