# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder correctness: HF ``Lfm2MoeDecoderLayer`` vs TTNN ``FunctionalDecoder``.

Layer kinds (real config shapes): 0 = ShortConv + dense MLP, 3 = ShortConv + MoE,
2 = attention + MoE.  Every case runs chunked paged prefill for one or more users in distinct
slots (random block ownership), then traced batch-32 decode steps for all of them at once, and
compares each user's prefill rows and decode rows with one causal full-sequence HF forward.

Real weights are used when the pinned checkpoint is fully downloaded (``LFM2_REAL_WEIGHTS=1``
forces it); otherwise deterministic synthetic weights from ``weight_stats.json``.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn

from ..reference.hf_layer import build_hf_layer, hf_layer_forward
from ..tt.functional_decoder import FunctionalDecoder, RopeSetup
from ..tt.model_config import TILE, Lfm2MoeArgs
from .harness import (
    DecodeInputs,
    PagedLayout,
    User,
    decode_step,
    from_dev,
    hf_config,
    layer_inputs,
    layer_state_dict,
    local_model_path,
    pcc,
    run_prefill,
)

PCC_BAR = 0.995
DELTA_PCC_BAR = 0.99  # PCC of the layer update out - x (see delta_pccs); set from the functional baseline
LAYERS = {"conv_dense": 0, "conv_moe": 3, "attn_moe": 2}


def _use_real():
    flag = os.environ.get("LFM2_REAL_WEIGHTS")
    if flag is not None:
        return flag == "1"
    return local_model_path() is not None


def _real_embed():
    if not _use_real():
        return None
    from ..tt.weights import load_tensors

    return load_tensors(local_model_path(), ["model.embed_tokens.weight"])["model.embed_tokens.weight"]


def run_case(
    mesh_device,
    layer_idx,
    users,
    *,
    chunk_size,
    trace_decode=True,
    max_seq_len=None,
    kv_dtype=ttnn.bfloat16,
    reserve_seqs=None,
    decoder_factory=None,
):
    logger.info(f"TT_METAL_TRACE_ALLOC_TRACKING={os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')}")
    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    real = _use_real()
    sd = layer_state_dict(layer_idx, real)
    embed = _real_embed()
    H = args.hidden_size
    total = max(u.prompt_len + u.decode_steps for u in users)
    max_seq_len = max_seq_len or max(total, TILE)

    # HF reference: one causal full-sequence forward per user
    hf_layer, hf_rope = build_hf_layer(cfg, layer_idx, sd)
    for u in users:
        u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
        u.ref = hf_layer_forward(hf_layer, hf_rope, u.x.unsqueeze(0))[0]
    del hf_layer

    decoder, rope = make_decoder(decoder_factory, sd, cfg, args, layer_idx, mesh_device, max(max_seq_len, TILE))
    layout = PagedLayout(users, max_seq_len, reserve_seqs=reserve_seqs)
    if decoder.is_attention:
        decoder.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size, dtype=kv_dtype)
        logger.info(f"KV cache: {layout.num_blocks} blocks x {layout.block_size} tokens per K and V")

    # ---- prefill (chunked, one request at a time, each into its own slot) ----
    for u in users:
        u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, chunk_size)

    # ---- decode: all slots each step (inactive slots get cur_pos -1 / garbage rows) ----
    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq)
    table = layout.decode_table(users)
    steps = max(u.decode_steps for u in users)
    trace_id = None
    garbage = torch.Generator().manual_seed(1234)
    for t in range(steps):
        x = torch.randn(1, 1, TILE, H, generator=garbage) * 0.5  # inactive rows: garbage on purpose
        cur = torch.full((TILE,), -1, dtype=torch.int32)
        for u in users:
            if t < u.decode_steps:
                p = u.prompt_len + t
                x[0, 0, u.slot] = u.x[p]
                cur[u.slot] = p
        inp.update(x, cur, table)
        if not trace_decode:
            out = decode_step(decoder, rope, inp)
        else:
            if trace_id is None:
                # compile (this executes step t once), then restore the state that step mutated
                snapshot = _snapshot_state(decoder)
                out = decode_step(decoder, rope, inp)
                ttnn.deallocate(out)
                _restore_state(decoder, snapshot)
                ttnn.synchronize_device(mesh_device)
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                trace_out = decode_step(decoder, rope, inp)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                _restore_state(decoder, snapshot)
                inp.update(x, cur, table)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            out = trace_out
        host = from_dev(out, mesh_device)[0, 0]
        if not trace_decode:
            ttnn.deallocate(out)
        for u in users:
            if t < u.decode_steps:
                u.tt_decode.append(host[u.slot])
    if trace_id is not None:
        ttnn.release_trace(mesh_device, trace_id)

    results = []
    for u in users:
        p_pcc = pcc(u.tt_prefill, u.ref[: u.prompt_len])
        d_ref = u.ref[u.prompt_len : u.prompt_len + u.decode_steps]
        d_pcc = pcc(torch.stack(u.tt_decode), d_ref) if u.decode_steps else 1.0
        dp, dd = delta_pccs(u)
        logger.info(
            f"layer {layer_idx} ({args.layer_kind(layer_idx)}) slot {u.slot} prompt {u.prompt_len} "
            f"decode {u.decode_steps}: prefill PCC {p_pcc:.6f} decode PCC {d_pcc:.6f} "
            f"delta PCC {dp:.6f}/{dd:.6f} real={real}"
        )
        results.append((u, p_pcc, d_pcc))
    return results


def delta_pccs(u):
    """PCC of the layer's update ``out - x`` (residual removed) for prefill and decode rows.

    The layer output is dominated by the residual input, so a broken mixer/FFN can still show a
    high output PCC; the update PCC is the sensitive check."""
    n = u.prompt_len
    p = pcc(u.tt_prefill - u.x[:n], u.ref[:n] - u.x[:n])
    if not u.tt_decode:
        u.delta = (p, 1.0)
        return u.delta
    m = len(u.tt_decode)
    d = pcc(torch.stack(u.tt_decode) - u.x[n : n + m], u.ref[n : n + m] - u.x[n : n + m])
    u.delta = (p, d)
    return p, d


def functional_factory(sd, cfg, args, layer_idx, mesh_device, max_seq_len):
    decoder = FunctionalDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    rope = RopeSetup(mesh_device, args, max_seq_len=max_seq_len) if decoder.is_attention else None
    return decoder, rope


def make_decoder(factory, sd, cfg, args, layer_idx, mesh_device, max_seq_len):
    """``factory(sd, cfg, args, layer_idx, mesh_device, max_seq_len) -> (decoder, rope_or_None)``."""
    return (factory or functional_factory)(sd, cfg, args, layer_idx, mesh_device, max_seq_len)


def _snapshot_state(decoder):
    """Device copies of every tensor a decode step mutates (conv taps / KV caches)."""
    m = decoder.mixer
    if decoder.is_attention:
        return [(m.k_cache, ttnn.clone(m.k_cache)), (m.v_cache, ttnn.clone(m.v_cache))]
    return [(s, ttnn.clone(s)) for s in m.state]


def _restore_state(decoder, snapshot):
    for live, saved in snapshot:
        ttnn.copy(saved, live)


def _assert(results):
    for u, p, d in results:
        assert p >= PCC_BAR, f"slot {u.slot} prefill PCC {p}"
        assert d >= PCC_BAR, f"slot {u.slot} decode PCC {d}"
        # Update-PCC bar applies to real weights only: with synthetic weights the HF reference itself
        # run in bf16 scores 0.9196 on one batch32 conv_moe user (TT: 0.9199) because a near-tie flips
        # that token's expert selection (doc/fused_decoder/work_log.md, synthetic-weight control).
        dp, dd = getattr(u, "delta", (1.0, 1.0))
        if _use_real():
            assert dp >= DELTA_PCC_BAR and dd >= DELTA_PCC_BAR, f"slot {u.slot} update (out - x) PCC {dp}/{dd}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize(
    "prompt_len, chunk",
    [
        (1, 2048),  # single token prompt
        (31, 2048),  # below one tile
        (32, 2048),  # exactly one tile
        (64, 2048),  # exactly one block
        (65, 2048),  # just across a block boundary
        (300, 128),  # multiple chunks + tail
        (300, 2048),  # unchunked control for the case above
        (2049, 2048),  # one past a chunk boundary
    ],
)
def test_decoder_single_user(mesh_device, kind, prompt_len, chunk):
    users = [User(slot=5, prompt_len=prompt_len, decode_steps=4, seed=prompt_len)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=chunk))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_decoder_batch32(mesh_device, kind):
    """32 users in all slots, mixed non-aligned prompt lengths, then concurrent traced decode."""
    lens = [1 + (37 * i) % 200 for i in range(TILE)]
    users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=128))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_decoder_slot_reuse(mesh_device, kind):
    """A long request then a short one in the SAME slot (fresh request must not see stale state),
    interleaved with traced decode of background users; the short one waits 2 replays before
    its first decode."""
    fg = [
        User(slot=7, prompt_len=700, decode_steps=2, seed=1),
        User(slot=7, prompt_len=33, decode_steps=3, seed=2, join_delay=2),
    ]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=256))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_decoder_increasing_lengths_same_instance(mesh_device, kind):
    """Increasing lengths around the 2048 chunk boundary through ONE decoder instance and ONE
    slot (each request gets different pages), then short ones; every prefill chunk is followed
    by a traced decode replay of background users, and requests wait before joining decode."""
    lens = [2047, 2048, 2049, 4100, 32, 33]
    fg = [User(slot=9, prompt_len=n, decode_steps=2, seed=500 + n, join_delay=i % 3) for i, n in enumerate(lens)]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=2048))


BG_SLOTS = (2, 20)
SCRATCH_SLOT = TILE - 1


def run_interleaved(mesh_device, layer_idx, fg_users, *, chunk_size, decoder_factory=None):
    """Serving-like schedule on one decoder instance with ONE captured decode trace.

    Background users (slots ``BG_SLOTS``) are prefilled first and then decode on every replay.
    Foreground requests run one after another: each prefill chunk is followed by a decode replay
    (the request's slot is idle, ``cur_pos = -1``), then ``join_delay`` replays without it, then
    its decode steps.  Every prefill shape is compiled before trace capture (warm-up into a
    scratch slot / scratch pages), so all post-capture prefill allocations are transient.
    Run under ``TT_METAL_TRACE_ALLOC_TRACKING=1`` to have ``execute_trace`` reject unsafe buffers."""
    logger.info(f"TT_METAL_TRACE_ALLOC_TRACKING={os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')}")
    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    H = args.hidden_size
    sd = layer_state_dict(layer_idx, _use_real())
    embed = _real_embed()
    assert all(u.slot not in BG_SLOTS + (SCRATCH_SLOT,) for u in fg_users)

    # replays: one per fg chunk + join delays + fg decode steps (+1 so bg always has a next token)
    n_replays = sum(math.ceil(u.prompt_len / chunk_size) + u.join_delay + u.decode_steps for u in fg_users) + 1
    bg_users = [
        User(slot=s, prompt_len=40 + 23 * i, decode_steps=n_replays, seed=70 + i) for i, s in enumerate(BG_SLOTS)
    ]
    scratch = User(slot=SCRATCH_SLOT, prompt_len=max(u.prompt_len for u in fg_users), decode_steps=0, seed=0)
    users = bg_users + fg_users
    max_seq = max(u.prompt_len + u.decode_steps for u in users)

    hf_layer, hf_rope = build_hf_layer(cfg, layer_idx, sd)
    for u in users:
        u.x = layer_inputs(u.prompt_len + u.decode_steps, H, u.seed, embed).to(torch.bfloat16).float()
        u.ref = hf_layer_forward(hf_layer, hf_rope, u.x.unsqueeze(0))[0]
    del hf_layer

    decoder, rope = make_decoder(decoder_factory, sd, cfg, args, layer_idx, mesh_device, max_seq)
    layout = PagedLayout(users + [scratch], max_seq)
    if decoder.is_attention:
        decoder.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)

    for u in bg_users:
        u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, chunk_size)
    # warm-up: compile every fg prefill chunk shape/offset before capture (scratch slot + pages)
    for u in fg_users:
        tmp = User(slot=SCRATCH_SLOT, prompt_len=u.prompt_len, decode_steps=0, seed=0, x=u.x, blocks=scratch.blocks)
        run_prefill(decoder, rope, mesh_device, layout, tmp, chunk_size)

    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq)
    garbage = torch.Generator().manual_seed(99)
    active = []  # users decoding on the next replay
    step = {id(u): 0 for u in users}

    def load_inputs():
        x = torch.randn(1, 1, TILE, H, generator=garbage) * 0.5
        cur = torch.full((TILE,), -1, dtype=torch.int32)
        for u in active:
            p = u.prompt_len + step[id(u)]
            x[0, 0, u.slot] = u.x[p]
            cur[u.slot] = p
        inp.update(x, cur, layout.decode_table(active))

    active.extend(bg_users)
    load_inputs()
    snapshot = _snapshot_state(decoder)
    ttnn.deallocate(decode_step(decoder, rope, inp))  # compile
    _restore_state(decoder, snapshot)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = decode_step(decoder, rope, inp)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _restore_state(decoder, snapshot)
    del snapshot

    replays = 0

    def replay():
        nonlocal replays
        load_inputs()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        host = from_dev(trace_out, mesh_device)[0, 0]
        for u in list(active):
            u.tt_decode.append(host[u.slot])
            step[id(u)] += 1
            if step[id(u)] == u.decode_steps:
                active.remove(u)
        replays += 1

    for u in fg_users:
        u.tt_prefill = run_prefill(decoder, rope, mesh_device, layout, u, chunk_size, after_chunk=replay)
        for _ in range(u.join_delay):
            replay()
        active.append(u)
        while u in active:
            replay()
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"interleaved schedule: {replays} decode replays of one trace, {len(fg_users)} fg requests")

    results = []
    for u in users:
        n = len(u.tt_decode)
        p = pcc(u.tt_prefill, u.ref[: u.prompt_len])
        d = pcc(torch.stack(u.tt_decode), u.ref[u.prompt_len : u.prompt_len + n])
        dp, dd = delta_pccs(u)
        logger.info(
            f"interleaved layer {layer_idx} slot {u.slot} prompt {u.prompt_len} decode {n} "
            f"(delay {u.join_delay}): prefill PCC {p:.6f} decode PCC {d:.6f} delta PCC {dp:.6f}/{dd:.6f} "
            f"real={_use_real()}"
        )
        results.append((u, p, d))
    return results


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_decoder_full_context(mesh_device, kind):
    """HF-advertised context (max_position_embeddings = 131072): a 131069-token prompt (not a
    multiple of the chunk/page/tile) in 2048-token chunks, then traced decode at positions
    131069..131071 (the last position of the context), alongside a short user.  The attention KV cache reserves pages for 32 users at full
    context (the decode batch), proving the full batch x context cache fits one chip per layer."""
    args = Lfm2MoeArgs.from_hf_config(hf_config())
    ctx = args.max_position_embeddings
    users = [
        User(slot=9, prompt_len=ctx - 3, decode_steps=3, seed=7),
        User(slot=30, prompt_len=77, decode_steps=1, seed=8),
    ]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=2048, max_seq_len=ctx, reserve_seqs=TILE))
