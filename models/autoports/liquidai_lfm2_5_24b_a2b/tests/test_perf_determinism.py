# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced decode measurement, determinism and the runtime fallback audit.

For each layer kind:

* prefill: one 2048-token chunk for a fresh request (compile run first, then the measured run);
* decode: all 32 slots active at different positions, captured once as a trace, replayed.

Every measured pass runs under ``no_host_fallback()``: any torch call (``TorchFunctionMode``),
``ttnn.from_torch``/``ttnn.to_torch``/``Tensor.cpu`` or TTNN host fallback raises.  Determinism:
the measured prefill must equal the compile run bit for bit, and two replays from the same
restored state must be identical.  Tracy signposts ``PERF_PREFILL``/``PERF_PREFILL_END`` and
``PERF_DECODE``/``PERF_DECODE_END`` bracket the warmed windows (decode: ``DECODE_ITERS`` replays)::

    python -m tracy -r -p -v -m pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_perf_determinism.py -k <kind>
"""

import contextlib
import time

import pytest
import torch
from loguru import logger
from torch.overrides import TorchFunctionMode

import ttnn

from ..tt.functional_decoder import ConvPrefillInputs
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
    pad_page_table_row,
    to_dev,
)
from .test_functional_decoder import LAYERS, _real_embed, _restore_state, _snapshot_state, _use_real, make_decoder

PREFILL_LEN = 2048
DECODE_ITERS = 5  # keeps every replay inside the device-profiler buffer

try:
    from tracy import signpost
except ImportError:  # profiler not built in: signposts are no-ops

    def signpost(header, message=None):
        pass


class _NoTorch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"torch call inside a measured device pass: {func}")


@contextlib.contextmanager
def no_host_fallback():
    """Fail on any torch / host conversion / TTNN host fallback inside the block."""

    def _forbidden(name):
        def f(*a, **k):
            raise AssertionError(f"{name} inside a measured device pass")

        return f

    patched = {
        (ttnn, "from_torch"): ttnn.from_torch,
        (ttnn, "to_torch"): ttnn.to_torch,
        (ttnn.Tensor, "cpu"): ttnn.Tensor.cpu,
    }
    old_throw = ttnn.CONFIG.throw_exception_on_fallback
    ttnn.CONFIG.throw_exception_on_fallback = True
    for obj, name in patched:
        setattr(obj, name, _forbidden(f"{getattr(obj, '__name__', obj)}.{name}"))
    try:
        with _NoTorch():
            yield
    finally:
        for (obj, name), fn in patched.items():
            setattr(obj, name, fn)
        ttnn.CONFIG.throw_exception_on_fallback = old_throw


def _prefill_inputs(decoder, mesh_device, layout, user, rope):
    """Device inputs for one fresh-request chunk at position 0 (built before the measured pass)."""
    n = user.prompt_len
    x = torch.zeros(1, 1, n, user.x.shape[-1])
    x[0, 0] = user.x[:n]
    tt_x = to_dev(x, mesh_device)
    if decoder.is_attention:
        table = to_dev(layout.user_table(user), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        chunk_table = to_dev(
            pad_page_table_row(user.blocks[: n // layout.block_size]).reshape(1, -1),
            mesh_device,
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        start = to_dev(torch.tensor([0], dtype=torch.int32), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kw = dict(rope=rope, chunk_start=0, page_table=table, chunk_page_table=chunk_table, chunk_start_tensor=start)
    else:
        kw = dict(
            conv_inputs=ConvPrefillInputs.create(
                mesh_device,
                slot=user.slot,
                valid_len=n,
                padded_len=n,
                continuation=False,
                windowed=getattr(decoder, "conv_prefill_windowed", False),
            )
        )
    return tt_x, kw


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_perf_determinism_no_fallback(mesh_device, kind):
    run_perf_determinism(mesh_device, kind)


def run_perf_determinism(
    mesh_device, kind, decoder_factory=None, decode_iters=DECODE_ITERS, active_users=TILE, active_slots=None
):
    """``active_users``: decode slots with a real token (cur_pos >= 0); the rest are idle
    (cur_pos = -1, garbage rows), e.g. 1 for single-user latency.  ``active_slots``: static decode
    bucket of the optimized decoder (the active users then occupy slots 0..active_users-1)."""
    layer_idx = LAYERS[kind]
    cfg = hf_config()
    args = Lfm2MoeArgs.from_hf_config(cfg)
    H = args.hidden_size
    sd = layer_state_dict(layer_idx, _use_real())
    embed = _real_embed()
    max_seq = PREFILL_LEN + 2 * TILE
    decoder, rope = make_decoder(decoder_factory, sd, cfg, args, layer_idx, mesh_device, max_seq)
    users = [User(slot=s, prompt_len=PREFILL_LEN, decode_steps=1, seed=900 + s) for s in range(TILE)]
    layout = PagedLayout(users, max_seq)
    if decoder.is_attention:
        decoder.mixer.allocate_kv_cache(layout.num_blocks, layout.block_size)
    u = users[3]
    u.x = layer_inputs(PREFILL_LEN, H, u.seed, embed).to(torch.bfloat16).float()

    # ---------------- prefill: compile run, then measured run (same fresh request) ----------------
    tt_x, kw = _prefill_inputs(decoder, mesh_device, layout, u, rope)
    out = decoder.prefill_forward(tt_x, **kw)
    first = from_dev(out, mesh_device)
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL")
    t0 = time.perf_counter()
    with no_host_fallback():
        out = decoder.prefill_forward(tt_x, **kw)
    ttnn.synchronize_device(mesh_device)
    prefill_s = time.perf_counter() - t0
    signpost("PERF_PREFILL_END")
    second = from_dev(out, mesh_device)
    ttnn.deallocate(out)
    assert torch.equal(first, second), "prefill is not deterministic for a repeated fresh request"

    # ---------------- decode: `active_users` slots active (slot 3 first), traced ----------------
    inp = DecodeInputs(mesh_device, H, layout.max_blocks_per_seq, active_slots=active_slots)
    gen = torch.Generator().manual_seed(4321)
    x = layer_inputs(TILE, H, 4321, embed).reshape(1, 1, TILE, H).to(torch.bfloat16).float()
    cur = torch.randint(PREFILL_LEN - TILE, PREFILL_LEN + TILE, (TILE,), generator=gen, dtype=torch.int32)
    if active_slots is None:
        active = [3] + [s for s in range(TILE) if s != 3][: active_users - 1]
    else:
        assert active_users <= active_slots
        active = list(range(active_users))
    idle = torch.ones(TILE, dtype=torch.bool)
    idle[active] = False
    cur[idle] = -1
    inp.update(x, cur, layout.decode_table(users))
    snapshot = _snapshot_state(decoder)
    out = decode_step(decoder, rope, inp)  # compile
    ttnn.deallocate(out)
    _restore_state(decoder, snapshot)
    ttnn.synchronize_device(mesh_device)
    with no_host_fallback():
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_out = decode_step(decoder, rope, inp)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _restore_state(decoder, snapshot)

    replays = []
    for _ in range(2):
        with no_host_fallback():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replays.append(from_dev(trace_out, mesh_device))
        _restore_state(decoder, snapshot)
    # idle slots (cur_pos = -1) have undefined output rows by contract (attention skips them);
    # every downstream op is row-local, so only active rows are compared
    rows = torch.tensor(active)
    assert torch.equal(replays[0][..., rows, :], replays[1][..., rows, :]), "traced decode replay is not deterministic"

    # warmed measurement window: DECODE_ITERS replays (state advances; values are not checked)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE")
    t0 = time.perf_counter()
    with no_host_fallback():
        for _ in range(decode_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_s = (time.perf_counter() - t0) / decode_iters
    signpost("PERF_DECODE_END")
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(
        f"{kind}: prefill {PREFILL_LEN} deterministic, traced decode x{decode_iters} deterministic, no host fallback"
    )
    return {"prefill_ms": prefill_s * 1e3, "decode_us": decode_s * 1e6}
