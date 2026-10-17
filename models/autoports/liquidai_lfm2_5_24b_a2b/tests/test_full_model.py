# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model / generator tests on the reduced probe (layers 0, 2, 3 = conv_dense, attn_moe, conv_moe: one layer of each
kind, real weights / shapes / cache / page tables / final norm / LM head / sampling) on the 1x4 mesh.

Run the whole file with the trace allocation tracker (fresh process, program-cache allocations included):

    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \\
        pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_full_model.py

One generator (max_seq_len 131072, full 32 x 131072 cache) serves every test, so trace ids, captures and counters
accumulate across tests in file order: no test restarts the generator to hide a recapture.
"""

import contextlib
import os

import pytest
import torch
from loguru import logger

import ttnn

from ..tt.generator import build_generator, open_mesh
from .harness import pcc
from .hf_probe import HFProbe

PROBE = (0, 2, 3)
FULL_CONTEXT = 131072


@pytest.fixture(scope="module")
def env():
    logger.info(
        f"TT_METAL_TRACE_ALLOC_TRACKING={os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')} "
        f"TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE={os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE')}"
    )
    mesh = open_mesh()
    gen = build_generator(mesh_device=mesh, layer_indices=PROBE, max_seq_len=FULL_CONTEXT)
    ref = HFProbe(gen.model.model_path, gen.model.hf_config, PROBE)
    yield gen, ref
    gen.close()
    ttnn.close_mesh_device(mesh)


def text_ids(gen, n, offset=0):
    """``n`` real token ids: the chat-template AIME24 prompt (index 0), repeated, starting at ``offset``."""
    from .perf_full_model import _prompt

    base = _prompt(gen, 157)
    reps = (n + offset) // len(base) + 1
    return (base * reps)[offset : offset + n]


def dev_ints(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).long()


def dev_float(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def host_snapshot(tensors):
    """Host copies of device state, every chip's own slice (ShortConv state is channel-sharded); a device-side clone
    alive across a replay would be trace-unsafe."""
    return [(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0)), t) for t in tensors]


def host_restore(snap, gen):
    for h, t in snap:
        src = ttnn.from_torch(h, dtype=t.dtype, layout=t.layout, mesh_mapper=ttnn.ShardTensorToMesh(gen.mesh, dim=0))
        ttnn.copy_host_to_device_tensor(src, t)


def counters_delta(gen, c0):
    return {k: v - c0.get(k, 0) for k, v in gen.counters.items() if v != c0.get(k, 0)}


@contextlib.contextmanager
def no_host_transfer():
    """No host <-> device tensor transfer and no TTNN host fallback inside the block (steady-state replay loop)."""
    patched = {
        (ttnn, "from_torch"): ttnn.from_torch,
        (ttnn, "to_torch"): ttnn.to_torch,
        (ttnn, "copy_host_to_device_tensor"): ttnn.copy_host_to_device_tensor,
        (ttnn.Tensor, "cpu"): ttnn.Tensor.cpu,
    }

    def forbid(name):
        def f(*a, **k):
            raise AssertionError(f"{name} inside the steady-state decode loop")

        return f

    old = ttnn.CONFIG.throw_exception_on_fallback
    ttnn.CONFIG.throw_exception_on_fallback = True
    for obj, name in patched:
        setattr(obj, name, forbid(name))
    try:
        yield
    finally:
        for (obj, name), fn in patched.items():
            setattr(obj, name, fn)
        ttnn.CONFIG.throw_exception_on_fallback = old


def _start(gen, slot, prompt):
    """Prefill ``prompt`` in ``slot`` and bind it for decode; returns the first sampled token."""
    t = torch.tensor(prompt).reshape(1, -1)
    first = int(
        gen.prefill_forward(t, page_table=gen.slot_table[slot : slot + 1], prompt_lens=[len(prompt)], slots=[slot])[0]
    )
    pos = gen.pos_host.clone()
    pos[slot] = len(prompt)
    gen._refresh_positions(pos)
    gen._set_slot_tokens([slot], [first])
    return first


# ------------------------------------------------------------------------------------------------
# FIRST test of the file: the first reset() / reset_slots() of the generator happens AFTER a capture (both must have
# been compiled by prepare(); under the tracker a late compile would leave live program buffers across the next replay)
# ------------------------------------------------------------------------------------------------
def test_reset_after_capture(env):
    gen, _ = env
    assert gen.counters["resets"] == 0 and not gen.traces, "must run before any other test"
    ids = text_ids(gen, 70, 4)
    a = gen.generate(ids, 6, stop_on_eos=False)  # captures bucket 1
    assert gen.traces
    gen.reset()  # first reset, after the capture
    gen.reset_slots([0])
    b = gen.generate(ids, 6, stop_on_eos=False)  # replays after the reset
    assert a == b, (a, b)


def test_serving_row_semantics(env):
    """Per-request seed writes keep other slots' device RNG state; a short `tokens` vector keeps the device tokens of
    active slots it does not cover."""
    gen, _ = env
    gen.reset()
    gen._refresh_table(gen.slot_table)
    _start(gen, 0, text_ids(gen, 40, 1))
    gen.set_sampling_params(slots=[0], top_k=8, temperature=1.0, seed=11)
    _start(gen, 3, text_ids(gen, 44, 9))
    for _ in range(3):
        gen.decode_forward(read_from_device=False)
    seeds_before = dev_ints(gen.sampler.seeds_tt_tensor)
    gen.set_sampling_params(slots=[3], top_k=8, temperature=1.0, seed=99)
    seeds_after = dev_ints(gen.sampler.seeds_tt_tensor)
    assert seeds_after[0] == seeds_before[0], "seed write for slot 3 reset slot 0's RNG state"
    assert seeds_after[3] == 99 + 3
    gen.set_sampling_params(top_k=1)
    st = gen.model.state
    snap = host_snapshot((*gen.model.conv_states(), st.tokens, st.cur_pos, st.rope_pos))
    pos = gen.pos_host.clone()
    gen.decode_forward(tokens=[1234], read_from_device=False)  # covers slot 0 only; slot 3 keeps its device token
    l_short = gen.model.logits_to_host(gen.traces[4][1])[0, 0, 3]
    host_restore(snap, gen)  # control: the same step with slot 3's token written explicitly must match
    gen.pos_host = pos
    full = torch.zeros(32, dtype=torch.int64)
    full[0] = 1234
    full[3] = int(dev_ints(st.tokens)[3])
    gen.decode_forward(tokens=full, read_from_device=False)
    l_full = gen.model.logits_to_host(gen.traces[4][1])[0, 0, 3]
    assert torch.equal(l_short, l_full), "short tokens vector clobbered slot 3's token input"


# ------------------------------------------------------------------------------------------------
# split sampling: token feedback, positions, page tables, mode / bucket trace reuse
# ------------------------------------------------------------------------------------------------
def test_split_sampling_trace(env):
    gen, ref = env
    st = gen.model.state
    gen.reset()
    gen._refresh_table(gen.slot_table)
    prompt = text_ids(gen, 65)
    first = _start(gen, 0, prompt)
    assert dev_ints(st.tokens)[0] == first and dev_ints(st.cur_pos)[0] == 65 and dev_ints(st.rope_pos)[0] == 65
    hf_first = ref.logits(torch.tensor(prompt))[-1]
    assert first in hf_first.topk(5).indices.tolist(), (first, hf_first.topk(5))  # probe logits are flat: near ties

    # the 3-layer probe echoes its input token under greedy decoding, so replay 1 samples (top-k 32, temp 20) to make
    # its output differ from its input: a stale token input of replay 2 is then visible
    x = first
    gen.set_sampling_params(slots=[0], top_k=32, top_p=1.0, temperature=20.0, seed=3)
    c0 = dict(gen.counters)
    caps0 = gen.counters["trace_captures"]
    gen.decode_forward(read_from_device=False)  # replay 1: input x at 65
    tid1, logits, sid1 = gen.traces[1]
    l1 = gen.model.logits_to_host(logits)[0, 0, 0]
    t1 = int(dev_ints(st.tokens)[0])
    v, i = l1.topk(5)
    logger.info(
        f"replay-1 logits top5 {list(zip(i.tolist(), [round(float(a), 3) for a in v]))}; k/p/temp rows0: "
        f"{dev_ints(gen.sampler.k_tensor)[:2].tolist()} {dev_float(gen.sampler.p_tensor).reshape(-1)[:2].tolist()} "
        f"{dev_float(gen.sampler.temp_tensor).reshape(-1)[:2].tolist()}"
    )
    assert t1 in l1.topk(32).indices.tolist(), "sampled token is not a top-32 token of the replay's logits"
    gen.set_sampling_params(slots=[0], top_k=1)
    c0 = dict(gen.counters)  # (the sampling-parameter refresh above is a request-boundary write)
    assert t1 != x, "probe repeats the input token: feedback not observable"
    assert dev_ints(st.cur_pos)[0] == 66 and dev_ints(st.rope_pos)[0] == 66 and dev_ints(st.cur_pos)[1] == -1
    snap = host_snapshot((*gen.model.conv_states(), st.tokens, st.cur_pos, st.rope_pos))
    gen.decode_forward(read_from_device=False)  # replay 2: NO host input: consumes t1 at 66 from the device
    l2 = gen.model.logits_to_host(logits)[0, 0, 0]
    t2 = int(dev_ints(st.tokens)[0])
    assert dev_ints(st.cur_pos)[0] == 67 and dev_ints(st.rope_pos)[0] == 67
    assert not torch.equal(l1, l2), "logits did not change between replays"
    assert t2 == int(l2.argmax())
    hf = ref.logits(torch.tensor(prompt + [x, t1]))
    # sanity only (the bitwise control below is the proof); t1 is a high-temperature sample (off-distribution input)
    assert pcc(l1, hf[-2]) > 0.98 and pcc(l2, hf[-1]) > 0.98, (pcc(l1, hf[-2]), pcc(l2, hf[-1]))
    d = counters_delta(gen, c0)
    # control: the same step with t1 / position 66 written from the host must give bitwise the same logits
    host_restore(snap, gen)
    gen.decode_forward(tokens=[t1], start_pos=[66], read_from_device=False)
    l2_host = gen.model.logits_to_host(logits)[0, 0, 0]
    assert torch.equal(l2, l2_host), "replay 2 did not consume the fed-back token t1 at position 66"
    assert int(dev_ints(st.tokens)[0]) == t2 and dev_ints(st.cur_pos)[0] == 67
    for k in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes", "mask_refreshes"):
        assert d.get(k, 0) == 0, (k, d)
    assert d["model_replays"] == 1 and d["sampling_replays"] == 1, d  # replay 2: replays only, zero host writes
    logger.info(f"feedback: first {first} forced {x} t1 {t1} t2 {t2}; counters {d}")

    # unchanged page table: no copy; changed page table: one copy, output changes; restoring it: identical output
    def one_step(table):
        snap = host_snapshot((*gen.model.conv_states(), st.tokens))
        gen.decode_forward(start_pos=[67], page_table=table, read_from_device=False)
        out = gen.model.logits_to_host(logits)[0, 0, 0]
        host_restore(snap, gen)
        return out

    c0 = dict(gen.counters)
    la = one_step(gen.slot_table)
    assert counters_delta(gen, c0).get("page_table_refreshes", 0) == 0
    other = gen.slot_table.clone()
    other[0] = gen.slot_table[31]  # slot 0 reads slot 31's (empty) blocks
    lb = one_step(other)
    la2 = one_step(gen.slot_table)
    d = counters_delta(gen, c0)
    assert d["page_table_refreshes"] == 2, d
    assert not torch.equal(la, lb) and torch.equal(la, la2), "page-table change not consumed / not restored"

    # sampling mode switch (greedy -> top-k -> greedy) and bucket switch (1 -> 4 -> 1) reuse the captured traces
    ids_before = {b: (t[0], t[2]) for b, t in gen.traces.items()}
    gen.set_sampling_params(slots=[0], top_k=5, top_p=0.9, temperature=0.8, seed=7)
    gen.decode_forward(start_pos=[67], read_from_device=False)
    gen.set_sampling_params(slots=[0], top_k=1)
    gen.decode_forward(read_from_device=False)
    assert gen.traces[1][0] == ids_before[1][0] and gen.traces[1][2] == ids_before[1][1]
    _start(gen, 3, text_ids(gen, 40, 5))  # slot 3 active -> bucket 4 (first capture of bucket 4)
    gen.decode_forward(read_from_device=False)
    caps4 = gen.counters["trace_captures"]
    pos = gen.pos_host.clone()
    pos[3] = -1
    gen.decode_forward(start_pos=pos, read_from_device=False)  # back to bucket 1
    _start(gen, 3, text_ids(gen, 40, 5))
    gen.decode_forward(read_from_device=False)  # bucket 4 again
    assert gen.counters["trace_captures"] == caps4, "bucket switch recaptured"
    assert gen.traces[1][0] == ids_before[1][0]
    logger.info(f"captures: before {caps0} after {gen.counters['trace_captures']}, traces {list(gen.traces)}")


# ------------------------------------------------------------------------------------------------
# batch decode: fixed slots with holes, mixed prompt lengths, idle rows untouched, slot-independent results
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "slots", [(0, 1), (0, 5), (0, 1, 2, 4, 5, 6, 9, 13), tuple(range(32))], ids=["bucket2", "b2_hole", "b8", "b32"]
)
def test_batch_decode(env, slots):
    gen, ref = env
    steps = 8
    gen.reset()
    gen._refresh_table(gen.slot_table)
    lens = {s: 33 + 7 * (i % 9) + (i * 13) % 50 for i, s in enumerate(slots)}
    offs = {s: (17 * i) % 120 for i, s in enumerate(slots)}
    if 17 in slots:  # same prompt in slots 0 and 17: identical results expected
        lens[17], offs[17] = lens[0], offs[0]
    prompts = {s: text_ids(gen, lens[s], offs[s]) for s in slots}
    L = max(lens.values())
    toks = torch.zeros(len(slots), L, dtype=torch.int64)
    for i, s in enumerate(slots):
        toks[i, : lens[s]] = torch.tensor(prompts[s])
    table = gen.slot_table[list(slots)]
    first = gen.prefill_forward(toks, page_table=table, prompt_lens=[lens[s] for s in slots], slots=list(slots))
    idle = [s for s in range(32) if s not in slots]
    idle_rows = [dev_float(t)[0, 0, idle] for t in gen.model.conv_states()] if idle else []
    start = torch.full((32,), -1)
    tk = torch.zeros(32, dtype=torch.int64)
    for i, s in enumerate(slots):
        start[s], tk[s] = lens[s], first[i]
    gen.decode_forward(
        tokens=tk, start_pos=start, page_table=gen.slot_table, read_from_device=False, record_history=True
    )
    for _ in range(steps - 1):
        gen.decode_forward(read_from_device=False, record_history=True)
    hist = gen._take_history()  # [steps, 32]
    if idle:
        after = [dev_float(t)[0, 0, idle] for t in gen.model.conv_states()]
        assert all(torch.equal(a, b) for a, b in zip(idle_rows, after)), "idle slots' ShortConv state changed"
    top1, top5, n = 0, 0, 0
    for i, s in enumerate(slots):
        gen_s = [int(first[i])] + [int(x) for x in hist[:, s]]
        hf = ref.logits(torch.tensor(prompts[s] + gen_s[:-1]))[lens[s] - 1 :]
        for k, t in enumerate(gen_s):
            top1 += int(t == int(hf[k].argmax()))
            top5 += int(t in hf[k].topk(5).indices.tolist())
            n += 1
    logger.info(f"batch {len(slots)}: TT tokens vs HF probe top1 {top1}/{n} top5 {top5}/{n}")
    # the 3-layer probe has flat logits (near ties): every TT token in the HF top 5, most of them the HF argmax
    assert top5 == n and top1 >= 0.8 * n, (top1, top5, n)
    if 17 in slots:
        assert torch.equal(hist[:, 0], hist[:, 17]) and first[0] == first[17], "slot 0 and slot 17 differ"


# ------------------------------------------------------------------------------------------------
# prompt lengths through the public API (non-aligned, chunk boundaries, multi-chunk)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 31, 33, 65, 77, 300, 2047, 2049, 4100])
def test_prefill_lengths(env, n):
    gen, ref = env
    ids = text_ids(gen, n, 3)
    lg = gen.prefill_logits(ids)
    assert tuple(lg.shape) == (1, n, gen.model.vocab)
    hf = ref.logits(torch.tensor(ids))
    p, a = pcc(lg[0], hf), (lg[0].argmax(-1) == hf.argmax(-1)).float().mean().item()
    logger.info(f"prefill_logits n={n}: PCC {p:.5f} top1 {a:.3f}")
    if n < 8:  # a handful of positions: near ties of the flat probe logits decide top-1
        assert p > 0.99 and all(int(lg[0, i].argmax()) in hf[i].topk(5).indices.tolist() for i in range(n)), (p, a)
    else:
        assert p > 0.99 and a >= 0.85, (p, a)
    if n in (33, 2049, 4100):
        gen.reset()
        out = gen.generate(ids, 4, stop_on_eos=False)
        seq = list(ids)
        for t in out:
            hl = ref.logits(torch.tensor(seq))[-1]
            assert t in hl.topk(5).indices.tolist()
            seq.append(t)


def test_prefill_continuation_interleaved(env):
    """Chunked prefill of slot 2 (4100 tokens: chunks at 0 / 2048 / 4096) with decode replays of slot 0 between the
    chunks, plus a ShortConv snapshot / restore round trip of slot 2 between chunks."""
    gen, ref = env
    gen.reset()
    gen._refresh_table(gen.slot_table)
    _start(gen, 0, text_ids(gen, 50, 11))
    ids = text_ids(gen, 4100, 7)
    row = gen.slot_table[2:3]
    for c0, c1 in ((0, 2048), (2048, 4096), (4096, 4100)):
        tok = gen.prefill_forward(
            torch.tensor(ids[c0:c1]).reshape(1, -1), page_table=row, prompt_lens=[c1 - c0], slots=[2], start_pos=[c0]
        )
        if c1 == 2048:
            snap = gen.model.snapshot_conv_state(2)
            before = [dev_float(s)[0, 0, 2] for s in gen.model.conv_states()]
            gen.model.reset_conv_slots([2])
            assert all(dev_float(s)[0, 0, 2].abs().max() == 0 for s in gen.model.conv_states())
            gen.model.restore_conv_state(2, snap)
            assert all(torch.equal(b, dev_float(s)[0, 0, 2]) for b, s in zip(before, gen.model.conv_states()))
            for t in snap:
                ttnn.deallocate(t)
        for _ in range(3):  # another request decodes between the chunks
            gen.decode_forward(read_from_device=False)
    hf = ref.logits(torch.tensor(ids))[-1]
    assert int(tok[0]) == int(hf.argmax()), (int(tok[0]), int(hf.argmax()))
    pos = gen.pos_host.clone()
    pos[2] = 4100
    gen._refresh_positions(pos)
    gen._set_slot_tokens([2], [int(tok[0])])
    out = gen.decode_forward()
    hl = ref.logits(torch.tensor(ids + [int(tok[0])]))[-1]
    assert int(out[2]) == int(hl.argmax())


# ------------------------------------------------------------------------------------------------
# traced prefill: bitwise equal to the eager path; one capture per (bucket, first/continuation, head) key
# ------------------------------------------------------------------------------------------------
def _prefill_state(gen, ids, slot, traced):
    gen.reset()
    gen.traced_prefill = traced
    try:
        tok = gen.prefill_forward(
            torch.tensor(ids).reshape(1, -1),
            page_table=gen.slot_table[slot : slot + 1],
            prompt_lens=[len(ids)],
            slots=[slot],
        )
    finally:
        gen.traced_prefill = True
    logits = gen.model.logits_to_host(gen.pf_logits)[0, 0]
    conv = [ttnn.to_torch(s, mesh_composer=ttnn.ConcatMeshToTensor(gen.mesh, dim=0)) for s in gen.model.conv_states()]
    k0 = gen.model.kv[0][0]
    blocks = gen.slot_table[slot, : (len(ids) + 63) // 64].long()
    kv = ttnn.to_torch(k0, mesh_composer=ttnn.ConcatMeshToTensor(gen.mesh, dim=0)).reshape(4, *k0.shape)[:, blocks]
    return int(tok[0]), logits, conv, kv


@pytest.mark.parametrize("n", [1, 33, 300, 2049, 4100])
def test_prefill_traced_vs_eager(env, n):
    gen, _ = env
    ids = text_ids(gen, n, 5)
    c0 = dict(gen.counters)
    t_tok, t_lg, t_conv, t_kv = _prefill_state(gen, ids, 2, traced=True)
    d = counters_delta(gen, c0)
    assert d.get("prefill_replays", 0) == (n + 2047) // 2048 and d.get("prefill_eager_chunks", 0) == 0, d
    e_tok, e_lg, e_conv, e_kv = _prefill_state(gen, ids, 2, traced=False)
    assert t_tok == e_tok and torch.equal(t_lg, e_lg), (t_tok, e_tok)
    assert all(torch.equal(a, b) for a, b in zip(t_conv, e_conv)), "ShortConv state differs traced vs eager"
    assert torch.equal(t_kv, e_kv), "KV blocks differ traced vs eager"


def test_prefill_trace_reuse(env):
    """Prompt lengths / slots / page-table rows / continuation starts change only runtime inputs: each key is captured
    at most once, no trace is released, decode replays in between keep working."""
    gen, ref = env
    keys0 = dict(gen.prefill_traces)
    caps0 = gen.counters["prefill_trace_captures"]
    gen.reset()
    gen._refresh_table(gen.slot_table)
    for n, slot in ((31, 0), (33, 1), (31, 4), (20, 9)):  # bucket 32 except 33 -> 64
        _start(gen, slot, text_ids(gen, n, slot))
        gen.decode_forward(read_from_device=False)
    added = gen.counters["prefill_trace_captures"] - caps0
    new = set(gen.prefill_traces) - set(keys0)
    assert added == len(new) and new <= {(32, False, "last"), (64, False, "last")}, (added, new)
    assert all(gen.prefill_traces[k] == v for k, v in keys0.items()), "a prefill trace was replaced"
    # two sequences in one call: both first tokens through one prefill-sampling replay
    ids_a, ids_b = text_ids(gen, 40, 3), text_ids(gen, 70, 8)
    toks = torch.zeros(2, 70, dtype=torch.int64)
    toks[0, :40], toks[1] = torch.tensor(ids_a), torch.tensor(ids_b)
    gen.reset()
    first = gen.prefill_forward(toks, page_table=gen.slot_table[[5, 6]], prompt_lens=[40, 70], slots=[5, 6])
    for i, ids in ((0, ids_a), (1, ids_b)):
        assert int(first[i]) in ref.logits(torch.tensor(ids))[-1].topk(5).indices.tolist()
    caps1 = gen.counters["prefill_trace_captures"]
    for start in (2048, 4096):  # continuation chunks at different starts reuse one key
        gen.prefill_forward(
            torch.tensor(text_ids(gen, 100, 1)).reshape(1, -1),
            page_table=gen.slot_table[7:8],
            prompt_lens=[100],
            slots=[7],
            start_pos=[start],
        )
    assert gen.counters["prefill_trace_captures"] - caps1 <= 1


def test_prefill_allocation_free(env):
    """After its keys are captured, a traced prefill only refreshes the persistent inputs (host copies) and replays:
    no device tensor is created from the host (device-bound ``from_torch``) between replays."""
    gen, _ = env
    lens = (33, 2049)
    for n in lens:  # make sure the keys exist
        gen.reset()
        gen.generate(text_ids(gen, n, 2), 1, stop_on_eos=False)
    orig = ttnn.from_torch

    def guarded(*a, **k):
        assert k.get("device") is None, "device allocation from host inside the traced prefill path"
        return orig(*a, **k)

    caps = gen.counters["prefill_trace_captures"]
    ttnn.from_torch = guarded
    try:
        for n in lens:
            gen.reset()
            ids = text_ids(gen, n, 2)
            gen.prefill_forward(
                torch.tensor(ids).reshape(1, -1), page_table=gen.slot_table[:1], prompt_lens=[n], slots=[0]
            )
    finally:
        ttnn.from_torch = orig
    assert gen.counters["prefill_trace_captures"] == caps


# ------------------------------------------------------------------------------------------------
# determinism, no host transfer in the steady-state loop, host-sampling compatibility
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b", [1, 32])
def test_determinism_no_fallback(env, b):
    gen, ref = env
    slots = list(range(b))
    runs = []
    for _ in range(2):
        gen.reset()
        gen._refresh_table(gen.slot_table)
        prompts = [text_ids(gen, 40 + s, s) for s in slots]
        L = max(len(p) for p in prompts)
        toks = torch.zeros(b, L, dtype=torch.int64)
        for i, p in enumerate(prompts):
            toks[i, : len(p)] = torch.tensor(p)
        first = gen.prefill_forward(
            toks, page_table=gen.slot_table[:b], prompt_lens=[len(p) for p in prompts], slots=slots
        )
        start = torch.full((32,), -1)
        start[:b] = torch.tensor([len(p) for p in prompts])
        tk = torch.zeros(32, dtype=torch.int64)
        tk[:b] = first
        gen.decode_forward(tokens=tk, start_pos=start, page_table=gen.slot_table, read_from_device=False)
        logits = gen.traces[None if b == 32 else 1][1]
        with no_host_transfer():
            for _ in range(16):
                gen.decode_forward(read_from_device=False, record_history=True)
        runs.append((gen._take_history(), gen.model.logits_to_host(logits)))
    assert torch.equal(runs[0][0], runs[1][0]), "token-out decode not deterministic"
    assert torch.equal(runs[0][1][..., :b, :], runs[1][1][..., :b, :]), "logits not bitwise reproducible"


def test_host_sampling_compat(env):
    gen, _ = env
    ids = text_ids(gen, 90, 2)
    gen.reset()
    dev = gen.generate(ids, 32, stop_on_eos=False)
    gen.reset()
    c0 = dict(gen.counters)
    host = gen.generate(ids, 32, host_sampling=True, stop_on_eos=False)
    d = counters_delta(gen, c0)
    assert host == dev, (host, dev)
    assert d.get("full_logits_readbacks", 0) >= 31 and d.get("sampling_replays", 0) == 0, d


# ------------------------------------------------------------------------------------------------
# full context: 131069-token prompt (64 chunks), decode to the last position
# ------------------------------------------------------------------------------------------------
def test_full_context(env, expect_error):
    gen, _ = env
    n = FULL_CONTEXT - 3
    ids = text_ids(gen, n, 1)
    gen.reset()
    c0 = dict(gen.counters)
    out = gen.generate(ids, 3, stop_on_eos=False)
    gen.reset()
    host = gen.generate(ids, 3, host_sampling=True, stop_on_eos=False)
    d = counters_delta(gen, c0)
    logger.info(f"full context {n} + 3: tokens {out} (host argmax {host}); counters {d}")
    assert out == host
    assert int(gen.pos_host[0]) == FULL_CONTEXT - 1
    with expect_error(ValueError, "exceeds max_seq_len"):
        gen.generate(ids + [1, 2, 3], 2)  # beyond the context
