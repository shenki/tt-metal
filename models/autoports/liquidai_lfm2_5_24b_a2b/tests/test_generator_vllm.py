# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the vLLM adapter (tt/generator_vllm.py) driven with plugin-shaped keyword arguments (no server).

Default target: the reduced serving model (layers 0, 2, 3: ShortConv + dense, attention + MoE, ShortConv + MoE) through
the same adapter; ``LFM2_VLLM_LAYERS=all`` runs the 40-layer model.  Controls are the adapter's / generator's own
paths (whole-prompt prefill, host-authoritative reloads, unmoved slots), compared exactly where the programs are the
same and by top-1 + PCC where they differ (eager readiness logits).

    pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_generator_vllm.py
"""

import os

import pytest
import torch

import ttnn

PT_W = 2048  # page-table width = max_model_len / block size
N_BLOCKS = 4096  # small vLLM pool for the tests (block 0 = null block; the adapter adds its scratch blocks)


def _sp(n, temperature=0.0, top_k=1, top_p=1.0, seed=None, presence=0.0, frequency=0.0, repetition=1.0):
    from vllm_tt_plugin.model_input import TTSamplingParams

    per_row = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * n
    return TTSamplingParams(
        temperature=per_row(temperature),
        top_k=per_row(top_k),
        top_p=[top_p] * n,
        presence_penalty=per_row(presence),
        frequency_penalty=per_row(frequency),
        repetition_penalty=per_row(repetition),
        seed=[seed] * n if not isinstance(seed, list) else seed,
        num_logprobs=[-2] * n,
        enable_log_probs=[False] * n,
    )


class Blocks:
    """Minimal block allocator: request -> list of physical blocks (block 0 never handed out)."""

    def __init__(self):
        self.next = 1

    def take(self, n):
        b = list(range(self.next, self.next + n))
        self.next += n
        assert self.next < N_BLOCKS
        return b


def _row(blocks):
    r = torch.zeros(PT_W, dtype=torch.int32)
    r[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return r


@pytest.fixture(scope="module")
def adapter():
    layers = os.environ.get("LFM2_VLLM_LAYERS", "0,2,3")
    if layers == "all":
        os.environ.pop("LFM2_VLLM_LAYERS", None)
    else:
        os.environ["LFM2_VLLM_LAYERS"] = layers
    from transformers import AutoConfig

    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import open_mesh
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm import TTLfm2MoeForCausalLM
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import resolve_model_path

    mesh = open_mesh(768 << 20)
    cfg = AutoConfig.from_pretrained(str(resolve_model_path()))
    ad = TTLfm2MoeForCausalLM.initialize_vllm_model(cfg, mesh, 32, max_seq_len=131072)
    kv = ad.allocate_kv_cache((N_BLOCKS, 2, 64, 64), torch.bfloat16, 10)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    ad.warmup_model_decode(
        kv_cache=kv, max_batch_size=32, num_blocks=PT_W, can_sample_on_device=True, enable_trace=False
    )
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(
        kv_cache=kv, max_batch_size=32, num_blocks=PT_W, can_sample_on_device=True, enable_trace=True
    )
    ad.kv = kv
    ad.blocks = Blocks()
    ad.captures0 = (int(ad.gen.counters["trace_captures"]), int(ad.gen.counters["prefill_trace_captures"]))
    yield ad
    ad.release_persistent_capture()
    ttnn.close_mesh_device(mesh)


def _prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, 60000, (n,), generator=g)


def _prefill(ad, rows, sampling=None):
    """rows: [(tokens history, start, end, slot, blocks)] -> adapter output."""
    L = max(len(t) for t, *_ in rows)
    toks = torch.zeros(len(rows), L, dtype=torch.int64)
    for i, (t, *_rest) in enumerate(rows):
        toks[i, : len(t)] = t
    import numpy as np

    return ad.prefill_forward(
        tokens=toks,
        start_pos=np.array([r[1] for r in rows]),
        prompt_lens=np.array([r[2] for r in rows]),
        page_table=torch.stack([_row(r[4]) for r in rows]),
        kv_cache=ad.kv,
        enable_trace=True,
        empty_slots=[r[3] for r in rows],
        sampling_params=sampling,
    )


def _decode(
    ad,
    B,
    tokens,
    pos,
    table_rows,
    sp=None,
    remap=None,
    reload=True,
    page_only=False,
    params=False,
    reset=False,
    hist=None,
):
    tk = torch.zeros(B, 1, dtype=torch.int64)
    p = torch.full((B,), -1, dtype=torch.int64)
    pt = torch.zeros(B, PT_W, dtype=torch.int32)
    for r, (t, q, blk) in enumerate(zip(tokens, pos, table_rows)):
        if q is None:
            continue
        tk[r, 0], p[r], pt[r] = int(t), int(q), _row(blk)
    out = ad.decode_forward(
        tokens=tk,
        start_pos=p,
        page_table=pt,
        kv_cache=ad.kv,
        enable_trace=True,
        read_from_device=False,
        slot_remap=remap,
        sampling_params=sp,
        reload_inputs=reload,
        reload_page_table=page_only,
        reload_sampling_params=params,
        reset_sampling_state=reset,
        **({} if hist is None else dict(prompt_tokens=hist[0], output_tokens=hist[1])),
    )
    h, ev = ad.read_decode_output(out, async_read=True)
    for e in ev:
        ttnn.event_synchronize(e)
    return ad.process_decode_output_host(h, is_tokens=sp is not None)


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _conv_rows(ad, slot):
    snap = ad.model.snapshot_conv_state(slot)
    rows = [ttnn.to_torch(ttnn.get_device_tensors(s)[0])[0, 0, 0].clone() for s in snap]
    for s in snap:
        ttnn.deallocate(s)
    return rows


def test_precision_and_registration(adapter):
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm import TTLfm2MoeForCausalLM

    assert adapter.model.precision.config_id == "C1"
    assert adapter.model.plan.ccl_dtype == "bfp8"
    assert TTLfm2MoeForCausalLM.decode_input_update_contract == 1
    assert TTLfm2MoeForCausalLM.tt_supported_decode_batch_sizes == (1, 2, 4, 8, 16, 32)


@pytest.mark.parametrize("n", [1, 31, 33, 129, 130, 131, 2047, 2049, 4100])
def test_nonaligned_prefill(adapter, n):
    """Host-logits prefill of a non-aligned prompt == last row of the eager whole-prompt readiness logits."""
    ad = adapter
    t = _prompt(n, n)
    blk = ad.blocks.take((n + 63) // 64)
    out = _prefill(ad, [(t, 0, n, 4, blk)])
    assert out.shape == (1, 1, ad.model.vocab)
    ref = ad.gen.prefill_forward(
        t.reshape(1, -1), page_table=_row(blk).reshape(1, -1), prompt_lens=[n], slots=[4], return_all_logits=True
    )
    r = ref[0, n - 1]
    assert int(out[0, 0].argmax()) == int(r.argmax())
    assert _pcc(out[0, 0], r) > 0.999
    tok = _prefill(ad, [(t, 0, n, 4, blk)], sampling=_sp(1))
    assert int(tok[0, 0]) == int(out[0, 0].argmax())


def test_chunked_continuation_and_slot_move(adapter):
    """Prompt split at 2048 / 4096 across calls, continuation landing in another slot (state restored from the pool)
    and in the same slot (no restore) == the whole prompt in one call (same traced programs: bitwise)."""
    ad = adapter
    n = 5000
    t = _prompt(n, 7)
    blk = ad.blocks.take((n + 63) // 64)
    whole = _prefill(ad, [(t, 0, n, 6, blk)])
    st = dict(ad.snaps.stats)
    _prefill(ad, [(t, 0, 2048, 7, blk)])  # intermediate chunk, slot 7
    _prefill(ad, [(t, 2048, 4096, 8, blk)])  # continuation moved to slot 8: restore
    last = _prefill(ad, [(t, 4096, n, 8, blk)])  # same slot: state already there
    assert torch.equal(last, whole)
    assert ad.snaps.stats["restores"] > st["restores"] and ad.snaps.stats["in_slot"] > st["in_slot"]
    assert all(torch.equal(a, b) for a, b in zip(_conv_rows(ad, 6), _conv_rows(ad, 8)))


def test_prefix_hit_and_evicted_walkdown(adapter):
    """B shares 4096 tokens with A; a vLLM prefix hit starts B at 4096 (A's blocks) in a new slot: logits and ShortConv
    rows == B prefilled from scratch.  With the 4096 snapshot evicted the adapter walks down to the 3584 grid point (an
    inner boundary of A's [2048, 4096) chunk, tapped) and recomputes 512 tokens."""
    ad = adapter
    a = _prompt(4100, 11)
    b = torch.cat([a[:4096], _prompt(70, 12)])
    blk_a = ad.blocks.take(65)
    _prefill(ad, [(a, 0, 4100, 9, blk_a)])
    blk_b_own = ad.blocks.take(3)
    blk_b = blk_a[:64] + blk_b_own
    ref_blk = ad.blocks.take(66)
    ref = _prefill(ad, [(b, 0, len(b), 10, ref_blk)])
    ref_rows = _conv_rows(ad, 10)
    hit = _prefill(ad, [(b, 4096, len(b), 11, blk_b)])
    assert torch.equal(hit, ref)
    assert all(torch.equal(x, y) for x, y in zip(_conv_rows(ad, 11), ref_rows))
    # evict the 4096 entry: walk down to 3584 (512 grid)
    h4096 = ad.snaps.chain(b, 8)[7]
    e = ad.snaps.lru.pop(h4096)
    ad.snaps.free.append(e)
    rec0 = ad.snaps.stats["recompute_tokens"]
    hit2 = _prefill(ad, [(b, 4096, len(b), 12, blk_b)])
    assert ad.snaps.stats["recompute_tokens"] - rec0 == 512
    # [3584, 4166) runs as one 1024-bucket chunk: bitwise equal to B prefilled with the same chunk split
    split_blk = ad.blocks.take(66)
    for lo, hi in ((0, 2048), (2048, 3584)):
        _prefill(ad, [(b, lo, hi, 13, split_blk)])
    split = _prefill(ad, [(b, 3584, len(b), 13, split_blk)])
    assert torch.equal(hit2, split)
    assert int(hit2[0, 0].argmax()) == int(ref[0, 0].argmax())


def test_tap_snapshot_equals_chunk_end_state(adapter):
    """The state tapped at an inner boundary of a 2048-row chunk (A: [2048, 4100) chunk, boundary 3584) is bitwise the
    end state of a chunk of the same bucket that stops there ([2048, 3584), also bucket 2048)."""
    ad = adapter
    a = _prompt(4100, 81)
    _prefill(ad, [(a, 0, 4100, 3, ad.blocks.take(65))])
    e = ad.snaps.lookup(ad.snaps.chain(a, 7)[6])
    assert e is not None, "no tap snapshot at 3584"
    ad.model.restore_conv_state_from(4, e)
    tapped = _conv_rows(ad, 4)
    blk = ad.blocks.take(56)
    _prefill(ad, [(a, 0, 2048, 5, blk)])
    _prefill(ad, [(a, 2048, 3584, 5, blk)])
    assert all(torch.equal(x, y) for x, y in zip(tapped, _conv_rows(ad, 5)))


def test_prefix_hit_on_512_grid(adapter):
    """B shares 2624 tokens with A (P = 2624: neither a chunk end nor on the 2048 grid): the adapter restores the 2560
    tap snapshot and recomputes [2560, 2624) through a continuation chunk at a 512-aligned start; matches B from
    scratch (top-1, PCC: different bucket split) and A's cached blocks are not written."""
    ad = adapter
    a = _prompt(4100, 91)
    blk_a = ad.blocks.take(65)
    _prefill(ad, [(a, 0, 4100, 6, blk_a)])
    b = torch.cat([a[:2624], _prompt(100, 92)])
    before = _kv_blocks(ad, blk_a[40:42])
    rec0, res0 = ad.snaps.stats["recompute_tokens"], ad.snaps.stats["restores"]
    hit = _prefill(ad, [(b, 2624, len(b), 7, blk_a[:41] + ad.blocks.take(2))])
    assert ad.snaps.stats["recompute_tokens"] - rec0 == 64 and ad.snaps.stats["restores"] == res0 + 1
    assert all(torch.equal(x, y) for x, y in zip(before, _kv_blocks(ad, blk_a[40:42])))
    # no bitwise control exists here: the hit reuses A's K/V of [2048, 2560) from a 2048-row chunk, which a fresh B
    # can only reproduce by computing past 2560 (exactness of the tap itself: test_tap_snapshot_equals_chunk_end_state)
    ref = _prefill(ad, [(b, 0, len(b), 9, ad.blocks.take(43))])
    assert int(hit[0, 0].argmax()) == int(ref[0, 0].argmax())
    assert _pcc(hit[0, 0], ref[0, 0]) > 0.999


def _kv_blocks(ad, ids):
    """Chip-0 K and V contents of physical blocks ``ids`` for every attention layer."""
    out = []
    for k, v in ad.kv_handle:
        for t in (k, v):
            out.append(ttnn.to_torch(ttnn.get_device_tensors(t)[0])[torch.as_tensor(ids)].clone())
    return out


def _decode_alone(ad, slot_tokens, pos, blk, steps):
    toks, seq = slot_tokens, []
    for i in range(steps):
        out = _decode(ad, 1, [toks], [pos], [blk], sp=_sp(1), reload=i == 0, params=i == 0, reset=i == 0)
        toks = int(out[0, 0])
        pos += 1
        seq.append(toks)
    return seq


def test_prefix_hit_does_not_mutate_shared_blocks(adapter):
    """Owner A (5000 tokens) caches its blocks; B hits A's cache at P = 4160 (not on the 2048 grid), so the adapter
    recomputes [4096, 4160) from the 4096 snapshot through a different prefill bucket.  That recompute must land in
    adapter scratch blocks: A's blocks are bitwise unchanged and A's greedy continuation equals a control without B."""
    ad = adapter
    a = _prompt(5000, 61)
    ctrl_blk = ad.blocks.take(80)
    t0 = int(_prefill(ad, [(a, 0, 5000, 0, ctrl_blk)], sampling=_sp(1))[0, 0])
    ctrl = _decode_alone(ad, t0, 5000, ctrl_blk, 8)
    blk_a = ad.blocks.take(80)
    t1 = int(_prefill(ad, [(a, 0, 5000, 0, blk_a)], sampling=_sp(1))[0, 0])
    assert t1 == t0
    before = _kv_blocks(ad, blk_a[64:66])
    b = torch.cat([a[:4160], _prompt(40, 62)])
    blk_b = blk_a[:65] + ad.blocks.take(1)
    rows0 = ad.snaps.stats["scratch_rows"]
    hit = _prefill(ad, [(b, 4160, 4200, 1, blk_b)])
    assert ad.snaps.stats["scratch_rows"] == rows0 + 1
    after = _kv_blocks(ad, blk_a[64:66])
    assert all(torch.equal(x, y) for x, y in zip(before, after))
    assert _decode_alone(ad, t1, 5000, blk_a, 8) == ctrl
    ref = _prefill(ad, [(b, 0, 4200, 2, ad.blocks.take(66))])
    assert int(hit[0, 0].argmax()) == int(ref[0, 0].argmax())
    assert _pcc(hit[0, 0], ref[0, 0]) > 0.999


def test_prefix_hit_two_level_eviction(adapter):
    """Every earlier snapshot evicted (512-grid points up to 4096): a hit at P = 4160 walks down to Q = 0 and recomputes 65 blocks
    into scratch (more than one chunk).  The owner's blocks stay bitwise unchanged; the hit matches a from-scratch
    prefill (top-1, PCC)."""
    ad = adapter
    a = _prompt(5000, 71)
    blk_a = ad.blocks.take(80)
    _prefill(ad, [(a, 0, 5000, 0, blk_a)], sampling=_sp(1))
    for h in ad.snaps.chain(a, 4160 // 512):  # every grid point <= 4160 (chunk ends and taps)
        e = ad.snaps.lru.pop(h, None)
        if e is not None:
            ad.snaps.free.append(e)
    before = _kv_blocks(ad, blk_a[:66])
    b = torch.cat([a[:4160], _prompt(40, 72)])
    rec0 = ad.snaps.stats["recompute_tokens"]
    hit = _prefill(ad, [(b, 4160, 4200, 1, blk_a[:65] + ad.blocks.take(1))])
    assert ad.snaps.stats["recompute_tokens"] - rec0 == 4160
    after = _kv_blocks(ad, blk_a[:66])
    assert all(torch.equal(x, y) for x, y in zip(before, after))
    ref = _prefill(ad, [(b, 0, 4200, 2, ad.blocks.take(66))])
    assert int(hit[0, 0].argmax()) == int(ref[0, 0].argmax())
    assert _pcc(hit[0, 0], ref[0, 0]) > 0.999


def _greedy_run(ad, slot_rows, steps, mode, seed0=100):
    """Prefill prompts into slots 0..k-1 then decode ``steps`` greedy steps.  mode "resident": reload only on the
    first step (device token feedback / positions after); "reload": host-authoritative reload every step."""
    prompts = [_prompt(n, seed0 + n) for n in slot_rows]
    blks = [ad.blocks.take((n + steps + 63) // 64) for n in slot_rows]
    first = _prefill(
        ad, [(p, 0, len(p), s, b) for s, (p, b) in enumerate(zip(prompts, blks))], sampling=_sp(len(prompts))
    )
    B = next(b for b in (1, 2, 4, 8, 16, 32) if b >= len(prompts))
    toks = [int(x) for x in first[:, 0]]
    pos = [len(p) for p in prompts]
    seq = [[t] for t in toks]
    for i in range(steps):
        reload = mode == "reload" or i == 0
        out = _decode(
            ad,
            B,
            toks + [0] * (B - len(toks)),
            pos + [None] * (B - len(pos)),
            blks + [[]] * (B - len(blks)),
            sp=_sp(B),
            reload=reload,
            params=reload,
            reset=reload,
        )
        toks = [int(x) for x in out[: len(prompts), 0]]
        pos = [q + 1 for q in pos]
        for s, t in enumerate(toks):
            seq[s].append(t)
    return seq


def test_stale_inputs_resident_vs_reload(adapter):
    """Steady device decode with stale host tokens / positions (reload_inputs=False) == host-authoritative reloads."""
    ad = adapter
    res = _greedy_run(ad, [37, 130, 75], 24, "resident")
    ref = _greedy_run(ad, [37, 130, 75], 24, "reload")
    assert res == ref


def test_page_growth_page_only_refresh(adapter):
    """Three requests crossing 64-token block boundaries with page-table-only refreshes (blocks handed out lazily) ==
    full reload every step; unchanged steps copy no page table."""
    ad = adapter
    prompts = [_prompt(n, 300 + n) for n in (60, 62, 127)]
    steps = 40
    own = [ad.blocks.take(1 + len(p) // 64) for p in prompts]
    first = _prefill(ad, [(p, 0, len(p), s, b) for s, (p, b) in enumerate(zip(prompts, own))], sampling=_sp(3))
    toks, pos = [int(x) for x in first[:, 0]], [len(p) for p in prompts]
    seq = []
    for i in range(steps):
        grow = False
        for s in range(3):
            if pos[s] // 64 >= len(own[s]):
                own[s] = own[s] + ad.blocks.take(1)
                grow = True
        before = int(ad.gen.counters["page_table_refreshes"])
        out = _decode(
            ad,
            4,
            toks + [0],
            pos + [None],
            own + [[]],
            sp=_sp(4),
            reload=i == 0,
            page_only=grow and i > 0,
            params=i == 0,
            reset=i == 0,
        )
        if i > 0 and not grow:
            assert int(ad.gen.counters["page_table_refreshes"]) == before
        toks = [int(x) for x in out[:3, 0]]
        pos = [q + 1 for q in pos]
        seq.append(toks)
    # control: same prompts, full reload every step with the final (fully grown) tables
    ref_seq = _greedy_run(ad, [60, 62, 127], steps, "reload", seed0=300)
    assert [list(x) for x in zip(*seq)] == [r[1:] for r in ref_seq]


def _swap_run(ad, prompts, steps, swap_at=None, swap_reload=False):
    """Greedy decode of 3 requests (slots 0..2); at step ``swap_at`` requests 0 and 2 trade slots via ``slot_remap``
    (resident inputs moved by the adapter, or reloaded permuted from the host).  Returns per-request token lists."""
    blks = [ad.blocks.take(4) for _ in prompts]
    first = _prefill(ad, [(p, 0, len(p), s, b) for s, (p, b) in enumerate(zip(prompts, blks))], sampling=_sp(3))
    order = [0, 1, 2]  # order[slot] = request
    toks, pos = [int(x) for x in first[:, 0]], [len(p) for p in prompts]
    seq = [[t] for t in toks]
    for i in range(steps):
        remap = None
        reload = i == 0
        if i == swap_at:
            remap = list(range(32))
            remap[0], remap[2] = 2, 0
            order = [order[j] for j in remap[:3]]
            reload = swap_reload
        req_tok = {order[s]: None for s in range(3)}
        out = _decode(
            ad,
            4,
            [toks[order[s]] for s in range(3)] + [0],
            [pos[order[s]] for s in range(3)] + [None],
            [blks[order[s]] for s in range(3)] + [[]],
            sp=_sp(4),
            remap=remap,
            reload=reload,
            page_only=(remap is not None and not reload),
            params=reload,
            reset=reload,
        )
        for s in range(3):
            req_tok[order[s]] = int(out[s, 0])
        for r in range(3):
            toks[r] = req_tok[r]
            pos[r] += 1
            seq[r].append(toks[r])
    return seq


@pytest.mark.parametrize("swap_reload", [False, True])
def test_slot_remap_swap(adapter, swap_reload):
    """slot_remap mid-decode (ShortConv rows, sampling rows, resident token / position / page-table rows) keeps every
    request's greedy continuation identical to the unswapped run."""
    ad = adapter
    prompts = [_prompt(n, 500 + n) for n in (41, 90, 66)]
    ref = _swap_run(ad, prompts, 8)
    got = _swap_run(ad, prompts, 8, swap_at=3, swap_reload=swap_reload)
    assert got == ref


def test_remap_exact_move(adapter):
    """A pure slot permutation moves every ShortConv row, the resident token and the position exactly."""
    ad = adapter
    p = _prompt(50, 77)
    blk = ad.blocks.take(2)
    _prefill(ad, [(p, 0, 50, 5, blk)], sampling=_sp(1))
    rows5 = _conv_rows(ad, 5)
    remap = list(range(32))
    remap[1], remap[5] = 5, 1
    ad.gen.permute_slots(remap, resident_inputs=True)
    assert all(torch.equal(a, b) for a, b in zip(_conv_rows(ad, 1), rows5))
    ad.gen.permute_slots(remap, resident_inputs=True)
    assert all(torch.equal(a, b) for a, b in zip(_conv_rows(ad, 5), rows5))


def test_host_logits_decode_matches_device_greedy(adapter):
    """Host-sampling decode (model trace only, logits out) argmax == device greedy tokens of the same request."""
    ad = adapter
    p = _prompt(88, 900)
    blk = ad.blocks.take(3)
    dev = _greedy_run_single(ad, p, blk, 12, host=False)
    host = _greedy_run_single(ad, p, blk, 12, host=True)
    assert dev == host


def _greedy_run_single(ad, p, blk, steps, host):
    first = _prefill(ad, [(p, 0, len(p), 0, blk)], sampling=None if host else _sp(1))
    t = int(first[0, 0].argmax()) if host else int(first[0, 0])
    pos, seq = len(p), [t]
    for i in range(steps):
        out = _decode(
            ad,
            1,
            [t],
            [pos],
            [blk],
            sp=None if host else _sp(1),
            reload=host or i == 0,
            params=not host and i == 0,
            reset=not host and i == 0,
        )
        t = int(out[0, 0].argmax()) if host else int(out[0, 0])
        pos += 1
        seq.append(t)
    return seq


def test_determinism_slots_and_repeats(adapter):
    """Same prompt in slot 0 and slot 13, twice: identical prefill logits."""
    ad = adapter
    p = _prompt(300, 4242)
    outs = [_prefill(ad, [(p, 0, 300, s, ad.blocks.take(5))]) for s in (0, 13, 0, 13)]
    assert all(torch.equal(o, outs[0]) for o in outs)


def test_seeded_sampling_reproducible_across_slots(adapter):
    """A seeded top-k request draws the same tokens alone in slot 0 and next to two other requests in slot 2."""
    ad = adapter
    p = _prompt(70, 31337)

    def run(slot, others):
        rows = [(_prompt(40 + i, 9 + i), 0, 40 + i, s, ad.blocks.take(2)) for i, s in enumerate(others)]
        rows.append((p, 0, 70, slot, ad.blocks.take(2)))
        rows.sort(key=lambda r: r[3])
        n = len(rows)
        seeds = [42 if r[3] == slot else None for r in rows]
        sp = _sp(n, temperature=0.8, top_k=20, top_p=0.95, seed=seeds)
        first = _prefill(ad, rows, sampling=sp)
        B = next(b for b in (1, 2, 4, 8) if b >= n)
        toks = [int(x) for x in first[:, 0]]
        pos = [r[2] for r in rows]
        blks = [r[4] for r in rows]
        seq = [toks[slot]]
        spB = _sp(B, temperature=0.8, top_k=20, top_p=0.95, seed=seeds + [None] * (B - n))
        for i in range(10):
            out = _decode(
                ad,
                B,
                toks + [0] * (B - n),
                pos + [None] * (B - n),
                blks + [[]] * (B - n),
                sp=spB,
                reload=i == 0,
                params=i == 0,
                reset=i == 0,
            )
            toks = [int(x) for x in out[:n, 0]]
            pos = [q + 1 for q in pos]
            seq.append(toks[slot])
        return seq

    alone = run(0, [])
    crowded = run(2, [0, 1])
    assert alone == crowded


# ----------------------------------------------------------------------------------------------
# device penalties and top_k up to 64 (stage 10)
# ----------------------------------------------------------------------------------------------
# strong enough to break the reduced 3-layer target's single-token loops within a few steps
PEN = dict(presence=2.0, frequency=1.5, repetition=1.5)


def _vllm_penalties(logits, prompt_mask, counts, presence, frequency, repetition):
    """vLLM's ``apply_penalties`` (``vllm/model_executor/layers/utils.py``) in fp32: repetition scaling on the raw
    logit sign for tokens in the prompt or the output, then ``- frequency * counts - presence * (counts > 0)``."""
    x = logits.clone().float()
    seen = (prompt_mask > 0) | (counts > 0)
    scale = torch.where(x > 0, torch.full_like(x, 1.0 / repetition), torch.full_like(x, repetition))
    x = torch.where(seen, x * scale, x)
    return x - frequency * counts - presence * (counts > 0).float()


def test_penalty_op_matches_vllm_order(adapter):
    """Logit-level check of the device penalty op on a known state (row 0: prompt tokens, output tokens with repeats,
    presence 0.5 / frequency 1.0 / repetition 1.3): equal to vLLM's order within bf16 rounding and measurably away from
    the presence/frequency-first order on tokens that are in both the prompt and the output."""
    ad = adapter
    g = ad.gen
    V = g.model.padded_vocab
    torch.manual_seed(5)
    logits = torch.randn(1, 1, 32, V) * 4
    hot = torch.arange(1000, 1032)  # in prompt and output, positive logits: where the two orders differ most
    logits[0, 0, 0, hot] = 6.0
    prompt = torch.cat([hot, torch.arange(5000, 5100)])
    output = torch.cat([hot.repeat(3), torch.arange(7000, 7050)])
    pt = torch.full((32, len(prompt)), -1, dtype=torch.int32)
    ot = torch.full((32, len(output)), -1, dtype=torch.int32)
    pt[0], ot[0] = prompt.to(torch.int32), output.to(torch.int32)
    g.set_penalty_params([0], presence=0.5, frequency=1.0, repetition=1.3)
    g.reset_penalty_state(pt, ot)
    dev_in = ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=g.mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(g.mesh, dim=-1),
    )
    out = g._apply_penalties(dev_in)
    got = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], -1).float()[0, 0, 0]
    for t in (dev_in, out):
        ttnn.deallocate(t)
    g.set_penalty_params([0])
    g.reset_penalty_state(torch.full((32, 1), -1), torch.full((32, 1), -1))
    x = logits[0, 0, 0].bfloat16().float()
    prompt_mask = torch.zeros(V)
    prompt_mask[prompt] = 1
    counts = torch.bincount(output, minlength=V).float()
    want = _vllm_penalties(x[None], prompt_mask[None], counts[None], 0.5, 1.0, 1.3)[0]
    tt_order = x - 1.0 * counts - 0.5 * (counts > 0).float()
    tt_order = torch.where(tt_order > 0, tt_order / 1.3, tt_order * 1.3) * (
        (prompt_mask > 0) | (counts > 0)
    ) + tt_order * ~((prompt_mask > 0) | (counts > 0))
    tol = 0.02 * want.abs() + 0.02
    assert ((got - want).abs() <= tol).all(), float((got - want).abs().max())
    assert ((got[hot] - tt_order[hot]).abs() > 0.5).all()  # 4.615 - 3.5 vs (6 - 3.5) / 1.3: 1.12 vs 1.92


def _hist(prompts, outputs):
    """Plugin-shaped penalty histories: [32, L] int32, -1 padded, row = decode row."""
    pt = torch.full((32, max(len(p) for p in prompts)), -1, dtype=torch.int32)
    ot = torch.full((32, max(1, max(len(o) for o in outputs))), -1, dtype=torch.int32)
    for r, (p, o) in enumerate(zip(prompts, outputs)):
        pt[r, : len(p)] = torch.as_tensor(p, dtype=torch.int32)
        ot[r, : len(o)] = torch.as_tensor(o, dtype=torch.int32)
    return pt, ot


def _device_run(ad, rows, steps, reset_at=(), temperature=0.0, top_k=1, seed=None):
    """rows: [(prompt, penalties dict or None)] in slots 0..n-1, one device prefill + ``steps`` device decode steps
    (resident token feedback; reload + params + reset with histories at step 0 and at ``reset_at``)."""
    n = len(rows)
    B = next(b for b in (1, 2, 4, 8) if b >= n)
    pens = [r[1] or dict(presence=0.0, frequency=0.0, repetition=1.0) for r in rows]
    sp = lambda m: _sp(  # noqa: E731
        m,
        temperature=temperature,
        top_k=top_k,
        seed=[seed] * m,
        presence=[p["presence"] for p in pens] + [0.0] * (m - n),
        frequency=[p["frequency"] for p in pens] + [0.0] * (m - n),
        repetition=[p["repetition"] for p in pens] + [1.0] * (m - n),
    )
    blks = [ad.blocks.take(3) for _ in rows]
    first = _prefill(ad, [(p, 0, len(p), s, blks[s]) for s, (p, _) in enumerate(rows)], sampling=sp(n))
    seqs = [[int(first[s, 0])] for s in range(n)]
    pos = [len(p) for p, _ in rows]
    for i in range(steps):
        boundary = i == 0 or i in reset_at
        out = _decode(
            ad,
            B,
            [q[-1] for q in seqs] + [0] * (B - n),
            pos + [None] * (B - n),
            blks + [[]] * (B - n),
            sp=sp(B),
            reload=boundary,
            params=boundary,
            reset=boundary,
            hist=_hist([p for p, _ in rows], seqs) if boundary else None,
        )
        for s in range(n):
            seqs[s].append(int(out[s, 0]))
        pos = [q + 1 for q in pos]
    return seqs, blks


def _host_logits_along(ad, p, blk, seq):
    """Host-path logits of request ``p`` teacher-forced along ``seq`` (model trace only, no sampler): [len(seq), V]."""
    out = [_prefill(ad, [(p, 0, len(p), 0, blk)], sampling=None)[0, 0]]
    for j, t in enumerate(seq[:-1]):
        out.append(_decode(ad, 1, [t], [len(p) + j], [blk], sp=None)[0, 0])
    return torch.stack(out).float()


def test_device_penalties_match_reference(adapter):
    """Greedy + presence / frequency / repetition on device: every step's token is the fp32 vLLM-order penalized argmax
    of the host logits (``_vllm_penalties``), up to bf16 near-ties; the penalties change the output; a neutral row next
    to it is bitwise the plain greedy stream; a mid-stream history rebuild leaves the stream unchanged."""
    ad = adapter
    p, q = _prompt(120, 777), _prompt(90, 778)
    stale0 = int(ad.gen.counters["penalty_stale_replays"])
    (pen_seq, plain_seq), _ = _device_run(ad, [(p, PEN), (q, None)], 24)
    (pen_seq_reset, _), _ = _device_run(ad, [(p, PEN), (q, None)], 24, reset_at=(7, 15))
    assert pen_seq_reset == pen_seq  # rebuild from histories == on-device counting
    greedy_q = _greedy_run_single(ad, q, ad.blocks.take(3), 24, host=False)
    assert plain_seq == greedy_q  # neutral rows are untouched by the penalized trace
    greedy_p = _greedy_run_single(ad, p, ad.blocks.take(3), 24, host=False)
    assert pen_seq != greedy_p  # the penalties act
    logits = _host_logits_along(ad, p, ad.blocks.take(3), pen_seq)
    V = logits.shape[-1]
    prompt_mask = torch.zeros(1, V)
    prompt_mask[0, p] = 1
    counts = torch.zeros(1, V)
    exact, worst = 0, 0.0
    for j, t in enumerate(pen_seq):
        ref = _vllm_penalties(
            logits[j : j + 1], prompt_mask, counts, PEN["presence"], PEN["frequency"], PEN["repetition"]
        )[0]
        exact += int(ref.argmax()) == t
        worst = max(worst, float(ref.max() - ref[t]))
        counts[0, t] += 1
    # bf16 penalty arithmetic: one bf16 ulp of the penalized logit's magnitude (plus one for the logit itself)
    assert worst <= 2 * 2.0 ** (
        torch.log2(logits.abs().max()).floor().item() - 7
    ), f"device token {worst:.3f} below max"
    assert exact >= len(pen_seq) - 3, f"only {exact}/{len(pen_seq)} steps match the fp32 reference argmax"
    assert int(ad.gen.counters["penalty_stale_replays"]) == stale0


@pytest.mark.skipif(
    os.environ.get("LFM2_MAX_DEVICE_TOP_K", "32") == "32", reason="device top-k is capped at 32 (ttnn.sampling)"
)
@pytest.mark.parametrize("top_k", [50, 64])
def test_device_topk_above_32(adapter, top_k):
    """top_k 50 / 64 stays on device: every sampled token is inside the host top-k of its logits, and with a flat
    distribution (temperature 200) tokens beyond rank 32 are drawn (the k > 32 candidates are really used)."""
    ad = adapter
    assert ad.model_capabilities["max_device_top_k"] == 64  # documents the ttnn.sampling 32 cap when forced to 64
    p = _prompt(64, 4100 + top_k)
    (seq,), (blk,) = _device_run(ad, [(p, None)], 30, temperature=200.0, top_k=top_k, seed=11)
    logits = _host_logits_along(ad, p, ad.blocks.take(3), seq)
    ranks = [int((lg > lg[t]).sum()) for lg, t in zip(logits, seq)]
    assert max(ranks) < top_k, ranks
    assert any(r >= 32 for r in ranks), ranks


def test_seed_mirror_matches_device(adapter):
    """The host seed mirror tracks the device RNG through prefill / decode / remap / host-logits steps, so request
    boundaries never read seeds back (one verification read at the end)."""
    ad = adapter
    g = ad.gen
    reads0 = int(g.counters["seed_readbacks"])
    _device_run(ad, [(_prompt(50, 61), PEN), (_prompt(40, 62), None)], 5, temperature=0.7, top_k=20, seed=5)
    _greedy_run_single(ad, _prompt(30, 63), ad.blocks.take(2), 3, host=True)
    remap = list(range(32))
    remap[0], remap[1] = 1, 0
    a, b = _prompt(33, 64), _prompt(35, 65)
    ba, bb = ad.blocks.take(2), ad.blocks.take(2)
    first = _prefill(ad, [(a, 0, 33, 0, ba), (b, 0, 35, 1, bb)], sampling=_sp(2, temperature=0.9, top_k=10, seed=3))
    _decode(
        ad,
        2,
        [int(first[1, 0]), int(first[0, 0])],
        [35, 33],
        [bb, ba],
        sp=_sp(2, temperature=0.9, top_k=10),
        remap=remap,
        params=True,
        reset=True,
        hist=_hist([b, a], [[int(first[1, 0])], [int(first[0, 0])]]),
    )
    assert int(g.counters["seed_readbacks"]) == reads0
    dev = g._read_seeds() & 0xFFFFFFFF
    assert torch.equal(dev, g.seed_host & 0xFFFFFFFF), (dev[:4], g.seed_host[:4])


def test_no_recapture_after_warmup(adapter):
    """Every trace was captured during warm-up: the tests above caused no capture."""
    ad = adapter
    assert (int(ad.gen.counters["trace_captures"]), int(ad.gen.counters["prefill_trace_captures"])) == ad.captures0
