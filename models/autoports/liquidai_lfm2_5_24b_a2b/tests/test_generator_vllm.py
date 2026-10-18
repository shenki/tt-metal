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


def _sp(n, temperature=0.0, top_k=1, top_p=1.0, seed=None):
    from vllm_tt_plugin.model_input import TTSamplingParams

    return TTSamplingParams(
        temperature=[temperature] * n,
        top_k=[top_k] * n,
        top_p=[top_p] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[1.0] * n,
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
    ad, B, tokens, pos, table_rows, sp=None, remap=None, reload=True, page_only=False, params=False, reset=False
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
    rows == B prefilled from scratch.  With the 4096 snapshot evicted the adapter walks down to 2048 / 0."""
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
    # evict the 4096 entry: walk down to 2048
    h4096 = ad.snaps.chain(b, 2)[1]
    e = ad.snaps.lru.pop(h4096)
    ad.snaps.free.append(e)
    rec0 = ad.snaps.stats["recompute_tokens"]
    hit2 = _prefill(ad, [(b, 4096, len(b), 12, blk_b)])
    assert ad.snaps.stats["recompute_tokens"] - rec0 == 2048
    assert torch.equal(hit2, ref)


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
    """Both earlier snapshots evicted (4096 and 2048): a hit at P = 4160 walks down to Q = 0 and recomputes 65 blocks
    into scratch (more than one chunk).  The owner's blocks stay bitwise unchanged; the hit matches a from-scratch
    prefill (top-1, PCC)."""
    ad = adapter
    a = _prompt(5000, 71)
    blk_a = ad.blocks.take(80)
    _prefill(ad, [(a, 0, 5000, 0, blk_a)], sampling=_sp(1))
    for h in ad.snaps.chain(a, 2):
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


def test_no_recapture_after_warmup(adapter):
    """Every trace was captured during warm-up: the tests above caused no capture."""
    ad = adapter
    assert (int(ad.gen.counters["trace_captures"]), int(ad.gen.counters["prefill_trace_captures"])) == ad.captures0
