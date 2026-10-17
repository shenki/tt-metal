# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model performance on a built generator (TTFT also at prompts 1024 / 2048 / 4100; (called by ``readiness_driver --runs perf``).

Workload shape of the headline numbers: batch 1, prompt 128 tokens (chat-template AIME24 prompt truncated to 128 ids) /
generate 128, greedy.  Reported (wall clock, tracking / profiler off):

* ``ttft_ms``: ``generate`` start -> first token on host (eager chunked prefill + traced prefill sampling + readback);
* ``token_out``: free-running decode (model trace + sampling trace + history-append trace per step, non-blocking replays,
  one history readback at the end) -> ms/step and t/s/u over 127 steps;
* ``teacher_forcing``: ``generate(next_input=...)`` (per step: replays + 128-byte token readback + callback);
* ``logits_only``: model-trace replays only (the PERF-style traced decode without sampling / feedback);
* ``sampling_only``: sampling-trace replays only;
* batch token-out ms/step for buckets 1 / 4 / 8 / 32 (slots 0..b-1 active, mixed prompt lengths);
* ``force_argmax``: the same on a second generator over the same model with ``sampling_strategy="argmax"``.
"""

import statistics
import time

import torch
from loguru import logger

import ttnn


def _prompt(gen, n=128, index=0):
    """Chat-template AIME24 prompt ``index`` (the readiness reference prompt source) cut / repeated to ``n`` ids."""
    from readiness_check.generate import DEFAULT_AIME24_PROMPTS_FILE, _load_aime24_prompt

    text = _load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE, index)
    ids = gen.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def _fresh(gen):
    """Request boundary for the measurements: reset, then wait for its device work (KV / state zeroing) so a request's
    TTFT does not include the previous request's reset."""
    gen.reset()
    ttnn.synchronize_device(gen.mesh)


def _time_replays(gen, trace_ids, iters):
    ttnn.synchronize_device(gen.mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        for t in trace_ids:
            ttnn.execute_trace(gen.mesh, t, cq_id=0, blocking=False)
    ttnn.synchronize_device(gen.mesh)
    return (time.perf_counter() - t0) / iters * 1e3


def batch_token_out(gen, prompt, b, steps=64):
    """Slots 0..b-1 active (prompt lengths prompt_len - 3 s), free-running token-out decode, ms/step."""
    _fresh(gen)
    lens = [max(8, len(prompt) - 3 * s) for s in range(b)]
    toks = torch.zeros(b, len(prompt), dtype=torch.int64)
    for s in range(b):
        toks[s, : lens[s]] = torch.tensor(prompt[: lens[s]])
    first = gen.prefill_forward(toks, page_table=gen.slot_table[:b], prompt_lens=lens, slots=list(range(b)))
    gen.decode_forward(tokens=first, start_pos=torch.tensor(lens), page_table=gen.slot_table, read_from_device=False)
    for _ in range(4):
        gen.decode_forward(read_from_device=False, record_history=True)
    ttnn.synchronize_device(gen.mesh)
    t0 = time.perf_counter()
    for _ in range(steps):
        gen.decode_forward(read_from_device=False, record_history=True)
    ttnn.synchronize_device(gen.mesh)
    ms = (time.perf_counter() - t0) / steps * 1e3
    gen._take_history()
    return ms


def run_perf(gen, repeats=3, batches=(1, 4, 8, 32), argmax=True):
    out = {"workload": "batch 1, prompt 128 / generate 128, greedy"}
    prompt = _prompt(gen)
    _fresh(gen)
    gen.generate(prompt, 8)  # deferred first captures (bucket 1 decode, the bucket-128 prefill trace, samplers)
    out["first_request_ttft_ms"] = gen.last_perf["ttft_s"] * 1e3
    res = []
    for _ in range(repeats):
        _fresh(gen)
        c0 = dict(gen.counters)
        gen.generate(prompt, 128, stop_on_eos=False)  # headline: no EOS polling (one history readback per request)
        c1 = dict(gen.counters)
        res.append(dict(gen.last_perf))
    out["token_out_counters_per_request"] = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1 if c1.get(k, 0) != c0.get(k, 0)}
    out["ttft_ms"] = statistics.median(r["ttft_s"] * 1e3 for r in res)
    out["token_out_tok_s_u"] = statistics.median(r["decode_tok_s_u"] for r in res)
    out["token_out_ms_step"] = 1e3 / out["token_out_tok_s_u"]
    res = []
    for _ in range(repeats):
        _fresh(gen)
        gen.generate(prompt, 128, next_input=lambda i, p: p, stop_on_eos=False)
        res.append(dict(gen.last_perf))
    out["teacher_forcing_tok_s_u"] = statistics.median(r["decode_tok_s_u"] for r in res)
    out["teacher_forcing_ms_step"] = 1e3 / out["teacher_forcing_tok_s_u"]
    out["ttft_ms_by_prompt_len"] = {}
    for n in (128, 1024, 2048, 4100):
        p = _prompt(gen, n)
        _fresh(gen)
        gen.generate(p, 2, stop_on_eos=False)  # captures this length's prefill keys once
        vals = []
        for _ in range(repeats):
            _fresh(gen)
            gen.generate(p, 2, stop_on_eos=False)
            vals.append(gen.last_perf["ttft_s"] * 1e3)
        out["ttft_ms_by_prompt_len"][str(n)] = statistics.median(vals)
        logger.info(f"PERF TTFT prompt {n}: {out['ttft_ms_by_prompt_len'][str(n)]:.1f} ms")
    out["prefill_trace_bytes_per_bank"] = dict(gen.last_perf.get("prefill_trace_bytes_per_bank", {}))
    out["prefill_capture_s"] = dict(gen.last_perf.get("prefill_capture_s", {}))
    tid, _, sid = gen.traces[1]
    out["logits_only_ms_step"] = _time_replays(gen, [tid], 100)
    out["sampling_only_ms_step"] = _time_replays(gen, [sid], 100)
    out["model_plus_sampling_ms_step"] = _time_replays(gen, [tid, sid], 100)
    out["batch_token_out_ms_step"] = {}
    for b in batches:
        out["batch_token_out_ms_step"][str(b)] = batch_token_out(gen, prompt, b)
        logger.info(f"PERF batch {b}: {out['batch_token_out_ms_step'][str(b)]:.2f} ms/step")
    _fresh(gen)
    out["trace_region_used"] = _trace_usage(gen.mesh)
    _fresh(gen)
    gen.generate(prompt, 128, stop_on_eos=True)  # EOS polling every 16 steps (default free-running mode)
    out["token_out_eos_polling_tok_s_u"] = gen.last_perf["decode_tok_s_u"]
    _fresh(gen)
    ref = gen.generate(prompt, 128, stop_on_eos=False)
    if argmax:
        # second generator over the same model: built after gen's last replay, and gen is not replayed afterwards
        # (gen's traces never see g2's buffers allocated after their capture)
        from ..tt.generator import Lfm2Generator

        g2 = Lfm2Generator(gen.model, sampling_strategy="argmax")
        g2.kv, g2.owns_cache, g2.slot_table, g2.bps = gen.kv, True, gen.slot_table, gen.bps
        g2.prepare()
        g2.reset()
        g2.generate(prompt, 8)
        res = []
        for _ in range(repeats):
            g2.reset()
            o = g2.generate(prompt, 128, stop_on_eos=False)
            res.append(dict(g2.last_perf))
        tid2, _, sid2 = g2.traces[1]
        out["force_argmax"] = {
            "token_out_tok_s_u": statistics.median(r["decode_tok_s_u"] for r in res),
            "sampling_only_ms_step": _time_replays(g2, [sid2], 100),
            "same_tokens_as_split_greedy": o == ref,
        }
        g2.release_traces()
        del g2  # its buffers / programs were created after gen's captures: run perf LAST in a process (driver order)
    return out


def _trace_usage(mesh):
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return {k: getattr(v, k) for k in dir(v) if not k.startswith("_") and isinstance(getattr(v, k), (int, float))}
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"
