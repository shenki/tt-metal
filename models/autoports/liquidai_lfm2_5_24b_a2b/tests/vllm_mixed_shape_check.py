# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-shape trace-reuse check ($vllm-integration) against a running server whose adapter writes LFM2_VLLM_EVENTS.

Workload (one live server, no restart, no per-signature warm-up): unseen non-aligned lengths, chunked long prompts
arriving while other requests decode (continuation starts 2048 / 4096, partial final chunks), repeats of known lengths
in different batch compositions / orders, greedy -> sampled -> host-sampled (logprobs) -> greedy switches and a
prefix-cache hit; device penalties (penalized greedy repeated in two batch compositions, penalized sampling, an
admission while penalized rows decode), checkpoint-default host sampling (top_k 50) and a hit on the 512-token
ShortConv grid.  Reports per-request TTFT / ITL, decode pauses and throughput, and from the adapter events the
actual chunk rows (start, end, slot, recompute start), decode buckets / reload modes and the trace capture counters
(must not change after warm-up).  Greedy determinism: identical requests must produce identical text in every batch.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_mixed_shape_check --events <jsonl> --out <dir>
"""

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import requests

MODEL = "LiquidAI/LFM2.5-24B-A2B"


def _ids(n, seed):
    import random

    r = random.Random(seed)
    return [r.randrange(1000, 60000) for _ in range(n)]


def stream(url, prompt, max_tokens, **params):
    """Streaming /v1/completions; returns dict(text, ttft, itl list, total, tokens)."""
    body = dict(model=MODEL, prompt=prompt, max_tokens=max_tokens, stream=True, repetition_penalty=1.0)
    body.update(params)
    t0 = time.perf_counter()
    times, text = [], []
    with requests.post(f"{url}/v1/completions", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                times.append(time.perf_counter() - t0)
                text.append(chunk["choices"][0]["text"])
    itl = [b - a for a, b in zip(times, times[1:])]
    return dict(
        text="".join(text), ttft=times[0] if times else None, itl=itl, total=time.perf_counter() - t0, chunks=len(times)
    )


def run_batch(url, reqs, stagger=0.0):
    """reqs: list of (name, prompt, max_tokens, params); concurrent, optional stagger between submissions."""
    out = {}
    threads = []

    def go(name, prompt, mt, params):
        out[name] = stream(url, prompt, mt, **params)
        out[name].update(prompt_len=len(prompt), max_tokens=mt, params=params)

    for i, (name, prompt, mt, params) in enumerate(reqs):
        t = threading.Thread(target=go, args=(name, prompt, mt, params))
        t.start()
        threads.append(t)
        if stagger:
            time.sleep(stagger)
    for t in threads:
        t.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_path = Path(a.events)
    n_ev0 = sum(1 for _ in open(ev_path))
    greedy = dict(temperature=0.0)
    sampled = dict(temperature=0.7, top_k=20, top_p=0.9)
    long5 = _ids(5000, 5)
    phases = []
    t_start = time.perf_counter()

    # 1. first encounters of unseen non-aligned lengths (each alone)
    for n in (129, 130, 131):
        phases.append(("unseen_" + str(n), run_batch(a.url, [(f"u{n}", _ids(n, n), 24, greedy)])))
    # 2. chunked prompts (3000: chunks 2048 + 952; 5000: 2048 + 2048 + 904) arriving while three requests decode
    phases.append(
        (
            "chunked_during_decode",
            run_batch(
                a.url,
                [
                    ("d77", _ids(77, 77), 96, greedy),
                    ("d300", _ids(300, 300), 96, sampled),
                    ("d1500", _ids(1500, 1500), 96, greedy),
                    ("c3000", _ids(3000, 3000), 32, greedy),
                    ("c5000", long5, 32, greedy),
                ],
                stagger=0.4,
            ),
        )
    )
    # 3. repeats of known lengths in other compositions / orders (greedy determinism across batches)
    phases.append(
        (
            "repeat_a",
            run_batch(
                a.url,
                [
                    ("u129", _ids(129, 129), 24, greedy),
                    ("d77", _ids(77, 77), 24, greedy),
                    ("u131", _ids(131, 131), 24, greedy),
                ],
            ),
        )
    )
    phases.append(
        (
            "repeat_b",
            run_batch(a.url, [("u131", _ids(131, 131), 24, greedy), ("u129", _ids(129, 129), 24, greedy)], stagger=0.2),
        )
    )
    # 4. sampling-mode switches: greedy -> sampled -> host (logprobs) -> greedy (same request as phase 1)
    phases.append(
        ("switch_sampled", run_batch(a.url, [("s1", _ids(200, 9), 24, sampled), ("g1", _ids(129, 129), 24, greedy)]))
    )
    phases.append(
        (
            "switch_host_logprobs",
            run_batch(
                a.url,
                [("h1", _ids(150, 10), 16, dict(temperature=0.0, logprobs=3)), ("g2", _ids(130, 130), 24, greedy)],
            ),
        )
    )
    phases.append(("switch_back_greedy", run_batch(a.url, [("u130", _ids(130, 130), 24, greedy)])))
    # 5. prefix-cache hit: 4096 tokens shared with c5000 + a new tail (chunk start lands mid-prompt)
    phases.append(("prefix_hit", run_batch(a.url, [("p4200", long5[:4096] + _ids(104, 41), 24, greedy)])))
    phases.append(("prefix_control", run_batch(a.url, [("p4200", long5[:4096] + _ids(104, 41), 24, greedy)])))
    # 6. device penalties / checkpoint defaults / 512-grid hit
    pen_greedy = dict(temperature=0.0, repetition_penalty=1.3, presence_penalty=0.5, frequency_penalty=0.3)
    pen_sampled = dict(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1, frequency_penalty=0.2)
    genconfig = dict(temperature=0.1, top_k=50, repetition_penalty=1.05)  # checkpoint generation_config: host sampled
    phases.append(("penalized_alone", run_batch(a.url, [("pg", _ids(140, 140), 48, pen_greedy)])))
    phases.append(
        (
            "penalized_mixed",
            run_batch(
                a.url,
                [
                    ("ps", _ids(160, 160), 64, pen_sampled),
                    ("pg", _ids(140, 140), 48, pen_greedy),
                    ("u129", _ids(129, 129), 24, greedy),
                    ("late", _ids(90, 90), 24, greedy),  # admitted while the penalized rows decode
                ],
                stagger=0.15,
            ),
        )
    )
    phases.append(("genconfig_host", run_batch(a.url, [("gc", _ids(120, 120), 24, genconfig)])))
    phases.append(("prefix_hit_512", run_batch(a.url, [("p2700", long5[:2624] + _ids(76, 42), 24, greedy)])))
    phases.append(("prefix_control_512", run_batch(a.url, [("p2700", long5[:2624] + _ids(76, 42), 24, greedy)])))
    wall = time.perf_counter() - t_start

    # ---- analysis
    reqs = {}
    for ph, res in phases:
        for name, r in res.items():
            reqs.setdefault(name, []).append(dict(phase=ph, **r))
    determinism = {}
    for name, runs in reqs.items():
        g = [r for r in runs if r["params"].get("temperature") == 0.0 and "logprobs" not in r["params"]]
        if len(g) > 1:
            m = min(len(r["text"]) for r in g)
            determinism[name] = all(r["text"][:m] == g[0]["text"][:m] for r in g)
    # u129 / u131 / u130 also ran with other max_tokens: compare common prefixes only (done above)
    ev = [e for e in map(json.loads, open(ev_path)) if e.get("kind") != "bench_mark"][n_ev0:]
    before = [e for e in map(json.loads, open(ev_path)) if e.get("kind") != "bench_mark"][:n_ev0]
    ready = [e for e in before if e["kind"] == "traces_ready"]
    cap0 = (before[-1]["captures"], before[-1]["prefill_captures"]) if before else None
    cap1 = (ev[-1]["captures"], ev[-1]["prefill_captures"]) if ev else cap0
    pre = [e for e in ev if e["kind"] == "prefill"]
    dec = [e for e in ev if e["kind"] == "decode"]
    chunk_rows = [
        dict(start=r[0], end=r[1], slot=r[2], recompute_from=r[3], device_sampling=e["device_sampling"])
        for e in pre
        for r in e["rows"]
    ]
    ttfts = [r["ttft"] for runs in reqs.values() for r in runs if r["ttft"] is not None]
    itls = [x for runs in reqs.values() for r in runs for x in r["itl"]]
    tokens = sum(r["chunks"] for runs in reqs.values() for r in runs)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None

    report = dict(
        wall_s=wall,
        requests=sum(len(v) for v in reqs.values()),
        completed_tokens=tokens,
        completed_request_throughput_tok_s=tokens / wall,
        ttft_s=dict(p50=pct(ttfts, 50), p95=pct(ttfts, 95), p99=pct(ttfts, 99), max=max(ttfts)),
        itl_s=dict(p50=pct(itls, 50), p95=pct(itls, 95), p99=pct(itls, 99), max=max(itls)),
        decode_pauses_over_250ms=sum(1 for x in itls if x > 0.25),
        determinism_greedy=determinism,
        traces_before_workload=dict(captures=cap0, inventory=ready[-1] if ready else None),
        traces_after_workload=dict(captures=cap1),
        recaptures=None if cap0 is None else (cap1[0] - cap0[0], cap1[1] - cap0[1]),
        prefill_chunk_rows=chunk_rows,
        continuation_starts=sorted({c["start"] for c in chunk_rows if c["start"] > 0}),
        recompute_starts=sorted({c["recompute_from"] for c in chunk_rows if c["start"] > 0}),
        decode_steps=len(dec),
        decode_modes=Counter(
            f"rows={e['rows']} {'device' if e['device_sampling'] else 'host'} "
            f"{'reload' if e['reload_inputs'] else ('page' if e['reload_page_table'] else 'resident')}"
            f"{' reset' if e['reset_sampling_state'] else ''}{' remap' if e['remap'] else ''}"
            f"{' pen' if e.get('penalized') else ''}"
            for e in dec
        ),
        penalized_decode_steps=sum(bool(e.get("penalized")) for e in dec),
        penalty_history_resets=sum(bool(e.get("histories")) for e in dec),
        penalty_stale_replays=(ev[-1].get("penalty_stale_replays", 0) - ev[0].get("penalty_stale_replays", 0))
        if ev
        else 0,
        seed_readbacks=(ev[-1].get("seed_readbacks", 0) - ev[0].get("seed_readbacks", 0)) if ev else 0,
        snapshot_stats=ev[-1]["snapshots"] if ev else None,
        per_request={k: [{kk: vv for kk, vv in r.items() if kk not in ("itl",)} for r in v] for k, v in reqs.items()},
    )
    report["pass"] = bool(
        report["recaptures"] == (0, 0)
        and all(determinism.values())
        and report["continuation_starts"]
        and report["penalty_stale_replays"] == 0
        and report["seed_readbacks"] == 0
    )
    (out / "mixed_shape_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "pass",
                    "recaptures",
                    "continuation_starts",
                    "recompute_starts",
                    "determinism_greedy",
                    "ttft_s",
                    "itl_s",
                    "decode_pauses_over_250ms",
                    "decode_modes",
                    "penalized_decode_steps",
                    "penalty_stale_replays",
                    "seed_readbacks",
                    "snapshot_stats",
                )
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
