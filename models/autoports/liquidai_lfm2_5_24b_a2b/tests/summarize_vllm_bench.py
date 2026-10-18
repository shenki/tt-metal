# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize one stage-10 benchmark set (session_state/scratchpad_tools/bench_set.sh output): per workload the
vllm bench serve metrics plus the adapter's step attribution from the LFM2_VLLM_EVENTS log, windowed by the set's
``bench_mark`` records (device- vs host-sampled decode steps, penalized replays, history rebuilds, readbacks, mean
adapter time per decode / prefill call).

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.summarize_vllm_bench <outdir> [--compare <before outdir>]
"""

import argparse
import json
from pathlib import Path

WORKLOADS = [
    "greedy_128_128_1",
    "greedy_128_128_n5",
    "k20_128_128_n5",
    "k20_rep105_128_128_n5",
    "genconfig_128_128_n5",
    "ci_100_100_32",
    "ci_k20_rep105_100_100_32",
    "ci_genconfig_100_100_32",
]


def _events(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _window(events, name):
    t0 = next((e["t"] for e in events if e.get("kind") == "bench_mark" and e["label"] == f"start:{name}"), None)
    t1 = next((e["t"] for e in events if e.get("kind") == "bench_mark" and e["label"] == f"end:{name}"), None)
    if t0 is None or t1 is None:
        return []
    return [e for e in events if t0 <= e["t"] <= t1 and e.get("kind") in ("decode", "prefill")]


def _attr(win):
    dec = [e for e in win if e["kind"] == "decode"]
    pre = [e for e in win if e["kind"] == "prefill"]
    first, last = (win[0], win[-1]) if win else ({}, {})
    delta = lambda k: int(last.get(k, 0)) - int(first.get(k, 0)) if win else 0  # noqa: E731
    mean = lambda xs: round(sum(xs) / len(xs) * 1000, 3) if xs else None  # noqa: E731
    return dict(
        decode_steps=len(dec),
        decode_device=sum(bool(e.get("device_sampling")) for e in dec),
        decode_host=sum(not e.get("device_sampling") for e in dec),
        decode_penalized=sum(bool(e.get("penalized")) for e in dec),
        decode_resets=sum(bool(e.get("reset_sampling_state")) for e in dec),
        decode_histories=sum(bool(e.get("histories")) for e in dec),
        prefill_calls=len(pre),
        prefill_device=sum(bool(e.get("device_sampling")) for e in pre),
        seed_readbacks=delta("seed_readbacks"),
        token_readbacks=delta("token_readbacks"),
        penalty_stale_replays=delta("penalty_stale_replays"),
        mean_decode_call_ms=mean([e["dur_s"] for e in dec if "dur_s" in e]),
        mean_prefill_call_ms=mean([e["dur_s"] for e in pre if "dur_s" in e]),
    )


def summarize(out):
    out = Path(out)
    events = _events(out / "events.jsonl")
    rows = {}
    for name in WORKLOADS:
        f = out / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        tpot = r.get("mean_tpot_ms")
        rows[name] = dict(
            completed=r.get("completed"),
            ttft_p50_ms=r.get("median_ttft_ms"),
            ttft_p99_ms=r.get("p99_ttft_ms"),
            tpot_mean_ms=tpot,
            tpot_p50_ms=r.get("median_tpot_ms"),
            itl_p50_ms=r.get("median_itl_ms"),
            itl_p99_ms=r.get("p99_itl_ms"),
            decode_t_s_u=round(1000.0 / tpot, 2) if tpot else None,
            output_tok_s=r.get("output_throughput"),
            **_attr(_window(events, name)),
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    rows = summarize(a.out)
    base = summarize(a.compare) if a.compare else {}
    (Path(a.out) / "summary.json").write_text(json.dumps(dict(rows=rows, compare=a.compare, before=base), indent=1))
    keys = ["ttft_p50_ms", "tpot_mean_ms", "decode_t_s_u", "itl_p99_ms", "output_tok_s"]
    for name, r in rows.items():
        b = base.get(name, {})
        cells = [
            f"{k}={r[k]:.2f}" + (f" (before {b[k]:.2f})" if b.get(k) is not None else "") for k in keys if r.get(k)
        ]
        print(name, " ".join(cells))
        print(
            "   steps dev/host/pen",
            r["decode_device"],
            r["decode_host"],
            r["decode_penalized"],
            "resets",
            r["decode_resets"],
            "prefill dev",
            f"{r['prefill_device']}/{r['prefill_calls']}",
            "readbacks seed/token",
            r["seed_readbacks"],
            r["token_readbacks"],
            "stale",
            r["penalty_stale_replays"],
            "call ms dec/pre",
            r["mean_decode_call_ms"],
            r["mean_prefill_call_ms"],
        )


if __name__ == "__main__":
    main()
