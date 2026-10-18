# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-caching A/B through the server: chat requests sharing a long system prompt (agent-style traffic).

Run once against a server with automatic prefix caching (default) and once with ``--no-enable-prefix-caching``; the
greedy outputs (temperature 0, repetition_penalty 1.0) must be identical and the TTFT of the repeated-prefix requests
shows the reuse.  ``--compare <other report>`` checks the outputs against the other run.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_prefix_ab --label on --out <dir> [--compare <json>]
"""

import argparse
import json
import time
from pathlib import Path

import requests

MODEL = "LiquidAI/LFM2.5-24B-A2B"
ROOT = Path(__file__).resolve().parents[1]


def system_prompt(target_words, tag=""):
    """A long, realistic system prompt: the model's own documentation, repeated up to the target length."""
    text = []
    for p in sorted(ROOT.glob("doc/*/README.md")):
        text.append(p.read_text(errors="replace"))
    words = " ".join(text).split()
    out = []
    while len(out) < target_words:
        out += words
    head = f"[{tag}] " if tag else ""
    return (
        head + "You are an assistant for the LFM2 bring-up team. Reference material:\n" + " ".join(out[:target_words])
    )


QUESTIONS = [
    "Summarize in two sentences what the datatype sweep selected and why.",
    "What is the decode throughput reported for the optimized full model?",
    "List three limitations mentioned in the reference material.",
    "Which precision does the KV cache use?",
]


def chat(url, system, q, n=48):
    body = dict(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": q}],
        max_tokens=n,
        temperature=0.0,
        repetition_penalty=1.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    t0 = time.perf_counter()
    ttft, text, usage = None, [], None
    with requests.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            if c.get("choices") and c["choices"][0]["delta"].get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(c["choices"][0]["delta"]["content"])
    return dict(question=q, ttft_s=ttft, total_s=time.perf_counter() - t0, text="".join(text), usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=6000)
    ap.add_argument("--compare")
    ap.add_argument("--tag", default="", help="distinct opening text: a prefix no earlier request cached")
    ap.add_argument("--concurrent", action="store_true", help="owner keeps decoding (160 tokens) while the hits land")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sysp = system_prompt(a.words, a.tag)
    if a.concurrent:
        import threading

        res = [None] * len(QUESTIONS)
        owner = threading.Thread(target=lambda: res.__setitem__(0, chat(a.url, sysp, QUESTIONS[0], n=160)))
        owner.start()
        time.sleep(2.0)  # owner prefill (cold, ~1.4 s) done; it is decoding when the hits are scheduled
        th = [
            threading.Thread(target=lambda i=i: res.__setitem__(i, chat(a.url, sysp, QUESTIONS[i])))
            for i in range(1, len(QUESTIONS))
        ]
        for t in th:
            t.start()
        for t in th + [owner]:
            t.join()
    else:
        res = [chat(a.url, sysp, q) for q in QUESTIONS]
    rep = dict(label=a.label, system_words=a.words, requests=res)
    if a.compare:
        other = json.loads(Path(a.compare).read_text())
        rep["identical_to"] = other["label"]
        rep["identical"] = [x["text"] == y["text"] for x, y in zip(res, other["requests"])]
        rep["first_char_divergence"] = [
            next((k for k in range(min(len(x["text"]), len(y["text"]))) if x["text"][k] != y["text"][k]), None)
            for x, y in zip(res, other["requests"])
        ]
    (out / f"prefix_ab_{a.label}.json").write_text(json.dumps(rep, indent=1))
    for r in res:
        print(
            a.label,
            "prompt_tokens",
            (r["usage"] or {}).get("prompt_tokens"),
            "ttft",
            round(r["ttft_s"], 3),
            repr(r["text"][:60]),
        )
    if a.compare:
        print("identical", rep["identical"])


if __name__ == "__main__":
    main()
