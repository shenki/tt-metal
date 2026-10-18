# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving parity on the shared qualitative suite (chat template, same rendered token ids as the full-model stages).

Pure greedy (temperature 0, repetition_penalty 1.0 sent explicitly, so the checkpoint generation_config's 1.05 penalty
does not apply) through ``/v1/completions`` with the rendered chat-template token ids, 128 tokens: first divergence
against the HF greedy control (``doc/full_model/qualitative/hf_greedy.json``, same rendering, repetition_penalty 1.0) and
against the selected-config standalone TT run (``doc/datatype_sweep/qualitative``); once one request at a time and once
all six concurrently (batch invariance).

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_qual_parity --out <dir>
"""

import argparse
import json
import threading
from pathlib import Path

import requests

DOC = Path(__file__).resolve().parents[1] / "doc"
MODEL = "LiquidAI/LFM2.5-24B-A2B"


def complete(url, ids, n):
    body = dict(model=MODEL, prompt=ids, max_tokens=n, temperature=0.0, repetition_penalty=1.0, return_token_ids=True)
    r = requests.post(f"{url}/v1/completions", json=body, timeout=1800)
    r.raise_for_status()
    c = r.json()["choices"][0]
    return dict(text=c["text"], token_ids=c.get("token_ids"))


def first_div(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((DOC / "datatype_sweep" / "qualitative" / "prompts.json").read_text())
    hf = {
        p["id"]: p["token_ids"] for p in json.loads((DOC / "full_model" / "qualitative" / "hf_greedy.json").read_text())
    }
    tt_path = DOC / "datatype_sweep" / "qualitative" / "tt_greedy.json"
    tt = {int(p["id"]): p["token_ids"] for p in json.loads(tt_path.read_text())} if tt_path.exists() else {}
    solo = {p["id"]: complete(a.url, p["token_ids"], a.tokens) for p in prompts}
    batch, th = {}, []
    for p in prompts:
        t = threading.Thread(target=lambda p=p: batch.__setitem__(p["id"], complete(a.url, p["token_ids"], a.tokens)))
        t.start()
        th.append(t)
    for t in th:
        t.join()
    rows = []
    for p in prompts:
        i = p["id"]
        s, b = solo[i]["token_ids"], batch[i]["token_ids"]
        rows.append(
            dict(
                id=i,
                prompt=p["prompt"],
                vllm_vs_hf_first_divergence=first_div(s, hf[i]),
                vllm_vs_tt_standalone_first_divergence=first_div(s, tt[i]) if i in tt else "n/a",
                solo_vs_batched_identical=s == b,
                vllm_len=len(s),
                vllm_text=solo[i]["text"],
            )
        )
    (out / "vllm_greedy_parity.json").write_text(json.dumps(dict(rows=rows, solo=solo, batched=batch), indent=1))
    for r in rows:
        print(
            r["id"],
            "hf",
            r["vllm_vs_hf_first_divergence"],
            "tt",
            r["vllm_vs_tt_standalone_first_divergence"],
            "batch==solo",
            r["solo_vs_batched_identical"],
            repr(r["vllm_text"][:70]),
        )


if __name__ == "__main__":
    main()
