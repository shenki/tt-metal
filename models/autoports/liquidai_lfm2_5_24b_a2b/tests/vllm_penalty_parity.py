# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device penalties vs the plugin's host sampler on a running server: each penalized greedy request is sent twice, once
plainly (device penalties + device greedy) and once with ``logprobs=1`` (the plugin then samples on host with vLLM's
fp32 penalties over the same model logits).  Reports the first token divergence per prompt and penalty setting; the
device path applies the penalties in bf16, so late divergences on near-ties are expected, systematic early ones are
not.  Prompts: the shared qualitative suite (chat template token ids) and random-id prompts.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_penalty_parity --out <dir>
"""

import argparse
import json
import random
from pathlib import Path

import requests

DOC = Path(__file__).resolve().parents[1] / "doc"
MODEL = "LiquidAI/LFM2.5-24B-A2B"
SETTINGS = {
    "rep1.3_pres0.5_freq0.3": dict(repetition_penalty=1.3, presence_penalty=0.5, frequency_penalty=0.3),
    "rep1.05": dict(repetition_penalty=1.05),
    "freq1.0": dict(frequency_penalty=1.0),
}


def complete(url, ids, n, params, host):
    body = dict(model=MODEL, prompt=ids, max_tokens=n, temperature=0.0, return_token_ids=True, **params)
    if host:
        body["logprobs"] = 1
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
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    prompts = [
        (f"qual{p['id']}", p["token_ids"])
        for p in json.loads((DOC / "datatype_sweep" / "qualitative" / "prompts.json").read_text())
    ]
    rnd = random.Random(140)
    prompts.append(("random140", [rnd.randrange(1000, 60000) for _ in range(140)]))
    rows = []
    for name, params in SETTINGS.items():
        for pid, ids in prompts:
            dev = complete(a.url, ids, a.tokens, params, host=False)
            host = complete(a.url, ids, a.tokens, params, host=True)
            plain = complete(a.url, ids, a.tokens, dict(repetition_penalty=1.0), host=False)
            rows.append(
                dict(
                    setting=name,
                    prompt=pid,
                    device_vs_host_first_divergence=first_div(dev["token_ids"], host["token_ids"]),
                    device_vs_unpenalized_first_divergence=first_div(dev["token_ids"], plain["token_ids"]),
                    device_text=dev["text"],
                    host_text=host["text"],
                )
            )
            r = rows[-1]
            print(
                name,
                pid,
                "dev~host",
                r["device_vs_host_first_divergence"],
                "dev~plain",
                r["device_vs_unpenalized_first_divergence"],
                repr(dev["text"][:50]),
            )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "penalty_parity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
