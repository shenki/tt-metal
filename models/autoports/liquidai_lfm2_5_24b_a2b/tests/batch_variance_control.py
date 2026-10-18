# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone (no vLLM) control for greedy batch variance on the shared qualitative suite.

Each of the six chat-rendered prompts is decoded alone (slot 0, decode bucket 1) and all six together (slots 0..5,
bucket 8) with the generator's low-level API (pure greedy split sampling, 128 tokens).  Reports the first divergence
solo vs batched per prompt and both against the HF greedy control, to compare with the vLLM solo / concurrent runs
(``doc/vllm_integration/qualitative_parity``).

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.batch_variance_control --out <dir>
"""

import argparse
import json
from pathlib import Path

import torch

DOC = Path(__file__).resolve().parents[1] / "doc"
N = 128


def first_div(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None)


def run(gen, prompts, slots):
    """Prefill ``prompts`` into ``slots`` (one call), then N - 1 greedy decode steps; returns token lists."""
    gen.reset()
    gen.set_sampling_params(top_k=1)
    L = max(len(p) for p in prompts)
    toks = torch.zeros(len(prompts), L, dtype=torch.int64)
    for i, p in enumerate(prompts):
        toks[i, : len(p)] = torch.tensor(p)
    first = gen.prefill_forward(
        toks, page_table=gen.slot_table[slots], prompt_lens=[len(p) for p in prompts], slots=slots
    )
    out = [[int(t)] for t in first]
    pos = torch.full((gen.B,), -1, dtype=torch.int64)
    ids = torch.zeros(gen.B, dtype=torch.int64)
    for i, (s, p) in enumerate(zip(slots, prompts)):
        pos[s], ids[s] = len(p), out[i][0]
    gen._refresh_table(gen.slot_table)
    nxt = gen.decode_forward(ids, pos, read_from_device=True)
    for _ in range(N - 1):
        for i, s in enumerate(slots):
            out[i].append(int(nxt[s]))
        if len(out[0]) >= N:
            break
        nxt = gen.decode_forward(read_from_device=True)
    return [o[:N] for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import ttnn
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import build_generator, open_mesh

    prompts = json.loads((DOC / "datatype_sweep" / "qualitative" / "prompts.json").read_text())
    hf = {
        p["id"]: p["token_ids"] for p in json.loads((DOC / "full_model" / "qualitative" / "hf_greedy.json").read_text())
    }
    ids = [p["token_ids"] for p in prompts]
    mesh = open_mesh(768 << 20)
    try:
        gen = build_generator(mesh_device=mesh)
        solo = [run(gen, [p], [0])[0] for p in ids]
        batched = run(gen, ids, list(range(len(ids))))
        rows = []
        for i, p in enumerate(prompts):
            rows.append(
                dict(
                    id=p["id"],
                    solo_vs_batched=first_div(solo[i], batched[i]),
                    solo_vs_hf=first_div(solo[i], hf[p["id"]]),
                    batched_vs_hf=first_div(batched[i], hf[p["id"]]),
                )
            )
            print(rows[-1])
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "standalone_batch_variance.json").write_text(
            json.dumps(dict(rows=rows, solo=solo, batched=batched), indent=1)
        )
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
