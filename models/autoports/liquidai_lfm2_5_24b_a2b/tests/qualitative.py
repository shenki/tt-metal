# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared qualitative prompt suite (``readiness_check/vllm_prompts.txt``) through the HF reference and the TT generator.

Prompt format: the checkpoint tokenizer has a chat template -> chat mode, ``apply_chat_template(add_generation_prompt=True)``
(the template ends the generation prompt with ``<think>``).  Controls / runs (128 new tokens, stop at ``<|im_end|>``):

* ``hf_greedy``: HF ``generate`` on CPU, bf16, ``do_sample=False`` and ``repetition_penalty=1.0`` (the checkpoint's
  generation_config sets 1.05, which HF applies even without sampling: pure greedy is the comparable control);
* ``tt_greedy``: TT token-out decode (split sampling, k=1);
* ``tt_topk``: TT sampled decode (top_k=20, top_p=0.9, temperature=0.7, seed 1234), a sampled-mode smoke.

CPU part:   python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.qualitative --hf
TT part:    readiness_driver --runs qual  (calls ``run_tt(gen)``)
Artifacts:  doc/full_model/qualitative/{qualitative_prompt_format.json, prompts.json, hf_greedy.json, tt_*.json}
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

# stage evidence dir (LFM2_QUAL_OUT); the HF control is reused from stage 6 when absent here (same model / revision /
# tokenizer / chat template / prompt suite: qualitative_prompt_format.json)
OUT = Path(
    os.environ.get("LFM2_QUAL_OUT") or Path(__file__).resolve().parents[1] / "doc" / "full_model" / "qualitative"
)
HF_CONTROL_FALLBACK = Path(__file__).resolve().parents[1] / "doc" / "full_model" / "qualitative" / "hf_greedy.json"
GEN = 128


def load_prompts():
    from readiness_check import __file__ as rc

    text = (Path(rc).parent / "vllm_prompts.txt").read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def render(tokenizer, prompts):
    out = []
    for i, p in enumerate(prompts):
        txt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        out.append({"id": i, "prompt": p, "rendered": txt, "token_ids": ids})
    return out


def write_format(tokenizer):
    from ..tt.model_config import HF_MODEL_ID, HF_REVISION

    tmpl = tokenizer.chat_template or ""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative_prompt_format.json").write_text(
        json.dumps(
            {
                "hf_model_id": HF_MODEL_ID,
                "hf_revision": HF_REVISION,
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_present": bool(tmpl),
                "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                "prompt_mode": "chat",
                "rendering": "tokenizer.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True)",
                "prompt_source": "readiness_check/vllm_prompts.txt (blank-line separated)",
                "max_new_tokens": GEN,
                "stop": "<|im_end|> (eos 124900)",
                "runs": {
                    "hf_greedy": "HF generate, CPU bf16, do_sample=False, repetition_penalty=1.0",
                    "tt_greedy": "TT generate, split sampling k=1 p=0 temp=1 (token-out traced decode)",
                    "tt_topk": "TT generate, top_k=20 top_p=0.9 temperature=0.7 seed=1234",
                },
            },
            indent=1,
        )
        + "\n"
    )


def run_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..tt.model_config import HF_MODEL_ID, HF_REVISION

    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, revision=HF_REVISION)
    write_format(tok)
    items = render(tok, load_prompts())
    (OUT / "prompts.json").write_text(json.dumps(items, indent=1) + "\n")
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, revision=HF_REVISION, dtype=torch.bfloat16).eval()
    res = []
    for it in items:
        ids = torch.tensor([it["token_ids"]])
        with torch.no_grad():
            g = model.generate(
                ids,
                max_new_tokens=GEN,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new = g[0, ids.shape[1] :].tolist()
        res.append({"id": it["id"], "token_ids": new, "text": tok.decode(new)})
        print(f"[hf {it['id']}] {res[-1]['text'][:200]!r}", flush=True)
    (OUT / "hf_greedy.json").write_text(json.dumps(res, indent=1) + "\n")


def run_tt(gen):
    write_format(gen.tokenizer)
    items = render(gen.tokenizer, load_prompts())
    (OUT / "prompts.json").write_text(json.dumps(items, indent=1) + "\n")
    summary = {}
    for name, kw in (
        ("tt_greedy", {}),
        ("tt_topk", dict(top_k=20, top_p=0.9, temperature=0.7, seed=1234)),
    ):
        res = []
        for it in items:
            gen.reset()
            new = gen.generate(it["token_ids"], GEN, **kw)
            res.append({"id": it["id"], "token_ids": new, "text": gen.tokenizer.decode(new)})
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
        summary[name] = [r["text"][:120] for r in res]
    hf_path = OUT / "hf_greedy.json"
    if not hf_path.exists():
        hf_path = HF_CONTROL_FALLBACK
    if hf_path.exists():
        hf = {r["id"]: r["token_ids"] for r in json.loads(hf_path.read_text())}
        tt = {r["id"]: r["token_ids"] for r in json.loads((OUT / "tt_greedy.json").read_text())}
        agree = {}
        for i in tt:
            a, b = hf.get(i, []), tt[i]
            n = min(len(a), len(b))
            agree[i] = {
                "first_divergence": next((k for k in range(n) if a[k] != b[k]), None),
                "hf_len": len(a),
                "tt_len": len(b),
            }
        (OUT / "hf_tt_greedy_agreement.json").write_text(json.dumps(agree, indent=1) + "\n")
        summary["agreement"] = agree
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    if ap.parse_args().hf:
        run_hf()
