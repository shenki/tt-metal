# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF CPU control for the qualitative first-token differences of the selected config: HF bf16 next-token distribution
after the chat-template generation prompt (which ends with <think>) for every suite prompt; shows whether the TT first
token (stage 7 and stage 8) is a near-tie with HF's argmax.  Usage: python <this> -> table on stdout."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import HF_MODEL_ID, HF_REVISION

D = Path(__file__).resolve().parents[2]
prompts = json.loads((D / "full_model" / "qualitative" / "prompts.json").read_text())
tt8 = json.loads((D / "datatype_sweep" / "qualitative" / "tt_greedy.json").read_text())
tt7 = json.loads((D / "optimized_full_model" / "qualitative" / "tt_greedy.json").read_text())
tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, revision=HF_REVISION)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, revision=HF_REVISION, dtype=torch.bfloat16).eval()
for it, a, b in zip(prompts, tt7, tt8):
    with torch.no_grad():
        p = model(torch.tensor([it["token_ids"]])).logits[0, -1].float().softmax(-1)
    v, i = p.topk(5)
    top = [(tok.decode([int(x)]), round(float(y), 4)) for x, y in zip(i, v)]
    f7, f8 = a["token_ids"][0], b["token_ids"][0]
    print(
        f"prompt {it['id']}: HF top5 {top} | stage7 first {tok.decode([f7])!r} p={float(p[f7]):.4f} "
        f"| stage8 first {tok.decode([f8])!r} p={float(p[f8]):.4f}",
        flush=True,
    )
