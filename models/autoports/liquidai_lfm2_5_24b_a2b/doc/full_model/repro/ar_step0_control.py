# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF CPU control for the run_autoregressive step-0 divergence: HF's own next-token distribution at the end of the
packaged raw-text prompt (bf16 forward, forward with mask, generate's raw logits, fp32 forward) and the checkpoint's
generation_config (repetition_penalty).  Usage: python <this> (from tt-metal/, readiness runtime on PYTHONPATH)."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model_config import HF_MODEL_ID, HF_REVISION

meta = json.loads(
    (Path(__file__).resolve().parents[3] / "readiness_autoregressive" / "autoregressive_meta.json").read_text()
)
tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, revision=HF_REVISION)
print("generation_config:", GenerationConfig.from_pretrained(HF_MODEL_ID, revision=HF_REVISION).to_diff_dict())
ids = torch.tensor([meta["prompt_token_ids"]])
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, revision=HF_REVISION).eval()
show = lambda tag, lg: print(
    tag, [(tok.decode([i]), round(p, 4)) for p, i in zip(*[x.tolist() for x in lg.float().softmax(-1).topk(3)])]
)
with torch.no_grad():
    show(f"bf16 forward ({model.dtype}):", model(ids).logits[0, -1])
    show("bf16 forward with mask:", model(ids, attention_mask=torch.ones_like(ids)).logits[0, -1])
    g = model.generate(
        ids,
        max_new_tokens=2,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    show("generate raw step-0 logits:", g.logits[0][0])
    print("generate (checkpoint config, do_sample=False) tokens:", repr(tok.decode(g.sequences[0, ids.shape[1] :])))
    g1 = model.generate(
        ids, max_new_tokens=2, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id, repetition_penalty=1.0
    )
    print("generate repetition_penalty=1.0 tokens:", repr(tok.decode(g1[0, ids.shape[1] :])))
    print("TT tokens:", repr(tok.decode(meta["tt"]["token_ids"][:2])))
    show("fp32 forward:", model.float()(ids).logits[0, -1])
