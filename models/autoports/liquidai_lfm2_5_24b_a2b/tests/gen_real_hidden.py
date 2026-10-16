# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real decoder-layer inputs for deep-layer tests: tokenize text from the Tale of Two Cities corpus,
run the HF reference layers 0..L-1 (fp32, CPU, one layer resident at a time) and save the hidden
state entering every layer in ``LAYERS_OUT``.  Embedding-row inputs (``harness.layer_inputs``) are
out of distribution for deep layers (the residual stream grows through the stack).

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.gen_real_hidden  # -> generated/lfm2_real_hidden.pt
"""

import bz2
from pathlib import Path

import torch

from ..reference.hf_layer import build_hf_layer, hf_layer_forward
from .harness import hf_config, layer_state_dict, local_model_path
from .test_functional_decoder import _real_embed

OUT = Path("generated/lfm2_real_hidden.pt")
CORPUS = Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
N_PROMPTS, N_TOKENS = 4, 400  # each prompt: N_TOKENS tokens from a different offset of the book
LAYERS_OUT = list(range(1, 40))


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_model_path())
    text = bz2.open(CORPUS, "rt").read()
    ids = tok(text[:200_000], add_special_tokens=False)["input_ids"]
    prompts = [ids[i * 9000 : i * 9000 + N_TOKENS] for i in range(N_PROMPTS)]
    cfg = hf_config()
    embed = _real_embed()
    x = torch.stack([embed[torch.tensor(p)].float() for p in prompts])  # [P, S, H]
    saved = {}
    for li in range(max(LAYERS_OUT)):
        layer, rope = build_hf_layer(cfg, li, layer_state_dict(li, True))
        x = hf_layer_forward(layer, rope, x)
        del layer
        if li + 1 in LAYERS_OUT:
            saved[li + 1] = x.to(torch.bfloat16).clone()
        print(f"layer {li} done, |x| rms {x.pow(2).mean().sqrt().item():.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts, "hidden": saved}, OUT)


if __name__ == "__main__":
    main()
