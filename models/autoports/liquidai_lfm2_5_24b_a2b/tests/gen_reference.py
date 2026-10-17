# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``python -m readiness_check.generate`` with one compatibility shim, then a metadata sidecar.

transformers >= 5 returns a ``BatchEncoding`` from ``apply_chat_template(tokenize=True)``; the packaged
``readiness_check.generate._chat_or_plain_prompt_tokens`` iterates it as a list of ids and fails
(``int('input_ids')``).  The shim unwraps ``input_ids``; everything else is the packaged CLI unchanged.
The sidecar ``<output>.meta.json`` records what the ``.refpt`` metadata does not (revision, tokenizer,
chat-template hash, prompt source, command, versions) so a later stage can prove an exact match.

Usage (same arguments as the packaged CLI):
    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.gen_reference --hf-model ... --output X.refpt
"""

import hashlib
import json
import sys
from pathlib import Path

from readiness_check import generate as G


def _tokens(tokenizer, prompt_text, *, chat_template):
    if chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=True
        )
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
    else:
        ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not ids:
        raise ValueError("Prompt tokenization produced no tokens")
    return [int(t) for t in ids]


def _arg(name, default=None):
    argv = sys.argv[1:]
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    G._chat_or_plain_prompt_tokens = _tokens
    G._main()
    import torch
    import transformers
    from transformers import AutoTokenizer

    from ..tt.model_config import HF_MODEL_ID, HF_REVISION

    hf_model = _arg("--hf-model")
    tok = AutoTokenizer.from_pretrained(hf_model)
    tmpl = getattr(tok, "chat_template", None) or ""
    meta = {
        "hf_model_id": hf_model,
        "hf_revision_pinned": HF_REVISION if hf_model == HF_MODEL_ID else None,
        "tokenizer_class": type(tok).__name__,
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
        "chat_template": "--chat-template" in sys.argv,
        "prompt_source": _arg("--prompt-source"),
        "aime24_prompt_index": int(_arg("--aime24-prompt-index", 0)),
        "gen_len": int(_arg("--gen-len", 100)),
        "top_k": int(_arg("--top-k", 100)),
        "command": "python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.gen_reference " + " ".join(sys.argv[1:]),
        "shim": "apply_chat_template BatchEncoding -> input_ids (transformers >= 5)",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    out = Path(_arg("--output"))
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
