# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record per-tensor stats (shape, dtype, mean, std) of the real checkpoint for the decoder
layers the tests cover, so CI tests can synthesize same-shape weights without the download.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.collect_weight_stats

Only the safetensors shards that hold the requested layers (and the embedding) are opened.
Layer indices: 0 = conv+dense, 2 = attention+MoE, 3 = conv+MoE (every other layer reuses the
stats of the first layer of its kind).
"""

import json

from ..tt.model_config import HF_MODEL_ID, HF_REVISION, Lfm2MoeArgs
from ..tt.weights import load_layer_state_dict, load_tensors
from .harness import STATS_PATH, hf_config, local_model_path

LAYERS = (0, 2, 3)


def main():
    path = local_model_path()
    assert path is not None, "download the pinned checkpoint first"
    args = Lfm2MoeArgs.from_hf_config(hf_config())
    out = {"model": HF_MODEL_ID, "revision": HF_REVISION, "layers": {}, "layer_kind_source": {}}
    for li in LAYERS:
        sd = load_layer_state_dict(path, li)
        out["layers"][str(li)] = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "mean": float(v.float().mean()),
                "std": float(v.float().std()),
            }
            for k, v in sd.items()
        }
    first_of_kind = {args.layer_kind(li): li for li in reversed(LAYERS)}
    for li in range(args.num_layers):
        out["layer_kind_source"][str(li)] = str(first_of_kind[args.layer_kind(li)])
    emb = load_tensors(path, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    out["embed_std"] = float(emb.float().std())
    out["embed_mean"] = float(emb.float().mean())
    with open(STATS_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {STATS_PATH}")


if __name__ == "__main__":
    main()
