# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for LFM2.5-24B-A2B.

The checkpoint stores experts per expert (``experts.{e}.w1/w3/w2``: gate/up/down, torch Linear
layout ``[out, in]``). ``load_layer_state_dict`` returns one decoder layer's tensors with the
``model.layers.{i}.`` prefix stripped; only the safetensors shards that hold the layer are opened.
"""

import json
from pathlib import Path

import torch


def _weight_map(model_path):
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def load_tensors(model_path, keys):
    from safetensors import safe_open

    weight_map = _weight_map(model_path)
    by_file = {}
    for k in keys:
        by_file.setdefault(weight_map[k], []).append(k)
    out = {}
    for fname, ks in by_file.items():
        with safe_open(str(Path(model_path) / fname), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out


def load_layer_state_dict(model_path, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in _weight_map(model_path) if k.startswith(prefix)]
    if not keys:
        raise KeyError(f"no tensors for layer {layer_idx}")
    return {k[len(prefix) :]: v for k, v in load_tensors(model_path, keys).items()}


def load_global_state_dict(model_path):
    keys = ["model.embed_tokens.weight", "model.embedding_norm.weight"]
    return {k[len("model.") :]: v for k, v in load_tensors(model_path, keys).items()}


def stacked_expert_weights(layer_sd, num_experts):
    """Per-expert checkpoint tensors -> (gate [E,I,H], up [E,I,H], down [E,H,I])."""
    gate = torch.stack([layer_sd[f"feed_forward.experts.{e}.w1.weight"] for e in range(num_experts)])
    up = torch.stack([layer_sd[f"feed_forward.experts.{e}.w3.weight"] for e in range(num_experts)])
    down = torch.stack([layer_sd[f"feed_forward.experts.{e}.w2.weight"] for e in range(num_experts)])
    return gate, up, down


def to_hf_module_state_dict(layer_sd, num_experts):
    """Checkpoint-layout layer dict -> ``Lfm2MoeDecoderLayer.state_dict()`` layout (fused experts)."""
    out = {}
    for k, v in layer_sd.items():
        if ".experts." not in k:
            out[k] = v
    if "feed_forward.gate.weight" in layer_sd:
        gate, up, down = stacked_expert_weights(layer_sd, num_experts)
        out["feed_forward.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
        out["feed_forward.experts.down_proj"] = down
        out["feed_forward.gate.weight"] = layer_sd["feed_forward.gate.weight"]
    return out
