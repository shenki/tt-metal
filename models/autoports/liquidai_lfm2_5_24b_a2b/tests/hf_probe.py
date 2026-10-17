# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF CPU reference for a (reduced) layer subset of the full model: embedding -> the given real layers (each with its own
layer index / kind) -> ``embedding_norm`` -> tied LM head.  With all 40 layers this is the HF model's logits; with
``[0, 2, 3]`` it is the reference of the reduced full-model probe (one layer of each kind, real terminal path)."""

import torch

from ..reference.hf_layer import build_hf_layer, hf_layer_forward
from ..tt.weights import load_global_state_dict, load_layer_state_dict


class HFProbe:
    def __init__(self, model_path, hf_config, layer_indices, dtype=torch.float32):
        g = load_global_state_dict(model_path)
        self.embed = g["embed_tokens.weight"].to(dtype)
        self.norm_w = g["embedding_norm.weight"].to(dtype)
        self.eps = hf_config.norm_eps
        self.layers = [build_hf_layer(hf_config, i, load_layer_state_dict(model_path, i), dtype) for i in layer_indices]
        self.dtype = dtype

    @torch.no_grad()
    def hidden(self, ids):
        """Final-norm input ``[S, H]`` for token ids ``[S]`` (positions 0..S-1, causal, no cache)."""
        h = self.embed[torch.as_tensor(ids).long()].unsqueeze(0)
        for layer, rope in self.layers:
            h = hf_layer_forward(layer, rope, h)
        return h[0]

    @torch.no_grad()
    def logits(self, ids):
        h = self.hidden(ids).float()
        var = h.pow(2).mean(-1, keepdim=True)
        hn = (h * torch.rsqrt(var + self.eps)).to(self.dtype) * self.norm_w
        return hn.float() @ self.embed.float().T  # [S, V]
