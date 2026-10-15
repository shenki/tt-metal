# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only HF reference: the installed ``transformers`` ``Lfm2MoeDecoderLayer`` with given weights.

The layer is causal (ShortConv is a causal depthwise conv, attention is causal SDPA), so the
output at position ``t`` of a full-sequence forward equals what a prefill/decode stack must
produce for that token.  We therefore compare TT prefill rows and TT decode steps against one
uncached full-sequence HF forward.
"""

import torch

from ..tt.weights import to_hf_module_state_dict


def build_hf_layer(hf_config, layer_idx, layer_sd, dtype=torch.float32):
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeDecoderLayer, Lfm2MoeRotaryEmbedding

    cfg = hf_config.__class__.from_dict(hf_config.to_dict())
    cfg._attn_implementation = "sdpa"
    cfg._experts_implementation = "eager"
    layer = Lfm2MoeDecoderLayer(cfg, layer_idx)
    sd = to_hf_module_state_dict(layer_sd, cfg.num_experts)
    if "feed_forward.expert_bias" in sd:
        sd["feed_forward.expert_bias"] = sd["feed_forward.expert_bias"].float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("expert_bias")]
    assert not missing and not unexpected, (missing, unexpected)
    layer = layer.to(dtype).eval()
    if hasattr(layer.feed_forward, "expert_bias"):
        layer.feed_forward.expert_bias.data = sd["feed_forward.expert_bias"].float()
    rope = Lfm2MoeRotaryEmbedding(cfg)
    return layer, rope


@torch.no_grad()
def hf_layer_forward(layer, rope, x, start_pos=0):
    """x ``[B, S, H]`` -> ``[B, S, H]`` at positions ``start_pos ...`` (no cache, causal)."""
    pos = torch.arange(start_pos, start_pos + x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
    cos, sin = rope(x, pos)
    return layer(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
