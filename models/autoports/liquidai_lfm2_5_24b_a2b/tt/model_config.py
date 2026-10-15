# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static model contract for LiquidAI/LFM2.5-24B-A2B (HF ``Lfm2MoeForCausalLM``).

All values are read from the HF config; nothing here invents or patches config values.
"""

from dataclasses import dataclass
from pathlib import Path

HF_MODEL_ID = "LiquidAI/LFM2.5-24B-A2B"
HF_REVISION = "1769516ac588c4c558089c94f27d96b0dc67b183"

TILE = 32


@dataclass(frozen=True)
class Lfm2MoeArgs:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    layer_types: tuple
    num_dense_layers: int
    dense_intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    use_expert_bias: bool
    conv_kernel: int
    conv_bias: bool
    norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf_config(cls, hf_config):
        rope = hf_config.rope_parameters
        if rope["rope_type"] != "default":
            raise ValueError(f"unsupported rope_type {rope['rope_type']}")
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        return cls(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            num_layers=hf_config.num_hidden_layers,
            layer_types=tuple(hf_config.layer_types),
            num_dense_layers=hf_config.num_dense_layers,
            dense_intermediate_size=hf_config.intermediate_size,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            num_experts=hf_config.num_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            norm_topk_prob=hf_config.norm_topk_prob,
            routed_scaling_factor=float(hf_config.routed_scaling_factor),
            use_expert_bias=hf_config.use_expert_bias,
            conv_kernel=hf_config.conv_L_cache,
            conv_bias=hf_config.conv_bias,
            norm_eps=hf_config.norm_eps,
            rope_theta=float(rope["rope_theta"]),
            max_position_embeddings=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            pad_token_id=hf_config.pad_token_id,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )

    def is_attention_layer(self, layer_idx):
        return self.layer_types[layer_idx] == "full_attention"

    def is_moe_layer(self, layer_idx):
        return layer_idx >= self.num_dense_layers

    def layer_kind(self, layer_idx):
        mixer = "attn" if self.is_attention_layer(layer_idx) else "conv"
        ffn = "moe" if self.is_moe_layer(layer_idx) else "dense"
        return f"{mixer}_{ffn}"


def load_hf_config(model_path=None):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path or HF_MODEL_ID, revision=None if model_path else HF_REVISION)


def resolve_model_path():
    """Local snapshot directory of the pinned revision (downloads metadata only if missing)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL_ID, revision=HF_REVISION, allow_patterns=["*.json", "*.jinja"]))
