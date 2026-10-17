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


def force_expert_swap(layer, pos, rank=3, alt_rank=4):
    """Router near-tie control: at sequence position ``pos`` replace the expert ranked ``rank`` by
    the one ranked ``alt_rank`` (routing weights renormalized as HF does).  Used to show that a
    low update PCC is exactly one top-k flip at a near-tied score, not a numerical drift."""
    ff = layer.feed_forward
    orig = ff.route_tokens_to_experts

    def route(logits):
        sel, _ = orig(logits)
        ranked = torch.topk(logits.sigmoid() + ff.expert_bias, alt_rank + 1, dim=-1).indices
        sel = sel.clone()
        sel[pos, rank] = ranked[pos, alt_rank]
        w = torch.gather(logits.sigmoid(), 1, sel)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        return sel, (w * ff.routed_scaling_factor).type_as(logits)

    ff.route_tokens_to_experts = route
    return layer


ROUTER_TIE_TAU = 1e-3  # fp32 biased-score gap between the k-th and (k+1)-th expert counted as a near-tie


@torch.no_grad()
def near_tie_alternatives(layer, rope, x, positions, tau=None):
    """For an MoE layer: references of ``x`` ``[S, H]`` with the k-th/(k+1)-th expert swapped at each
    position in ``positions`` whose biased router scores are within ``tau``.  A bf16 forward can
    legitimately pick either expert there (``torch.topk`` on fp32 is itself only one choice), so a
    test may accept the update PCC against such an alternative.  Returns ``[(pos, gap, ref_alt)]``."""
    ff = layer.feed_forward
    if not hasattr(ff, "gate"):
        return []
    tau = ROUTER_TIE_TAU if tau is None else tau  # module value at call time (the multichip tests widen it)
    cap = {}
    hook = ff.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach().reshape(-1, a[0].shape[-1])))
    hf_layer_forward(layer, rope, x.unsqueeze(0))
    hook.remove()
    k = ff.top_k
    s = torch.topk(torch.sigmoid(cap["h"] @ ff.gate.weight.T) + ff.expert_bias, k + 1, dim=-1).values
    gaps = s[:, k - 1] - s[:, k]
    alts = []
    prev = ff.__dict__.get("route_tokens_to_experts")  # an hf_patch already installed on the instance
    for p in positions:
        if gaps[p] < tau:
            force_expert_swap(layer, p, rank=k - 1, alt_rank=k)
            alts.append((p, gaps[p].item(), hf_layer_forward(layer, rope, x.unsqueeze(0))[0]))
            # back to the previous routing (class method, or the caller's patch)
            if prev is None:
                del ff.route_tokens_to_experts
            else:
                ff.route_tokens_to_experts = prev
    return alts
