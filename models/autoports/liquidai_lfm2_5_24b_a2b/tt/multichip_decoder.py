# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multichip TTNN decoder layer for LiquidAI/LFM2.5-24B-A2B (stage 4): 1x4 Blackhole mesh, TP = 4.

Single-chip baseline: ``optimized_decoder.OptimizedDecoder`` with ``DEFAULT_POLICY`` (stage 3).
Every module is that stage-3 module built on a chip-local slice of the layer (so all of its program
configs, shard specs and fused paths apply to the local shapes), with the per-chip weights placed by
``ShardTensorToMesh``.  Plan (``doc/multichip_decoder/README.md``):

* attention: Q/K/V column-parallel by heads (chip d owns q heads 8d..8d+7 and their GQA kv heads
  2d, 2d+1), local paged KV cache ``[blocks, 2, 64, 64]``, local SDPA, out_proj row-parallel;
* ShortConv: channel-parallel (512 channels per chip: in_proj B/C/x groups, taps and state sliced),
  out_proj row-parallel;
* dense MLP: w1/w3 column-parallel (2944 per chip), w2 row-parallel;
* MoE: router / top-k replicated (identical on every chip), every expert tensor-parallel over its
  intermediate dimension (384 of 1536 per chip), so each chip runs the same selected experts on a
  quarter of their weights (balanced for any routing; the compact bucket path reads b*K quarter
  experts per chip);
* each sublayer ends in a row-parallel partial ``[rows, 2048]``; one all-reduce restores the
  replicated residual: decode ``all_reduce_async`` on width-sharded L1, prefill ``ttnn.all_reduce``.

Residual contract: ``[1, 1, rows, 2048]`` bf16, identical on all chips.  Decode keeps it width-sharded
in L1 on the 8-core ffn-norm grid (``MeshPlan.resid_cores``) from layer input to layer output; prefill
keeps it DRAM interleaved.  The prefill/decode call signatures are the single-chip ones.
"""

from dataclasses import dataclass, replace

import torch

import ttnn

from .functional_decoder import DRAM, L1, TILE, _as_tt
from .fused_decoder import DEFAULT_FUSIONS, FusedDenseMLP, make_rope
from .model_config import Lfm2MoeArgs
from .optimized_decoder import (
    DEFAULT_POLICY,
    DTYPES,
    DramShardedLinear,
    OptimizedAttention,
    OptimizedDecoder,
    OptimizedDenseMLP,
    OptimizedMoE,
    OptimizedShortConv,
    width_shard_layout,
)
from .weights import stacked_expert_weights


@dataclass(frozen=True)
class MeshPlan:
    """Mesh / collective choices (bound at construction)."""

    tp: int = 4  # chips on the single mesh axis
    num_links: int = 2
    resid_cores: int = 8  # decode residual: width-sharded [32, 2048] on this many cores (= ffn_norm_cores)
    decode_reduce: str = "all_reduce_async"  # | "all_reduce" (composite RS + AG)
    prefill_reduce: str = "all_reduce"
    persistent_scratch: bool = True  # decode all_reduce_async scratch: one persistent L1 buffer per mesh (stage 5)
    moe_sharded_input: bool = True  # decode MoE experts read the width-sharded ffn-norm output (no S2I) (stage 5)
    # dense MLP decode: w1 | w3 as ONE packed DRAM-sharded matmul (cores, in0_block_w, readers); None = two matmuls
    dense_w13_packed: tuple = (32, 2, 3)
    # ShortConv decode chain on the out_proj input shards (8 cores x [32, 64]): B / C / x sliced straight from the
    # sharded in_proj output, state / taps / slot mask on the same shards, no S2I / I2S (stage 5 review P1)
    conv_sharded_chain: bool = True
    # ---- stage 8 (datatype sweep) ----
    # decode all-reduce payload dtype: the row-parallel partials (attention wo, ShortConv out_proj, dense w2 and, with
    # ccl_moe, the compact MoE expert sum) are written in this dtype and summed by all_reduce_async into a bf16
    # residual (doc/optimized_multichip_decoder/sweeps/decode_ccl_dtype_probe.log: bfp8 -2.1 us per all-reduce)
    ccl_dtype: str = "bf16"
    ccl_moe: bool = True

    def summary(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_MESH_PLAN = MeshPlan()

# Stage-3 DEFAULT_POLICY with the program configs that depend on per-chip shapes re-chosen for TP = 4
# (doc/multichip_decoder/work_log.md); dtypes / fidelities / fused paths unchanged.
MC_POLICY = replace(
    DEFAULT_POLICY,
    # decode DRAM-sharded projections (cores, in0_block_w, readers) for the per-chip shapes
    # (sweeps/local_ds_c*.log, 1-chip traced): every input / output grid is the 8-core residual grid
    ds_wqkv=(8, 8, 3),  # [2048, 768]: 10.5 us (16 cores / 1 reader: 15.0)
    ds_wo=(8, 2, 2),  # [512, 2048]: 9.1 us
    ds_conv_in=(8, 8, 2),  # [2048, 1536]: 11.5 us (32 cores / 3 readers: 30.0)
    ds_conv_out=(8, 2, 2),  # [512, 2048]: 9.1 us
    ds_dense_w13=(8, 8, 2),  # [2048, 2944] -> N padded to 3072: 21.3 us each (32 / 3: 30.7)
    ds_dense_w2_decode=(8, 6, 2),  # [3072 (K padded), 2048]: 19.9 us
    ds_dense_w2=(46, 2, 2),  # prefill [2944, 2048]
    # expert sparse_matmul 1D configs (grid_x, grid_y, in0_block_w, out_subblock_w) for I = 384 per chip
    # (sweeps/local_moe_*.jsonl): gate_up N = 2I = 24 tiles -> at most 24 cores, so a deep in0 block
    moe_gate_up_cfg=(8, 3, 32, 1),
    moe_down_cfg=(8, 8, 12, 1),  # K = 384 = 12 tiles
    # 4 slots: 30.5 us (in0_block_w 8: 42.9).  in0_block_w 64 (30.0 us) made the traced bucket-4 conv_moe
    # decode nondeterministic (2/2 runs; 32 / 8: 2/2 pass, logs/det_bisect.log) while the isolated op is
    # bitwise stable (repro/sparse_mm_idx_bw64_determinism.py): 256 KB of in0 CB per core next to live L1
    moe_gate_up_compact_cfg=(8, 3, 32, 1),
    moe_down_compact_cfg=(8, 8, 12, 1),  # 4 slots: 17.1 us
    moe_gate_up_prefill_cfg=(8, 3, 8, 1),
    moe_down_prefill_cfg=(8, 8, 12, 1),
    attn_residual_sharded=False,  # the multichip decoder owns the (sharded) residual
    # hybrid_routed_expert_moe sizes its L1 arena as all of L1 above the allocator base, so it OOMs
    # next to any L1 allocation -- here the CCL global semaphores (2.7 KB/bank); unified FFN instead
    dispatch_ffn="unified",
    dispatch_fidelity="hifi2",
    # ---- stage 5 (doc/optimized_multichip_decoder/work_log.md) ----
    moe_route_v3_l1=True,  # routing constants in L1 (only the unused hybrid op forbade it): -1.3 us / MoE layer
    attn_concat_persistent=True,  # persistent L1 concat buffer per bucket: -1.7 us / attention layer
    moe_down_small_bucket_cfg=(
        8,
        (8, 6, 12, 1),
    ),  # down at ~26 active experts (b8): 110.3 vs 124.3 us (b8 only measured)
    moe_compact_weight_before_down=True,  # weight the [n,32,384] activation, not the [n,32,2048] output
    moe_compact_out_sharded=True,  # expert sum written in the all-reduce layout: conv_moe b1 206.4 -> 204.6 us (with ^)
)


# ------------------------------------------------------------------------------------------------
# per-chip slicing of a layer state dict (HF checkpoint layout: Linear weights are [out, in])
# ------------------------------------------------------------------------------------------------


def local_state_dict(sd, args: Lfm2MoeArgs, d, tp):
    """Chip ``d``'s slice of a layer: column-parallel weights keep their output rows of chip d,
    row-parallel weights their input columns; everything else (norms, router, bias) unchanged."""
    out = dict(sd)
    rows = lambda t, n: t[d * n : (d + 1) * n]
    cols = lambda t, n: t[:, d * n : (d + 1) * n]
    HD = args.head_dim
    if "self_attn.q_proj.weight" in sd:
        nq, nkv = args.num_heads // tp, args.num_kv_heads // tp
        out["self_attn.q_proj.weight"] = rows(sd["self_attn.q_proj.weight"], nq * HD)
        out["self_attn.k_proj.weight"] = rows(sd["self_attn.k_proj.weight"], nkv * HD)
        out["self_attn.v_proj.weight"] = rows(sd["self_attn.v_proj.weight"], nkv * HD)
        out["self_attn.out_proj.weight"] = cols(sd["self_attn.out_proj.weight"], nq * HD)
    if "conv.in_proj.weight" in sd:
        H, c = args.hidden_size, args.hidden_size // tp
        w = sd["conv.in_proj.weight"]  # [3H, H] rows B | C | x
        out["conv.in_proj.weight"] = torch.cat([w[g * H + d * c : g * H + (d + 1) * c] for g in range(3)])
        out["conv.conv.weight"] = rows(sd["conv.conv.weight"], c)
        out["conv.out_proj.weight"] = cols(sd["conv.out_proj.weight"], c)
    if "feed_forward.w1.weight" in sd:
        i = sd["feed_forward.w1.weight"].shape[0] // tp
        out["feed_forward.w1.weight"] = rows(sd["feed_forward.w1.weight"], i)
        out["feed_forward.w3.weight"] = rows(sd["feed_forward.w3.weight"], i)
        out["feed_forward.w2.weight"] = cols(sd["feed_forward.w2.weight"], i)
    if "feed_forward.gate.weight" in sd:
        i = args.moe_intermediate_size // tp
        for e in range(args.num_experts):
            p = f"feed_forward.experts.{e}."
            out[p + "w1.weight"] = rows(sd[p + "w1.weight"], i)
            out[p + "w3.weight"] = rows(sd[p + "w3.weight"], i)
            out[p + "w2.weight"] = cols(sd[p + "w2.weight"], i)
    return out


def local_args(args: Lfm2MoeArgs, tp, part):
    """Module args at chip-local sizes for ``part``: "attn" (heads), "conv" (ShortConv channels:
    ``hidden_size`` is the channel count inside ShortConv) or "moe" (expert intermediate)."""
    kw = {
        "attn": dict(num_heads=args.num_heads // tp, num_kv_heads=args.num_kv_heads // tp),
        "conv": dict(hidden_size=args.hidden_size // tp),
        "moe": dict(moe_intermediate_size=args.moe_intermediate_size // tp),
    }[part]
    return replace(args, **kw)


# ------------------------------------------------------------------------------------------------
# mesh-sharded weights
# ------------------------------------------------------------------------------------------------


def _shard_like(like, per_chip, dim):
    """Mesh tensor whose device d holds ``per_chip[d]``, with ``like``'s dtype / layout / memory config
    (``like`` = the chip-0-shaped replicated tensor the stage-3 constructor built; deallocated)."""
    md = like.device()
    t = ttnn.from_torch(
        torch.cat([p.contiguous() for p in per_chip], dim=dim),
        dtype=like.dtype,
        layout=like.layout,
        device=md,
        memory_config=like.memory_config(),
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=dim),
    )
    assert tuple(t.shape) == tuple(like.shape), (tuple(t.shape), tuple(like.shape))
    ttnn.deallocate(like)
    return t


def _ds_host(lin, w):
    """Host weight of ``DramShardedLinear`` ``lin`` (its zero padding of K / N applied to ``w``)."""
    k0, n0 = w.shape
    if (k0, n0) == (lin.k, lin.n):
        return w
    wp = torch.zeros(lin.k, lin.n, dtype=w.dtype)
    wp[:k0, :n0] = w
    return wp


def _reshard_ds(lin, per_chip_w):
    """Swap ``lin.weight`` for the per-chip (column- or row-parallel) weights, ``[K, N]`` each."""
    lin.weight = _shard_like(lin.weight, [_ds_host(lin, w) for w in per_chip_w], dim=-1)
    return lin


class _PartialOut:
    """Mixin for a row-parallel ``DramShardedLinear``: the decode output stays width-sharded in L1
    (``out_mem``, the all-reduce input) instead of being converted to interleaved.  (fp32 partials summed in
    fp32 did not change the decode router-score error: work_log M-10.)"""

    def decode(self, x, ckc, mem=L1):
        dt = getattr(self, "partial_dtype", None)  # MeshPlan.ccl_dtype (stage 8)
        if dt is None:
            return self.decode_sharded(x, ckc)
        already = x.memory_config() == self.in_mem
        xs = x if already else ttnn.to_memory_config(x, self.in_mem)
        o = ttnn.linear(
            xs,
            self.weight,
            program_config=self.decode_cfg,
            compute_kernel_config=ckc,
            memory_config=self.out_mem,
            dtype=dt,
        )
        if not already:
            ttnn.deallocate(xs)
        return o


def _as_partial(lin):
    lin.__class__ = type("PartialDramShardedLinear", (_PartialOut, DramShardedLinear), {})
    return lin


# ------------------------------------------------------------------------------------------------
# modules
# ------------------------------------------------------------------------------------------------


class MCAttention(OptimizedAttention):
    """Head-parallel attention: chip d computes q heads 8d..8d+7 over its local kv heads 2d, 2d+1 and
    returns the row-parallel out-projection partial (no residual)."""

    def __init__(self, mesh_device, args_local, sds, max_batch_size, policy, fusions, sdpa_chunk):
        super().__init__(mesh_device, args_local, sds[0], max_batch_size, policy, fusions, sdpa_chunk=sdpa_chunk)
        qkv = [
            torch.cat([s["self_attn.q_proj.weight"], s["self_attn.k_proj.weight"], s["self_attn.v_proj.weight"]]).T
            for s in sds
        ]
        wo = [s["self_attn.out_proj.weight"].T for s in sds]
        assert self.ds_qkv is not None and self.ds_wo is not None, "multichip attention needs ds_wqkv / ds_wo"
        _reshard_ds(self.ds_qkv, qkv)
        _as_partial(self.ds_wo)
        self.ds_wo.weight = _shard_like(self.ds_wo.weight, [_ds_host(self.ds_wo, w) for w in wo], dim=-2)


class MCShortConv(OptimizedShortConv):
    """Channel-parallel ShortConv (512 channels per chip, local state); returns the out_proj partial."""

    def __init__(self, mesh_device, args_local, sds, max_batch_size, policy, fusions, sharded_chain=False):
        super().__init__(mesh_device, args_local, sds[0], max_batch_size, policy, fusions)
        assert self.ds_in is not None and self.ds_out is not None, "multichip conv needs ds_conv_in / ds_conv_out"
        _reshard_ds(self.ds_in, [s["conv.in_proj.weight"].T for s in sds])
        _as_partial(self.ds_out)
        self.ds_out.weight = _shard_like(
            self.ds_out.weight, [_ds_host(self.ds_out, s["conv.out_proj.weight"].T) for s in sds], dim=-2
        )
        taps = [s["conv.conv.weight"][:, 0, :] for s in sds]  # [c, K] per chip
        self.taps = [_shard_like(self.taps[k], [w[:, k].reshape(1, 1, 1, -1) for w in taps], dim=-1) for k in range(3)]
        self.sharded_chain = sharded_chain
        if sharded_chain:
            # the chain intermediates run on the out_proj input shards (8 cores x [32, c / 8]); the persistent operands
            # (per-slot state, taps expanded to [32, c]) stay in DRAM: L1 is allocated at the same address in every bank,
            # so 30 layers x 5 x 4 KB would pin ~600 KB of every core's L1 for 1.3 us per layer (work_log O-8)
            self.chain_mem = self.ds_out.in_mem
            self.taps_sh = [
                ttnn.from_torch(
                    torch.cat(
                        [w[:, k].reshape(1, 1, 1, -1).expand(1, 1, self.B, -1) for w in taps], dim=-1
                    ).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                )
                for k in range(3)
            ]

    def decode_forward(self, x, slot_active, residual=None):
        if not self.sharded_chain:
            return super().decode_forward(x, slot_active, residual)
        assert residual is None
        mem, C, T = self.chain_mem, self.H, x.shape[-2]
        assert slot_active.shape[-1] == C, "sharded ShortConv chain needs the full-width slot_mask [1, 1, B, c]"
        mask = slot_active if slot_active.memory_config() == mem else ttnn.to_memory_config(slot_active, mem)
        bcx = self.ds_in.decode_sharded(x, self.ckc)
        b = ttnn.slice(bcx, (0, 0, 0, 0), (1, 1, T, C), memory_config=mem)
        c = ttnn.slice(bcx, (0, 0, 0, C), (1, 1, T, 2 * C), memory_config=mem)
        xx = ttnn.slice(bcx, (0, 0, 0, 2 * C), (1, 1, T, 3 * C), memory_config=mem)
        ttnn.deallocate(bcx)
        bx = ttnn.multiply(b, xx, memory_config=mem)
        ttnn.deallocate(b)
        ttnn.deallocate(xx)
        conv = ttnn.multiply(self.state[0], self.taps_sh[0], memory_config=mem)
        conv = ttnn.mac(self.state[1], self.taps_sh[1], conv)
        conv = ttnn.mac(bx, self.taps_sh[2], conv)
        ttnn.where(mask, self.state[1], self.state[0], output_tensor=self.state[0])
        ttnn.where(mask, bx, self.state[1], output_tensor=self.state[1])
        ttnn.deallocate(bx)
        if mask is not slot_active:
            ttnn.deallocate(mask)
        y = ttnn.multiply(c, conv, memory_config=mem)
        ttnn.deallocate(c)
        ttnn.deallocate(conv)
        out = self.ds_out.decode(y, self.ckc)  # y already in the out_proj input layout; partial stays sharded
        ttnn.deallocate(y)
        return out


class MCDenseMLP(OptimizedDenseMLP):
    """w1/w3 column-parallel, w2 row-parallel; returns the w2 partial."""

    def __init__(self, mesh_device, sds, policy, fusions, packed=None):
        super().__init__(mesh_device, sds[0], policy, fusions)
        w1 = [s["feed_forward.w1.weight"].T for s in sds]
        w3 = [s["feed_forward.w3.weight"].T for s in sds]
        w2 = [s["feed_forward.w2.weight"].T for s in sds]
        assert "dense_minimal_swiglu" not in self.f and "dense_merged_w13" not in self.f
        self.w1 = _shard_like(self.w1, w1, dim=-1)  # prefill (interleaved)
        self.w3 = _shard_like(self.w3, w3, dim=-1)
        assert self.ds_w2 is not None and self.ds_w1 is not None, "multichip dense MLP needs ds_dense_w2 / ds_dense_w13"
        self.ds_w2.weight = _shard_like(self.ds_w2.weight, [_ds_host(self.ds_w2, w) for w in w2], dim=-2)
        _reshard_ds(self.ds_w1, w1)
        _reshard_ds(self.ds_w3, w3)
        _as_partial(self.ds_w2_dec)
        self.ds_w2_dec.weight = _shard_like(self.ds_w2_dec.weight, [_ds_host(self.ds_w2_dec, w) for w in w2], dim=-2)
        self.ds_w13 = None
        if packed:  # decode: one DS matmul over [w1 | w3] (each zero-padded to ds_w1.n, the K of the padded w2)
            n1 = self.ds_w1.n
            pk = [torch.cat([_ds_host(self.ds_w1, a), _ds_host(self.ds_w3, b)], dim=-1) for a, b in zip(w1, w3)]
            self.ds_w13 = DramShardedLinear(mesh_device, pk[0], DTYPES[policy.dense_w13], packed)
            self.ds_w13.weight = _shard_like(self.ds_w13.weight, pk, dim=-1)
            assert self.ds_w13.n == 2 * n1
            for lin in (self.ds_w1, self.ds_w3):  # the decode-only per-projection copies are not used
                ttnn.deallocate(lin.weight)
            self.ds_w1 = self.ds_w3 = None
            self.n1 = n1

    def forward(self, x, residual=None):
        if self.ds_w13 is None or x.shape[-2] > TILE:
            return super().forward(x, residual)
        assert (
            residual is None
        ), "multichip dense MLP returns the row-parallel partial (residual added after the all-reduce)"
        gu = self.ds_w13.decode(x, self.ckc, L1)  # [32, 2 * n1] interleaved
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, TILE, self.n1), memory_config=L1)
        u = ttnn.slice(gu, (0, 0, 0, self.n1), (1, 1, TILE, 2 * self.n1), memory_config=L1)
        ttnn.deallocate(gu)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = self.ds_w2_dec.decode(h, self.ckc, DRAM)  # row-parallel partial (sharded, _PartialOut)
        ttnn.deallocate(h)
        return out


class MCMoE(OptimizedMoE):
    """Router / routing replicated (identical on every chip).

    * decode and short prefill chunks (the stacked ``sparse_matmul`` paths): every expert tensor-parallel
      over its intermediate dimension (384 of 1536 per chip) -- balanced for any routing, the compact
      bucket path reads b*K quarter-experts per chip;
    * prefill token dispatch (chunks >= ``moe_dispatch_min_rows``): expert-parallel -- chip d runs its 16
      experts 16d..16d+15 at full width on the pairs routed to them (the dispatch FFN op is 2.2x faster
      per chip on 16 full experts than on 64 quarter experts, ``sweeps/prefill_expert_ffn_tp_vs_ep.log``).

    Both return the chip's partial of the MoE output (sum over the chips = the MoE output)."""

    def __init__(self, mesh_device, args_local, sds, policy, fusions, sd_full, tp):
        self._ep_src = (sd_full, tp)  # read by _init_dispatch (called from the stage-3 constructor)
        super().__init__(mesh_device, args_local, sds[0], policy, fusions)
        del self._ep_src
        E = self.E
        per_chip = [stacked_expert_weights(s, E) for s in sds]  # (gate [E,I,H], up [E,I,H], down [E,H,I])
        assert "moe_split_gate_up" not in self.f
        # stacked sparse_matmul weights [1, E, H, 2I] = [up | gate] and [1, E, I, H] (FusedMoE / MoE layout)
        self.w_up_gate = _shard_like(
            self.w_up_gate,
            [torch.cat([u.transpose(1, 2), g.transpose(1, 2)], dim=-1).unsqueeze(0) for g, u, _ in per_chip],
            dim=-1,
        )
        self.w_down = _shard_like(self.w_down, [dn.transpose(1, 2).unsqueeze(0) for _, _, dn in per_chip], dim=-2)
        if (
            getattr(self, "v3_bias", None) is not None
            and self.v3_bias.memory_config().buffer_type == ttnn.BufferType.L1
        ):
            # moe_route_v3_l1 puts the SHARED routing constants in L1 (allocated once per device); the per-layer
            # [32, 64] fp32 bias would pin ~4 KB of every core's L1 per MoE layer (~150 KB for 38 layers): keep it in DRAM
            b = ttnn.to_memory_config(self.v3_bias, DRAM)
            ttnn.deallocate(self.v3_bias)
            self.v3_bias = b

    # ---------------- decode: compact experts (bucket <= moe_compact_max_users) ----------------
    def _compact_forward(self, x, slot_active, b):
        """``OptimizedMoE._compact_forward`` with ``Policy.moe_compact_weight_before_down`` (routing weight applied to
        the [n, 32, I] activation) and ``moe_compact_out_sharded`` (expert-sum written in ``self.out_mem``, the
        all-reduce layout, so no reshard before the all-reduce)."""
        p = self.policy
        if not (p.moe_compact_weight_before_down or p.moe_compact_out_sharded):
            return super()._compact_forward(x, slot_active, b)
        from .optimized_decoder import _mcast_1d_config

        T, I, H, K = x.shape[-2], self.I, self.H, self.K
        n = b * K
        ids, wc = (
            self._compact_route_v3(x, b) if (p.moe_route_v3 and T == TILE) else self._compact_route(x, slot_active, b)
        )
        x = ttnn.reshape(x, (1, 1, T, H))
        kw = dict(
            sparsity=self.c_sparsity,
            indices=ids,
            memory_config=L1,
            output_tile=ttnn.Tile([32, 32]),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        gu_cfg = p.moe_gate_up_compact_cfg or p.moe_gate_up_cfg
        ug = ttnn.sparse_matmul(x, self.w_up_gate, program_config=_mcast_1d_config(gu_cfg, T, 2 * I), **kw)
        ug = ttnn.reshape(ug, (1, n, T, 2 * I))
        up = ttnn.slice(ug, (0, 0, 0, 0), (1, n, T, I), memory_config=L1)
        gate = ttnn.slice(ug, (0, 0, 0, I), (1, n, T, 2 * I), memory_config=L1)
        ttnn.deallocate(ug)
        act = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        wt = ttnn.permute(ttnn.reshape(wc, (1, 1, T, n)), (0, 3, 2, 1), memory_config=L1)  # [1, n, T, 1]
        ttnn.deallocate(wc)
        if p.moe_compact_weight_before_down:
            act_w = ttnn.multiply(act, wt, memory_config=L1)
            ttnn.deallocate(act)
            ttnn.deallocate(wt)
            act = act_w
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            program_config=_mcast_1d_config(p.moe_down_compact_cfg or p.moe_down_cfg, T, H),
            is_input_a_sparse=True,
            **kw,
        )  # [1, n, T, H]
        ttnn.deallocate(act)
        ttnn.deallocate(ids)
        if not p.moe_compact_weight_before_down:
            down_w = ttnn.multiply(down, wt, memory_config=L1)
            ttnn.deallocate(down)
            ttnn.deallocate(wt)
            down = down_w
        out_mem = self.out_mem if (p.moe_compact_out_sharded and getattr(self, "out_mem", None) is not None) else L1
        dt = getattr(self, "partial_dtype", None)  # MeshPlan.ccl_dtype with ccl_moe (stage 8): expert sum in that dtype
        if dt is not None:
            buf = ttnn.empty((1, 1, T, H), dt, ttnn.TILE_LAYOUT, self.mesh_device, out_mem)
            out = ttnn.experimental.fast_reduce_nc(down, dims=[1], output=buf, memory_config=out_mem)
        else:
            out = ttnn.experimental.fast_reduce_nc(down, dims=[1], memory_config=out_mem)
        ttnn.deallocate(down)
        return ttnn.reshape(out, (1, 1, T, H))

    # ---------------- decode: bucket-dependent down config ----------------
    def forward(self, x, slot_active=None, x_sharded=None, active_slots=None):
        self._bucket = active_slots
        try:
            return super().forward(x, slot_active=slot_active, x_sharded=x_sharded, active_slots=active_slots)
        finally:
            self._bucket = None

    def _mm_cfg(self, role, rows, n, groups):
        small = self.policy.moe_down_small_bucket_cfg
        b = getattr(self, "_bucket", None)
        if small and role == "down" and groups == 1 and rows <= TILE and b is not None and b <= small[0]:
            from .optimized_decoder import _mcast_1d_config

            return _mcast_1d_config(small[1], rows, n)
        return super()._mm_cfg(role, rows, n, groups)

    # ---------------- prefill token dispatch, expert-parallel ----------------
    def _init_dispatch(self, sd_local):
        """Expert-parallel dispatch constants.  Local expert table (``W`` = 32 entries, one tile): entry 0
        is a dummy expert whose one-tile region (rows 0..31) gets zero input rows (output exactly 0),
        entries 1..16 the chip's experts 16d..16d+15 (full intermediate), the rest unused (count 0).  A pair
        routed to another chip's expert gets destination row ``p mod 32`` (a dummy row, exact zero back);
        unused buffer rows gather table row ``r mod 32`` (the gather table starts with 32 zero rows): the
        references to zeros are spread over 32 rows instead of hammering one DRAM page (one zero row: the
        two gathers took 1138 vs 480 us, tracy_v1/conv_moe/a1_b1/ep_prefill)."""
        sd_full, tp = self._ep_src
        md, E, K, H = self.mesh_device, self.E, self.K, self.H
        El, W = E // tp, TILE
        assert El + 1 <= W
        self.d_width = W
        gate, up, down = stacked_expert_weights(sd_full, E)  # [E, I, H], [E, I, H], [E, H, I] (full I)
        dt_gu, dt_d = DTYPES[self.policy.expert_gate_up], DTYPES[self.policy.expert_down]

        def chips(ts, dtype):  # device d gets ts[d]
            return ttnn.from_torch(
                torch.cat([t.contiguous() for t in ts], dim=0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
            )

        loc_g = [chips([gate[d * El + j].T for d in range(tp)], dt_gu) for j in range(El)]  # (H, I)
        loc_u = [chips([up[d * El + j].T for d in range(tp)], dt_gu) for j in range(El)]
        loc_d = [chips([down[d * El + j].T for d in range(tp)], dt_d) for j in range(El)]  # (I, H)
        pad = W - 1 - El  # unused entries (count 0) alias local expert 0's weights
        # dummy entry 0: gate / up alias expert 0, down is ZERO, so its output rows are exactly 0 whatever their
        # (finite) input rows -> the gather table can be x itself (no zero-row header / concat)
        self.d_zero_down = ttnn.from_torch(
            torch.zeros(gate.shape[1], H),
            dtype=dt_d,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )  # (I, H)
        self.d_gates = [loc_g[0]] + loc_g + [loc_g[0]] * pad
        self.d_ups = [loc_u[0]] + loc_u + [loc_u[0]] * pad
        self.d_downs = [self.d_zero_down] + loc_d + [loc_d[0]] * pad
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.d_gidx = _as_tt(torch.arange(W, dtype=torch.int32), md, dtype=ttnn.uint32, layout=rm)
        ids = torch.full((tp, W), -1.0)  # global expert id of each local entry (-1: matches no pair)
        for d in range(tp):
            ids[d, 1 : El + 1] = torch.arange(d * El, (d + 1) * El, dtype=torch.float32)
        self.d_arange_e = ttnn.from_torch(
            ids,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
        )  # [1, W] per chip
        dummy = torch.zeros(1, W)
        dummy[0, 0] = float(TILE)
        self.d_dummy_count = _as_tt(dummy, md, dtype=ttnn.float32)  # the dummy region: one full tile
        pmax = self.policy.moe_dispatch_max_rows * K
        mb = self._max_buf(self.policy.moe_dispatch_max_rows)
        # scatter source of each pair (token t -> table row t, the gather table is x) as a [P, 1] column; pairs and
        # buffer rows with nothing local get token row r mod 32 (finite rows: dummy / unused rows are never read back)
        self.d_src_col = _as_tt(
            (torch.arange(pmax).div(K, rounding_mode="floor") + TILE).float().reshape(pmax, 1), md, dtype=ttnn.float32
        )  # + 32: sources >= 32 mark written (local) slots for the max-with-spread below; subtracted after
        self.d_pmod_col = _as_tt((torch.arange(pmax) % TILE).float().reshape(pmax, 1), md, dtype=ttnn.float32)
        # buffer row r without a local pair gathers x row (r mod 32).  Exactness of max(src, spread) below: a local slot
        # holds token + 32 and r mod 32 <= token there (a token's top-K experts are distinct, so an expert region holds
        # at most one pair per earlier token: slot index inside a region <= token index), so max() keeps the token
        self.d_spread = _as_tt((torch.arange(mb) % TILE + TILE).float().reshape(1, mb), md, dtype=ttnn.float32)
        self.d_src_zero = _as_tt(torch.zeros(1, mb), md, dtype=ttnn.float32, layout=rm)
        R = self.policy.moe_scatter_rows
        if R > 1:
            self.d_src_zero_rows = _as_tt(
                torch.zeros(R, self._max_buf(self.policy.moe_dispatch_max_rows)), md, dtype=ttnn.float32, layout=rm
            )
        gmax = pmax // TILE
        self.d_tri32 = _as_tt(
            torch.tril(torch.ones(TILE, TILE)).reshape(1, 1, TILE, TILE).expand(1, gmax, TILE, TILE).contiguous(), md
        )
        self.d_blocks_strict = _as_tt(torch.tril(torch.ones(gmax, gmax), -1), md)
        self.d_experts_strict = _as_tt(torch.triu(torch.ones(W, W), 1), md)
        self.d_scan_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.d_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity={"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[
                self.policy.dispatch_fidelity
            ],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _max_buf(self, T):
        # every local region (dummy included) starts tile-aligned: <= T*K pairs + 31 padding rows per entry
        return ((T * self.K + self.d_width * 31) + TILE - 1) // TILE * TILE

    def _pair_cumsum(self, onehot, P):
        """``OptimizedMoE._pair_cumsum`` over the ``d_width``-wide local one-hot."""
        E, self.E = self.E, self.d_width
        try:
            return super()._pair_cumsum(onehot, P)
        finally:
            self.E = E

    def _dispatch_forward(self, x):
        """``OptimizedMoE._dispatch_forward`` over the chip's local expert table (see ``_init_dispatch``)."""
        K, H, W = self.K, self.H, self.d_width
        T = x.shape[-2]
        P, maxbuf = T * K, self._max_buf(T)
        rm = lambda a: ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
        idx, w = self._route_topk(x)  # [1,1,T,K] uint16 / fp32, identical on every chip
        e_col = ttnn.reshape(ttnn.typecast(idx, ttnn.float32), (P, 1))
        ttnn.deallocate(idx)
        onehot = ttnn.eq(e_col, self.d_arange_e)  # [P, W]: pair -> local entry (all zero: another chip's expert)
        ttnn.deallocate(e_col)
        cum = self._pair_cumsum(onehot, P)
        counts = ttnn.add(ttnn.slice(cum, (P - 1, 0), (P, W)), self.d_dummy_count)  # [1, W], dummy = 1
        tiles = ttnn.floor(ttnn.multiply(ttnn.add(counts, TILE - 1.0), 1.0 / TILE))
        region = ttnn.multiply(
            ttnn.matmul(
                ttnn.typecast(tiles, ttnn.bfloat16),
                self.d_experts_strict,
                compute_kernel_config=self.d_scan_ckc,
                dtype=ttnn.float32,
            ),
            float(TILE),
        )  # dummy region = rows 0..31
        ttnn.deallocate(tiles)
        dest = ttnn.sum(ttnn.multiply(onehot, ttnn.add(ttnn.add(cum, region), -1.0)), dim=-1, keepdim=True)  # [P,1]
        local = ttnn.sum(onehot, dim=-1, keepdim=True)  # [P, 1] 0/1
        ttnn.deallocate(onehot)
        ttnn.deallocate(cum)
        pmod = ttnn.slice(self.d_pmod_col, (0, 0), (P, 1))
        dest = ttnn.add(dest, ttnn.subtract(pmod, ttnn.multiply(pmod, local)))  # non-local: dummy row p mod 32
        # ([P, 1] -> [1, P] then row-major [R, P / R]: going straight to the [R, P / R] tile form is slower,
        # 248 vs 104 us of ReshapeView per chunk: work_log O-7)
        dest_u = ttnn.typecast(rm(ttnn.reshape(dest, (1, P))), ttnn.uint32)
        ttnn.deallocate(dest)
        src_col = ttnn.multiply(ttnn.slice(self.d_src_col, (0, 0), (P, 1)), local)  # non-local -> 0
        ttnn.deallocate(local)
        src_tok = rm(ttnn.reshape(src_col, (1, P)))
        ttnn.deallocate(src_col)
        R = self.policy.moe_scatter_rows
        if R > 1:
            # every local destination is written exactly once; row 0 (dummy) only ever receives 0
            init = ttnn.slice(self.d_src_zero_rows, (0, 0), (R, maxbuf))
            part = ttnn.scatter(init, -1, ttnn.reshape(dest_u, (R, P // R)), ttnn.reshape(src_tok, (R, P // R)))
            part_t = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
            ttnn.deallocate(part)
            red = ttnn.sum(part_t, dim=0, keepdim=True)
            ttnn.deallocate(part_t)
            src_t = red
        else:
            src_t = ttnn.to_layout(
                ttnn.scatter(ttnn.slice(self.d_src_zero, (0, 0), (1, maxbuf)), -1, dest_u, src_tok), ttnn.TILE_LAYOUT
            )
        ttnn.deallocate(src_tok)
        # local slots hold token + 32, everything else 0 -> max with (r mod 32) + 32; minus 32: local -> token row of x,
        # others -> row r mod 32 of x (finite; only the zero-down dummy / never-read rows use them)
        src_f = ttnn.maximum(src_t, ttnn.slice(self.d_spread, (0, 0), (1, maxbuf)))
        ttnn.deallocate(src_t)
        src_x = ttnn.add(src_f, -float(TILE))  # spread values are r mod 32 + 32, local slots token + 32
        ttnn.deallocate(src_f)
        src = ttnn.typecast(rm(src_x), ttnn.uint32)
        ttnn.deallocate(src_x)
        table = ttnn.reshape(rm(x), (T, H))
        buf = ttnn.reshape(ttnn.embedding(src, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16), (maxbuf, H))
        ttnn.deallocate(src)
        ttnn.deallocate(table)
        region_u = ttnn.reshape(ttnn.typecast(rm(region), ttnn.uint32), (W,))
        counts_u = ttnn.reshape(ttnn.typecast(rm(counts), ttnn.uint32), (W,))
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            buf,
            region_u,
            counts_u,
            self.d_gidx,
            self.d_gates,
            self.d_ups,
            self.d_downs,
            T,
            compute_kernel_config=self.d_ckc,
        )  # bfp8 TILE [maxbuf, H]; dummy rows exactly 0
        ttnn.deallocate(buf)
        out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out)
        comb = ttnn.reshape(
            ttnn.embedding(dest_u, out_rm, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16), (1, 1, T, K, H)
        )
        ttnn.deallocate(out_rm)
        ttnn.deallocate(dest_u)
        w5 = ttnn.reshape(rm(ttnn.typecast(w, ttnn.bfloat16)), (1, 1, T, K, 1))
        ttnn.deallocate(w)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(comb, w5, expert_dim=3, output_memory_config=DRAM)
        ttnn.deallocate(comb)
        return ttnn.reshape(y, (1, 1, T, H))


# ------------------------------------------------------------------------------------------------
# collectives
# ------------------------------------------------------------------------------------------------


_AR_SCRATCH = {}  # (mesh id, H, tp, cores) -> (mesh, persistent all_reduce_async scratch)


class MeshReduce:
    """Sum of the per-chip partials, replicated on every chip (bitwise identical on all chips)."""

    def __init__(self, mesh_device, plan: MeshPlan, hidden):
        from models.common.modules.tt_ccl import get_tt_ccl

        self.mesh_device = mesh_device
        self.plan = plan
        self.ccl = get_tt_ccl(mesh_device)  # one semaphore set per mesh, shared by all layers
        self.topology = ttnn.Topology.Ring
        self.H = hidden
        self.resid_mem = width_shard_layout(mesh_device, plan.resid_cores, hidden // plan.resid_cores)
        # all_reduce_async scratch: every chip's partial side by side on the residual cores
        self.buf_mem = width_shard_layout(mesh_device, plan.resid_cores, hidden * plan.tp // plan.resid_cores)
        self.scratch = None
        if plan.persistent_scratch:
            # two [32, H * tp] L1 buffers per mesh, shared by every layer and bucket trace, used alternately: all_reduce_async
            # writes into the OTHER chips' scratch, and a chip can be at most one all-reduce ahead of the slowest one (it
            # cannot complete all-reduce k+1 before every chip has sent its k+1 data, which each sends only after finishing
            # k), so with two buffers a leading chip never overwrites a scratch another chip is still reducing from.
            # Every layer issues 2 all-reduces, so each layer / trace starts on buffer 0.
            # one pair per payload dtype (MeshPlan.ccl_dtype: partials may arrive bf16 or reduced) with ONE alternation
            # counter shared by all pairs: consecutive all-reduces always use buffers of different parity (different
            # physical buffers whatever their dtypes), and every trace issues an even number of all-reduces (2 per layer),
            # so each trace starts on parity 0 as before
            key0 = (id(mesh_device), hidden, plan.tp, plan.resid_cores)
            idx = _AR_SCRATCH.get(key0 + ("idx",))
            if idx is None or idx[0] is not mesh_device:
                idx = _AR_SCRATCH[key0 + ("idx",)] = (mesh_device, [0])
            self._scratch_idx = idx[1]
            self.scratch = {}
            for dt in {ttnn.bfloat16, DTYPES[plan.ccl_dtype]}:
                key = key0 + (str(dt),)
                hit = _AR_SCRATCH.get(key)
                if hit is None or hit[0] is not mesh_device:
                    bufs = [
                        ttnn.empty((1, 1, TILE, hidden * plan.tp), dt, ttnn.TILE_LAYOUT, mesh_device, self.buf_mem)
                        for _ in range(2)
                    ]
                    hit = _AR_SCRATCH[key] = (mesh_device, bufs)
                self.scratch[dt] = hit[1]

    def decode(self, partial):
        """``partial`` [1, 1, 32, H] (any memory) -> reduced, width-sharded on ``resid_mem``."""
        p = partial
        if p.memory_config() != self.resid_mem:
            p = ttnn.to_memory_config(partial, self.resid_mem)
            ttnn.deallocate(partial)
        if self.plan.decode_reduce == "all_reduce":
            out = ttnn.all_reduce(
                p, num_links=self.plan.num_links, topology=self.topology, memory_config=self.resid_mem
            )
            ttnn.deallocate(p)
            return out
        # persistent shared scratch (MeshPlan.persistent_scratch: dead between all-reduces, so one buffer serves
        # every layer and every bucket trace), else allocated per call inside the trace like any intermediate
        buf = None
        if self.scratch is not None:
            buf = self.scratch[p.dtype][self._scratch_idx[0]]
            self._scratch_idx[0] ^= 1
        if buf is None:
            buf = ttnn.empty(
                (1, 1, p.shape[-2], self.H * self.plan.tp),
                p.dtype,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                self.buf_mem,
            )
        out = ttnn.experimental.all_reduce_async(
            p,
            buf,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=self.topology,
            num_links=self.plan.num_links,
            memory_config=self.resid_mem,
            dtype=ttnn.bfloat16,  # residual stays bf16 whatever the payload dtype
        )
        if self.scratch is None:
            ttnn.deallocate(buf)
        ttnn.deallocate(p)
        return out

    def prefill(self, partial):
        """``partial`` [1, 1, S, H] DRAM -> reduced, DRAM interleaved."""
        if partial.shape[-2] <= TILE:  # single-tile prefill chunk: the decode-shaped modules ran
            r = self.decode(partial)
            out = ttnn.to_memory_config(r, DRAM)
            ttnn.deallocate(r)
            return out
        out = ttnn.all_reduce(partial, num_links=self.plan.num_links, topology=self.topology, memory_config=DRAM)
        ttnn.deallocate(partial)
        return out


# ------------------------------------------------------------------------------------------------
# decoder
# ------------------------------------------------------------------------------------------------


class MultichipDecoder(OptimizedDecoder):
    """TP = 4 decoder layer on a 1x4 mesh (see module docstring).  Same prefill / decode signatures as
    the single-chip decoder; decode input/output may be passed DRAM/L1 interleaved or already in
    ``resid_mem`` (the stacked-model contract: no conversion between layers)."""

    @property
    def slot_mask_mem(self):
        """Memory config the ShortConv decode wants ``slot_mask`` in (None: any; a caller can build it once per step)."""
        return getattr(self.mixer, "chain_mem", None) if not self.is_attention else None

    @property
    def slot_mask_width(self):
        """Width of the decode ``slot_mask`` the ShortConv state update takes (its local channels)."""
        return None if self.is_attention else self.mixer.H

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=TILE,
        policy=MC_POLICY,
        fusions=DEFAULT_FUSIONS,
        plan=DEFAULT_MESH_PLAN,
        **kwargs,
    ):
        return cls(state_dict, hf_config, layer_idx, mesh_device, max_batch_size, policy, fusions, plan, **kwargs)

    def __init__(self, sd, hf_config, layer_idx, mesh_device, max_batch_size, policy, fusions, plan, sdpa_chunk=256):
        from .functional_decoder import _as_tt, _hifi4
        from .optimized_decoder import ShardedNorm

        assert tuple(mesh_device.shape) == (1, plan.tp), f"mesh {tuple(mesh_device.shape)} != (1, {plan.tp})"
        assert policy.sharded_decode_norm and policy.ffn_norm_cores == plan.resid_cores
        self.policy, self.plan = policy, plan
        self.fusions = frozenset(fusions)
        args = Lfm2MoeArgs.from_hf_config(hf_config)
        self.args = args
        self.layer_idx = layer_idx
        self.kind = args.layer_kind(layer_idx)
        self.mesh_device = mesh_device
        self.eps = args.norm_eps
        self.ckc_norm = _hifi4(mesh_device)
        H = args.hidden_size
        self.operator_norm = _as_tt(
            sd["operator_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.ffn_norm = _as_tt(sd["ffn_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        sds = [local_state_dict(sd, args, d, plan.tp) for d in range(plan.tp)]
        self.is_attention = args.is_attention_layer(layer_idx)
        if self.is_attention:
            la = local_args(args, plan.tp, "attn")
            self.mixer = MCAttention(
                mesh_device, la, sds, max_batch_size, policy, self.fusions, policy.sdpa_prefill_chunk
            )
        else:
            la = local_args(args, plan.tp, "conv")
            self.mixer = MCShortConv(
                mesh_device, la, sds, max_batch_size, policy, self.fusions, plan.conv_sharded_chain
            )
        if args.is_moe_layer(layer_idx):
            self.ffn = MCMoE(mesh_device, local_args(args, plan.tp, "moe"), sds, policy, self.fusions, sd, plan.tp)
        else:
            self.ffn = MCDenseMLP(mesh_device, sds, policy, self.fusions, plan.dense_w13_packed)
        del sds
        self.conv_prefill_windowed = (not self.is_attention) and policy.conv_prefill_windowed_tail
        consumer = self.mixer.ds_qkv if self.is_attention else self.mixer.ds_in
        self.op_norm_sh = ShardedNorm(mesh_device, H, consumer.cores, self.eps, self.ckc_norm)
        assert self.op_norm_sh.mem == consumer.in_mem
        self.ffn_norm_sh = ShardedNorm(mesh_device, H, policy.ffn_norm_cores, self.eps, self.ckc_norm)
        self.reduce = MeshReduce(mesh_device, plan, H)
        assert self.ffn_norm_sh.mem == self.reduce.resid_mem
        self.resid_mem = self.reduce.resid_mem
        if isinstance(self.ffn, MCMoE):
            self.ffn.out_mem = (
                self.resid_mem
            )  # compact expert sum written in the all-reduce layout (moe_compact_out_sharded)
        if plan.ccl_dtype != "bf16":  # stage 8: reduced decode all-reduce payload
            dt = DTYPES[plan.ccl_dtype]
            part = self.mixer.ds_wo if self.is_attention else self.mixer.ds_out
            part.partial_dtype = dt
            if isinstance(self.ffn, MCMoE):
                if plan.ccl_moe:
                    self.ffn.partial_dtype = dt
            else:
                self.ffn.ds_w2_dec.partial_dtype = dt

    # ---- decode ----
    def decode_forward(
        self,
        x,
        *,
        rope=None,
        rope_cos=None,
        rope_sin=None,
        cur_pos=None,
        page_table=None,
        slot_active=None,
        active_slots=None,
        slot_mask=None,
    ):
        """Residual in ``resid_mem`` (width-sharded L1, identical on all chips); returns the same layout.
        Other arguments as ``OptimizedDecoder.decode_forward``."""
        xr = x if x.memory_config() == self.resid_mem else ttnn.to_memory_config(x, self.resid_mem)
        xn = self.op_norm_sh(xr, self.operator_norm)  # sharded on the qkv / in_proj input grid
        if self.is_attention:
            m = self.mixer.decode_forward(
                xn,
                rope=rope,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                cur_pos=cur_pos,
                page_table=page_table,
                residual=None,
                active_slots=active_slots,
            )
        else:
            m = self.mixer.decode_forward(xn, slot_active if slot_mask is None else slot_mask, residual=None)
        ttnn.deallocate(xn)
        r = self.reduce.decode(m)
        h = ttnn.add(xr, r, memory_config=self.resid_mem)
        ttnn.deallocate(r)
        if xr is not x:
            ttnn.deallocate(xr)
        hs = self.ffn_norm_sh(h, self.ffn_norm)  # h already on the ffn-norm grid
        if isinstance(self.ffn, FusedDenseMLP):
            hn = ttnn.to_memory_config(hs, L1)
            ttnn.deallocate(hs)
            f = self.ffn.forward(hn, residual=None)
            ttnn.deallocate(hn)
        elif self.plan.moe_sharded_input:  # router and expert matmuls both read the sharded copy
            f = self.ffn.forward(hs, slot_active=slot_active, x_sharded=hs, active_slots=active_slots)
            ttnn.deallocate(hs)
        else:
            hn = ttnn.to_memory_config(hs, L1)
            f = self.ffn.forward(hn, slot_active=slot_active, x_sharded=hs, active_slots=active_slots)
            ttnn.deallocate(hs)
            ttnn.deallocate(hn)
        r = self.reduce.decode(f)
        out = ttnn.add(h, r, memory_config=self.resid_mem)
        ttnn.deallocate(r)
        ttnn.deallocate(h)
        return out

    # ---- prefill ----
    def prefill_forward(
        self,
        x,
        *,
        rope=None,
        chunk_start=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_tensor=None,
        conv_inputs=None,
    ):
        xn = self._norm(x, self.operator_norm, DRAM)
        if self.is_attention:
            m = self.mixer.prefill_forward(
                xn,
                rope=rope,
                chunk_start=chunk_start,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_tensor=chunk_start_tensor,
                residual=None,
            )
        else:
            m = self.mixer.prefill_forward(xn, conv_inputs, residual=None)
        ttnn.deallocate(xn)
        r = self.reduce.prefill(m)
        h = ttnn.add(x, r, memory_config=DRAM)
        ttnn.deallocate(r)
        hn = self._norm(h, self.ffn_norm, DRAM)
        f = self.ffn.forward(hn, residual=None) if isinstance(self.ffn, FusedDenseMLP) else self.ffn.forward(hn)
        ttnn.deallocate(hn)
        r = self.reduce.prefill(f)
        out = ttnn.add(h, r, memory_config=DRAM)
        ttnn.deallocate(r)
        ttnn.deallocate(h)
        return out


def make_multichip(
    sd,
    cfg,
    args,
    layer_idx,
    mesh_device,
    max_seq_len,
    policy=MC_POLICY,
    fusions=DEFAULT_FUSIONS,
    plan=DEFAULT_MESH_PLAN,
):
    dec = MultichipDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, policy=policy, fusions=fusions, plan=plan
    )
    rope = make_rope(mesh_device, args, max_seq_len, fusions) if dec.is_attention else None
    return dec, rope


def mesh_device_params(trace_region_size=96 << 20):
    """pytest ``device_params`` for the 1x4 ring (fabric config before the mesh opens)."""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "fabric_router_config": router,
        # CCL ops place their semaphores in L1_SMALL; without it the prefill all_reduce allocates them in L1,
        # where they clash with later static circular buffers (logs/m1_full_suite.log)
        "l1_small_size": 24576,
        "trace_region_size": trace_region_size,
    }


__all__ = [
    "MeshPlan",
    "DEFAULT_MESH_PLAN",
    "MC_POLICY",
    "MultichipDecoder",
    "make_multichip",
    "local_state_dict",
    "mesh_device_params",
]
