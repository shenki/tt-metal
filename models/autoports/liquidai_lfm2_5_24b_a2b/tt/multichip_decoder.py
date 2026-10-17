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
        return self.decode_sharded(x, ckc)


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

    def __init__(self, mesh_device, args_local, sds, max_batch_size, policy, fusions):
        super().__init__(mesh_device, args_local, sds[0], max_batch_size, policy, fusions)
        assert self.ds_in is not None and self.ds_out is not None, "multichip conv needs ds_conv_in / ds_conv_out"
        _reshard_ds(self.ds_in, [s["conv.in_proj.weight"].T for s in sds])
        _as_partial(self.ds_out)
        self.ds_out.weight = _shard_like(
            self.ds_out.weight, [_ds_host(self.ds_out, s["conv.out_proj.weight"].T) for s in sds], dim=-2
        )
        taps = [s["conv.conv.weight"][:, 0, :] for s in sds]  # [c, K] per chip
        self.taps = [_shard_like(self.taps[k], [w[:, k].reshape(1, 1, 1, -1) for w in taps], dim=-1) for k in range(3)]


class MCDenseMLP(OptimizedDenseMLP):
    """w1/w3 column-parallel, w2 row-parallel; returns the w2 partial."""

    def __init__(self, mesh_device, sds, policy, fusions):
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
        pad = W - 1 - El  # dummy (zero input) and unused entries alias local expert 0's weights
        self.d_gates = [loc_g[0]] + loc_g + [loc_g[0]] * pad
        self.d_ups = [loc_u[0]] + loc_u + [loc_u[0]] * pad
        self.d_downs = [loc_d[0]] + loc_d + [loc_d[0]] * pad
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
        # scatter source of each pair (token t -> table row t + 32; rows 0..31 are zero) as a [P, 1] column
        self.d_src_col = _as_tt(
            (torch.arange(pmax).div(K, rounding_mode="floor") + TILE).float().reshape(pmax, 1), md, dtype=ttnn.float32
        )
        self.d_pmod_col = _as_tt((torch.arange(pmax) % TILE).float().reshape(pmax, 1), md, dtype=ttnn.float32)
        self.d_spread = _as_tt((torch.arange(mb) % TILE).float().reshape(1, mb), md, dtype=ttnn.float32)
        self.d_src_zero = _as_tt(torch.zeros(1, mb), md, dtype=ttnn.float32, layout=rm)
        self.d_zero_row = _as_tt(torch.zeros(1, 1, TILE, H), md, layout=rm)  # gather-table header
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
        # local destinations hold token rows >= 32, everything else 0 -> zero row r mod 32 of the table
        src_f = ttnn.maximum(src_t, ttnn.slice(self.d_spread, (0, 0), (1, maxbuf)))
        ttnn.deallocate(src_t)
        src = ttnn.typecast(rm(src_f), ttnn.uint32)
        ttnn.deallocate(src_f)
        table = ttnn.reshape(ttnn.concat([self.d_zero_row, rm(x)], dim=2), (T + TILE, H))
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
        # scratch allocated per call (in the trace like any intermediate).  A persistent L1 scratch is no longer
        # blocked (the hybrid prefill op that claimed all of L1 is not used): optimized-multichip-stage candidate
        buf = ttnn.empty(
            (1, 1, p.shape[-2], self.H * self.plan.tp), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, self.buf_mem
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
        )
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
            self.mixer = MCShortConv(mesh_device, la, sds, max_batch_size, policy, self.fusions)
        if args.is_moe_layer(layer_idx):
            self.ffn = MCMoE(mesh_device, local_args(args, plan.tp, "moe"), sds, policy, self.fusions, sd, plan.tp)
        else:
            self.ffn = MCDenseMLP(mesh_device, sds, policy, self.fusions)
        del sds
        self.conv_prefill_windowed = (not self.is_attention) and policy.conv_prefill_windowed_tail
        consumer = self.mixer.ds_qkv if self.is_attention else self.mixer.ds_in
        self.op_norm_sh = ShardedNorm(mesh_device, H, consumer.cores, self.eps, self.ckc_norm)
        assert self.op_norm_sh.mem == consumer.in_mem
        self.ffn_norm_sh = ShardedNorm(mesh_device, H, policy.ffn_norm_cores, self.eps, self.ckc_norm)
        self.reduce = MeshReduce(mesh_device, plan, H)
        assert self.ffn_norm_sh.mem == self.reduce.resid_mem
        self.resid_mem = self.reduce.resid_mem

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
        hn = ttnn.to_memory_config(hs, L1)
        if isinstance(self.ffn, FusedDenseMLP):
            ttnn.deallocate(hs)
            f = self.ffn.forward(hn, residual=None)
        else:
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
