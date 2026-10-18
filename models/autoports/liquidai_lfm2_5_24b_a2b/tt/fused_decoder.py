# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for LiquidAI/LFM2.5-24B-A2B (stage 2: graph fusing).

Same prefill/decode contract as ``functional_decoder`` (see its module docstring); the op graph
is rewritten with dedicated tt-metal ops.  Each rewrite is a named fusion so it can be A/B
measured; ``DEFAULT_FUSIONS`` is the selected set (see ``doc/fused_decoder/README.md``).

Decode differences vs functional: ``rope.decode_cos_sin`` returns height-sharded cos/sin for
``rotary_embedding_hf`` (one user per core, same grid as ``nlp_create_qkv_heads_decode``).
"""

import torch

import ttnn

from .functional_decoder import (
    DRAM,
    L1,
    TILE,
    Attention,
    DenseMLP,
    FunctionalDecoder,
    MoE,
    RopeSetup,
    ShortConv,
    _as_tt,
    _hifi2,
    _hifi4,
    _sparse_matmul_config,
)
from .model_config import Lfm2MoeArgs
from .weights import stacked_expert_weights

ALL_FUSIONS = (
    "qkv_heads_decode",  # slice/reshape/pad/reshard -> nlp_create_qkv_heads_decode
    "rope_hf",  # x*cos + (x@R)*sin -> rotary_embedding_hf (prefill + decode)
    "qk_norm_weight",  # rms_norm then multiply(weight) -> rms_norm(weight=)
    "concat_heads_decode",  # decode attn reshape -> nlp_concat_heads_decode
    "moe_grouped_topk",  # sigmoid/add/topk/gather/sum/div/mul -> moe_grouped_topk
    "moe_decode_no_permute",  # decode: G=1, permute -> reshape
    "moe_swiglu_mul_act",  # ttnn.swiglu composite -> slices + mul(input_tensor_b_activations=[SILU])
    "moe_down_union_sparsity",  # decode down: all experts -> union-of-selected sparsity
    "dense_merged_w13",  # w1, w3 linears -> one linear over [w1|w3] + slices + mul(silu)
    "conv_state_inplace",  # where -> copy into state  ->  where(output_tensor=state)
    "dense_minimal_swiglu",  # dense MLP gate/up -> minimal_matmul(fuse_swiglu) over tile-interleaved [w1|w3]
    "residual_fold_prefill",  # prefill out_proj/w2 matmul + residual add -> dit_minimal_matmul_addcmul_fused
    "kv_fused_update",  # 2x paged_update_cache -> paged_fused_update_cache (V on a disjoint core grid)
    "moe_prefill_down_sparse",  # prefill: permute + all-expert down -> per-(tile, expert) sparse down, no permute
    "router_fold",  # router linear + sigmoid -> linear(activation=sigmoid); drop multiply by scale 1.0
    "conv_split_in_proj",  # in_proj [H,3H] + 3 slices -> 3 narrower matmuls (slice pushed into the operand)
    "moe_split_gate_up",  # sparse_matmul [up|gate] + 2 slices -> 2 sparse_matmuls (slice pushed into the weights)
    "router_tile_mask",  # topk -> gather -> untilize/scatter/tilize -> tile-only k-th-value threshold mask
    "conv_tile_shift",  # conv prefill: 3x slice of the TILE tensor (unaligned ones untilize/tilize inside slice)
    "conv_hybrid_shift",  # conv prefill: 1 untilize, 2 unaligned shifts (slice+tilize), aligned shift sliced in TILE
)
# Selected by measurement (doc/fused_decoder/README.md); rejected candidates stay switchable.
REJECTED_FUSIONS = frozenset(
    {
        "dense_merged_w13",
        "dense_minimal_swiglu",
        "kv_fused_update",
        "moe_grouped_topk",
        "conv_split_in_proj",
        "moe_split_gate_up",
        "conv_tile_shift",  # superseded by conv_hybrid_shift (2 untilize + 2 tilize vs 1 + 2)
    }
)
DEFAULT_FUSIONS = frozenset(ALL_FUSIONS) - REJECTED_FUSIONS


def decode_head_mem(mesh_device, batch, head_dim):
    """Height-sharded ``[32, head_dim]`` per user, users laid out ROW-MAJOR over the full compute
    grid width.  ``paged_scaled_dot_product_attention_decode`` with sharded Q places user ``b``'s
    reducer at ``(b % grid.x, b // grid.x)`` of its compute grid, so the Q shard grid must use the
    same width as the SDPA grid (an 8-wide shard grid with the 11-wide Blackhole SDPA grid reads
    other cores' L1: wrong and run-to-run nondeterministic output, see doc/fused_decoder)."""
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return _height_shard(cores, (TILE, head_dim))


def _row_major_cores(start, count, grid_x):
    """CoreRangeSet of ``count`` cores starting at linear index ``start``, row-major over ``grid_x``."""
    ranges = []
    i = start
    while i < start + count:
        y, x = divmod(i, grid_x)
        n = min(grid_x - x, start + count - i)
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + n - 1, y)))
        i += n
    return ttnn.CoreRangeSet(ranges)


def _ones_row(mesh_device, n):
    return _as_tt(torch.ones(1, 1, 1, n), mesh_device)


def _project(inp, w, residual, ones, ckc, fold):
    """``inp @ w`` (+ ``residual``): one fused minimal-matmul+addcmul when ``fold`` and the input
    has more than one tile of rows (prefill; measured slower than linear + add for decode M=32)."""
    # the fused addcmul requires the weight tile size to match the residual's (bf16 weights only)
    if fold and residual is not None and inp.shape[-2] > TILE and w.dtype == residual.dtype:
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            inp, w, 1.0, residual, ones, compute_kernel_config=ckc, memory_config=DRAM, dtype=ttnn.bfloat16
        )
    out = ttnn.linear(inp, w, compute_kernel_config=ckc, memory_config=DRAM)
    if residual is None:
        return out
    h = ttnn.add(residual, out, memory_config=DRAM)
    ttnn.deallocate(out)
    return h


def _height_shard(grid, shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class FusedRope(RopeSetup):
    """``RopeSetup`` whose decode cos/sin are height-sharded ``[1, B, 1(32), hd]`` (one user per
    core) for ``rotary_embedding_hf(is_decode_mode=True)``."""

    def __init__(self, mesh_device, args: Lfm2MoeArgs, max_seq_len=None, batch=TILE):
        super().__init__(mesh_device, args, max_seq_len=max_seq_len)
        self.head_mem = decode_head_mem(mesh_device, batch, self.head_dim)
        self._bucket_head_mems = {}

    def decode_cos_sin(self, rope_pos, batch=None):
        """``rope_pos`` uint32 ``[1, B]`` -> sharded cos/sin ``[1, B, 1, hd]``; ``batch`` < B: users
        0..batch-1 only (``[1, batch, 1, hd]`` on ``batch`` cores, for bucketed decode)."""
        head_mem, pos = self.head_mem, rope_pos
        if batch is not None and batch < rope_pos.shape[-1]:
            head_mem = self._bucket_head_mems.setdefault(batch, decode_head_mem(self.mesh_device, batch, self.head_dim))
            pos = ttnn.slice(rope_pos, (0, 0), (1, batch))
        out = []
        for table in (self.cos_rm, self.sin_rm):
            t = ttnn.embedding(pos, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, B, hd]
            t = ttnn.transpose(ttnn.unsqueeze_to_4D(t), 1, 2)  # [1, B, 1, hd]
            out.append(ttnn.interleaved_to_sharded(t, head_mem))
        if pos is not rope_pos:
            ttnn.deallocate(pos)
        return out

    def apply_hf(self, x, cos, sin, decode, ckc):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=decode, compute_kernel_config=ckc)


class FusedAttention(Attention):
    def __init__(self, mesh_device, args, sd, max_batch_size, dtype, fusions, sdpa_chunk=256):
        super().__init__(mesh_device, args, sd, max_batch_size, dtype, sdpa_chunk=sdpa_chunk)
        self.f = fusions
        if "qk_norm_weight" in fusions:
            # rms_norm weight must be row-major with last dim 32
            self.q_norm_rm = _as_tt(
                sd["self_attn.q_layernorm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.k_norm_rm = _as_tt(
                sd["self_attn.k_layernorm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        self.head_mem = decode_head_mem(mesh_device, self.B, self.HD)
        self.fold = "residual_fold_prefill" in fusions
        self.ones_h = _ones_row(mesh_device, self.H) if self.fold else None
        if "kv_fused_update" in fusions:
            grid = mesh_device.compute_with_storage_grid_size()
            # paged_fused_update_cache needs K and V shards on disjoint core grids
            self.v_update_mem = _height_shard(_row_major_cores(self.B, self.B, grid.x), (TILE, self.HD))
        # nlp_concat_heads_decode needs a rectangular B-core grid (rejects the row-major set above)
        self.concat_in_mem = _height_shard(ttnn.CoreGrid(y=self.B // 8, x=8), (TILE, self.HD))

    def _qkv_linear(self, x, mem):
        return ttnn.linear(x, self.wqkv, compute_kernel_config=self.ckc, memory_config=mem)

    def _out_project(self, attn, residual):
        return _project(attn, self.wo, residual, self.ones_h, self.ckc, self.fold)

    def _qk_norm_f(self, t, which, mem):
        if "qk_norm_weight" in self.f:
            w = self.q_norm_rm if which == "q" else self.k_norm_rm
            return ttnn.rms_norm(t, epsilon=self.eps, weight=w, memory_config=mem, compute_kernel_config=self.ckc_sdpa)
        return self._qk_norm(t, self.q_norm if which == "q" else self.k_norm, mem)

    def _rope(self, rope, t, cos, sin, decode):
        if "rope_hf" in self.f:
            return rope.apply_hf(t, cos, sin, decode, self.ckc_sdpa)
        return rope.apply(t, cos, sin, self.ckc_sdpa)

    def prefill_forward(self, x, *, rope, chunk_start, page_table, chunk_page_table, chunk_start_tensor, residual=None):
        S = x.shape[-2]
        block_size = self.k_cache.shape[2]
        assert S % TILE == 0 and chunk_start % block_size == 0, (
            f"chunk_start {chunk_start} must be a multiple of the KV block size {block_size} "
            f"(paged_fill_cache writes from block chunk_start // block_size); S_pad {S} a multiple of {TILE}"
        )
        qkv = self._qkv_linear(x, DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.NH, num_kv_heads=self.NKV, transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        q = self._qk_norm_f(q, "q", DRAM)
        k = self._qk_norm_f(k, "k", DRAM)
        cos, sin = rope.prefill_cos_sin(chunk_start, S)
        q = self._rope(rope, q, cos, sin, False)
        k = self._rope(rope, k, cos, sin, False)
        if k.dtype != self.k_cache.dtype:
            k = ttnn.typecast(k, self.k_cache.dtype)
            v = ttnn.typecast(v, self.v_cache.dtype)
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx=0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q_chunk = self.prefill_sdpa_chunk(chunk_start, S)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=q_chunk, k_chunk_size=q_chunk, exp_approx_mode=False
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            page_table,
            chunk_start_idx_tensor=chunk_start_tensor,
            scale=self.scale,
            program_config=cfg,
            compute_kernel_config=self.ckc_sdpa,
        )
        ttnn.deallocate(q)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=DRAM)
        out = self._out_project(attn, residual)
        ttnn.deallocate(attn)
        return out

    def decode_forward(self, x, *, rope_cos, rope_sin, rope, cur_pos, page_table, residual=None):
        if "qkv_heads_decode" not in self.f:
            assert "rope_hf" not in self.f, "rope_hf decode needs the sharded heads of qkv_heads_decode"
            return self._decode_unfused_heads(x, rope_cos, rope_sin, rope, cur_pos, page_table, residual)
        B, NH, HD = x.shape[-2], self.NH, self.HD
        qkv = self._qkv_linear(x, L1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NH, num_kv_heads=self.NKV, memory_config=self.head_mem
        )  # [1, B, 32(pad), HD] height-sharded, one user per core
        ttnn.deallocate(qkv)
        # per-head RMSNorm does not accept height-sharded inputs: L1 interleaved round trip
        q = ttnn.to_memory_config(self._qk_norm_f(ttnn.to_memory_config(q, L1), "q", L1), self.head_mem)
        k = ttnn.to_memory_config(self._qk_norm_f(ttnn.to_memory_config(k, L1), "k", L1), self.head_mem)
        if "rope_hf" in self.f:
            q = rope.apply_hf(q, rope_cos, rope_sin, True, self.ckc_sdpa)
            k = rope.apply_hf(k, rope_cos, rope_sin, True, self.ckc_sdpa)
        else:
            raise NotImplementedError("qkv_heads_decode is only wired with rope_hf")
        if "kv_fused_update" in self.f:
            v_d = ttnn.to_memory_config(v, self.v_update_mem)
            ttnn.deallocate(v)
            v = v_d
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache, k, self.v_cache, v, update_idxs_tensor=cur_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = self._sdpa_decode(q, cur_pos, page_table)
        ttnn.deallocate(q)
        return self._out_proj_decode(attn, B, residual)

    def _sdpa_decode(self, q, cur_pos, page_table):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=cfg,
            compute_kernel_config=self.ckc_sdpa,
            memory_config=L1,
        )

    def _out_proj_decode(self, attn, B, residual=None):
        if "concat_heads_decode" in self.f:
            attn_sh = ttnn.to_memory_config(attn, self.concat_in_mem)
            ttnn.deallocate(attn)
            attn = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=self.NH)  # width-sharded
            ttnn.deallocate(attn_sh)
            attn_i = ttnn.to_memory_config(attn, L1)
            ttnn.deallocate(attn)
            attn = attn_i
        else:
            attn = ttnn.reshape(attn, (1, 1, B, self.NH * self.HD))
        out = self._out_project(attn, residual)
        ttnn.deallocate(attn)
        return out

    def _decode_unfused_heads(self, x, rope_cos, rope_sin, rope, cur_pos, page_table, residual=None):
        B, NH, NKV, HD = x.shape[-2], self.NH, self.NKV, self.HD
        qkv = self._qkv_linear(x, L1)
        q = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, B, NH * HD)), (1, B, NH, HD))
        k = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, NH * HD), (1, 1, B, (NH + NKV) * HD)), (1, B, NKV, HD))
        v = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, (NH + NKV) * HD), (1, 1, B, (NH + 2 * NKV) * HD)), (1, B, NKV, HD))
        ttnn.deallocate(qkv)
        q = self._qk_norm_f(q, "q", L1)
        k = self._qk_norm_f(k, "k", L1)
        q = rope.apply(q, rope_cos, rope_sin, self.ckc_sdpa)
        k = rope.apply(k, rope_cos, rope_sin, self.ckc_sdpa)
        k_sh = ttnn.to_memory_config(ttnn.pad(k, [1, B, TILE, HD], [0, 0, 0, 0], 0.0), self.kv_update_mem)
        v_sh = ttnn.to_memory_config(ttnn.pad(v, [1, B, TILE, HD], [0, 0, 0, 0], 0.0), self.kv_update_mem)
        ttnn.experimental.paged_update_cache(self.k_cache, k_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(self.v_cache, v_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        attn = self._sdpa_decode(q, cur_pos, page_table)
        return self._out_proj_decode(attn, B, residual)


class FusedShortConv(ShortConv):
    def __init__(self, mesh_device, args, sd, max_batch_size, dtype, fusions):
        super().__init__(mesh_device, args, sd, max_batch_size, dtype)
        self.f = fusions
        self.fold = "residual_fold_prefill" in fusions
        self.ones_h = _ones_row(mesh_device, self.H) if self.fold else None
        if "conv_split_in_proj" in fusions:
            w = sd["conv.in_proj.weight"]  # [3H, H] rows ordered B, C, x
            H = self.H
            self.in_b, self.in_c, self.in_x = (
                _as_tt(w[i * H : (i + 1) * H].T, mesh_device, dtype=dtype) for i in range(3)
            )

    def _gated_input(self, x, mem):
        if "conv_split_in_proj" not in self.f:
            return super()._gated_input(x, mem)
        b = ttnn.linear(x, self.in_b, compute_kernel_config=self.ckc, memory_config=mem)
        xx = ttnn.linear(x, self.in_x, compute_kernel_config=self.ckc, memory_config=mem)
        bx = ttnn.multiply(b, xx, memory_config=mem)
        ttnn.deallocate(b)
        ttnn.deallocate(xx)
        c = ttnn.linear(x, self.in_c, compute_kernel_config=self.ckc, memory_config=mem)
        return bx, c

    def _out_project(self, y, residual):
        return _project(y, self.out_proj, residual, self.ones_h, self.ckc, self.fold)

    def prefill_forward(self, x, inputs, residual=None):
        """``ShortConv.prefill_forward`` with an optional residual added by the out projection."""
        S = x.shape[-2]
        H = self.H
        bx, c = self._gated_input(x, DRAM)
        carry = ttnn.matmul(inputs.carry_sel_0, self.state[0], compute_kernel_config=self.ckc)
        carry = ttnn.add(carry, ttnn.matmul(inputs.carry_sel_1, self.state[1], compute_kernel_config=self.ckc))
        ext = ttnn.concat([carry, bx], dim=2)
        ttnn.deallocate(carry)
        ttnn.deallocate(bx)
        hybrid = "conv_hybrid_shift" in self.f
        tile_shift = "conv_tile_shift" in self.f and not hybrid
        ext_rm = ext if tile_shift else ttnn.to_layout(ext, ttnn.ROW_MAJOR_LAYOUT)
        conv = None
        for k in range(3):
            start = TILE - 2 + k
            if hybrid and start % TILE == 0:
                shifted = ttnn.slice(ext, (0, 0, start, 0), (1, 1, start + S, H))  # tile-aligned: no layout change
            else:
                shifted = ttnn.slice(ext_rm, (0, 0, start, 0), (1, 1, start + S, H))
                if not tile_shift:
                    shifted = ttnn.to_layout(shifted, ttnn.TILE_LAYOUT)
            conv = ttnn.multiply(shifted, self.taps[k]) if conv is None else ttnn.mac(shifted, self.taps[k], conv)
            ttnn.deallocate(shifted)
        if not tile_shift:
            ttnn.deallocate(ext_rm)
        tail_src = ext
        if inputs.tail_window is not None:  # selection over a <= 64-row window instead of all 32 + S rows
            ws, we = inputs.tail_window
            tail_src = ttnn.slice(ext, (0, 0, ws, 0), (1, 1, we, H))
        for j, sel in enumerate((inputs.tail_sel_0, inputs.tail_sel_1)):
            row = ttnn.matmul(sel, tail_src, compute_kernel_config=self.ckc)
            new_state = ttnn.mac(self.state[j], inputs.keep_mask, row)
            ttnn.copy(new_state, self.state[j])
            ttnn.deallocate(row)
            ttnn.deallocate(new_state)
        if tail_src is not ext:
            ttnn.deallocate(tail_src)
        if inputs.tap_sel_0 is not None and getattr(self, "tap_buf", None) is not None:
            # serving prefix caching: the state at each inner snapshot boundary of this chunk (rows of ext selected
            # exactly like the tail rows above) into this layer's persistent tap buffers, row r = boundary r
            for j, sel in enumerate((inputs.tap_sel_0, inputs.tap_sel_1)):
                rows = ttnn.matmul(sel, ext, compute_kernel_config=self.ckc)
                ttnn.copy(rows, self.tap_buf[j])
                ttnn.deallocate(rows)
        ttnn.deallocate(ext)
        y = ttnn.multiply(c, conv)
        ttnn.deallocate(c)
        ttnn.deallocate(conv)
        out = self._out_project(y, residual)
        ttnn.deallocate(y)
        return out

    def decode_forward(self, x, slot_active, residual=None):
        if "conv_state_inplace" not in self.f:
            out = super().decode_forward(x, slot_active)
            if residual is None:
                return out
            h = ttnn.add(residual, out, memory_config=DRAM)
            ttnn.deallocate(out)
            return h
        bx, c = self._gated_input(x, L1)
        conv = ttnn.multiply(self.state[0], self.taps[0], memory_config=L1)
        conv = ttnn.mac(self.state[1], self.taps[1], conv)
        conv = ttnn.mac(bx, self.taps[2], conv)
        # elementwise in place: each element reads (state1, state0) before writing state0, then
        # (bx, state1) before writing state1
        ttnn.where(slot_active, self.state[1], self.state[0], output_tensor=self.state[0])
        ttnn.where(slot_active, bx, self.state[1], output_tensor=self.state[1])
        ttnn.deallocate(bx)
        y = ttnn.multiply(c, conv, memory_config=L1)
        ttnn.deallocate(c)
        ttnn.deallocate(conv)
        out = self._out_project(y, residual)
        ttnn.deallocate(y)
        return out


class FusedDenseMLP(DenseMLP):
    def __init__(self, mesh_device, sd, dtype, fusions):
        self.f = fusions
        self.ckc = _hifi2(mesh_device)
        w1, w3 = sd["feed_forward.w1.weight"].T, sd["feed_forward.w3.weight"].T  # [H, I]
        self.I = w1.shape[-1]
        H = w1.shape[0]
        self.fold = "residual_fold_prefill" in fusions
        self.ones_h = _ones_row(mesh_device, H) if self.fold else None
        if "dense_minimal_swiglu" in fusions:
            # fuse_swiglu consumes alternating [gate tile | up tile] column tiles
            packed = torch.stack([w1.reshape(H, -1, TILE), w3.reshape(H, -1, TILE)], dim=2).reshape(H, -1)
            self.w13_tiles = _as_tt(packed, mesh_device, dtype=dtype)
        elif "dense_merged_w13" in fusions:
            self.w13 = _as_tt(torch.cat([w1, w3], dim=-1), mesh_device, dtype=dtype)
        else:
            self.w1 = _as_tt(w1, mesh_device, dtype=dtype)
            self.w3 = _as_tt(w3, mesh_device, dtype=dtype)
        self.w2 = _as_tt(sd["feed_forward.w2.weight"].T, mesh_device, dtype=dtype)

    def _down_project(self, h, residual):
        return _project(h, self.w2, residual, self.ones_h, self.ckc, self.fold)

    def forward(self, x, residual=None):
        T = x.shape[-2]
        if "dense_minimal_swiglu" in self.f:
            h = ttnn.experimental.minimal_matmul(
                x,
                self.w13_tiles,
                compute_kernel_config=self.ckc,
                memory_config=DRAM,
                dtype=ttnn.bfloat16,
                fuse_swiglu=True,
            )
        elif "dense_merged_w13" in self.f:
            gu = ttnn.linear(x, self.w13, compute_kernel_config=self.ckc, memory_config=DRAM)
            g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, T, self.I))
            u = ttnn.slice(gu, (0, 0, 0, self.I), (1, 1, T, 2 * self.I))
            ttnn.deallocate(gu)
            h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
        else:
            g = ttnn.linear(x, self.w1, compute_kernel_config=self.ckc, memory_config=DRAM)
            u = ttnn.linear(x, self.w3, compute_kernel_config=self.ckc, memory_config=DRAM)
            h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
        out = self._down_project(h, residual)
        ttnn.deallocate(h)
        return out


class FusedMoE(MoE):
    ROUTE_ROWS = 512  # routing in row groups of at most this size (bias is pre-broadcast to it)

    def __init__(self, mesh_device, args, sd, dtype, fusions):
        super().__init__(mesh_device, args, sd, dtype)
        self.f = fusions
        if "moe_split_gate_up" in fusions:
            gate, up, _ = stacked_expert_weights(sd, self.E)  # [E, I, H]
            self.w_up = _as_tt(up.transpose(1, 2).unsqueeze(0), mesh_device, dtype=dtype)  # [1, E, H, I]
            self.w_gate = _as_tt(gate.transpose(1, 2).unsqueeze(0), mesh_device, dtype=dtype)
            del self.w_up_gate
        if "moe_grouped_topk" in fusions:
            bias = sd["feed_forward.expert_bias"].float().reshape(1, 1, 1, -1)
            # the op needs the bias at the full scores shape
            self.bias_rows = {
                n: _as_tt(bias.expand(1, 1, n, self.E), mesh_device, dtype=ttnn.float32)
                for n in (TILE, self.ROUTE_ROWS)
            }

    def _bias_for(self, n):
        if n in self.bias_rows:
            return self.bias_rows[n], False
        return ttnn.slice(self.bias_rows[self.ROUTE_ROWS], (0, 0, 0, 0), (1, 1, n, self.E)), True

    def route(self, x):
        if "moe_grouped_topk" not in self.f:
            if "router_tile_mask" in self.f:
                return self._route_tile_mask(x)
            if "router_fold" in self.f:
                return self._route_folded(x)
            return super().route(x)
        T = x.shape[-2]
        if T > self.ROUTE_ROWS:
            parts = [
                self.route(ttnn.slice(x, (0, 0, s, 0), (1, 1, min(s + self.ROUTE_ROWS, T), self.H)))
                for s in range(0, T, self.ROUTE_ROWS)
            ]
            out = ttnn.concat(parts, dim=2)
            for p in parts:
                ttnn.deallocate(p)
            return out
        xf = ttnn.typecast(x, ttnn.float32)
        logits = ttnn.linear(xf, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32)
        ttnn.deallocate(xf)
        bias, sliced = self._bias_for(T)
        w, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.K,
            route_scale=self.scale,
            epsilon=1e-6,
            score_func="sigmoid",
        )
        if sliced:
            ttnn.deallocate(bias)
        # dense [T, E] routing tensor: on-device zero fill (trace-safe) + row-major fp32 scatter
        rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        zeros = rm(ttnn.zeros_like(logits))
        ttnn.deallocate(logits)
        idx_rm = rm(idx) if idx.dtype == ttnn.uint32 else ttnn.typecast(rm(idx), ttnn.uint32)
        dense = ttnn.scatter(zeros, -1, idx_rm, rm(ttnn.typecast(w, ttnn.float32)))
        ttnn.deallocate(w)
        ttnn.deallocate(idx)
        return ttnn.typecast(ttnn.to_layout(dense, ttnn.TILE_LAYOUT), ttnn.bfloat16)

    def _mm_cfg(self, role, rows, n, groups):
        """Program config hook per sparse-matmul role ("gate_up", "down", "down_prefill")."""
        return _sparse_matmul_config(rows, n)

    def _router_scores(self, x):
        """fp32 sigmoid router scores [.., T, E]."""
        xf = ttnn.typecast(x, ttnn.float32)
        if "router_fold" in self.f:
            scores = ttnn.linear(
                xf, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32, activation="sigmoid"
            )
        else:
            scores = ttnn.sigmoid(
                ttnn.linear(xf, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32)
            )
        ttnn.deallocate(xf)
        return scores

    def _route_tile_mask(self, x):
        """Dense routing without gather/scatter or layout changes: the selected experts are those
        whose biased score is >= the k-th largest biased score (``topk`` values), identical to the
        top-k index set except on exact fp32 ties of the k-th and (k+1)-th biased scores, where
        more than k experts would be selected (all renormalized).  HF's ``torch.topk`` breaks such
        ties arbitrarily, so neither choice is canonical; not observed on 2x512 real tokens
        (doc/fused_decoder/logs/router_selection_probe.log)."""
        scores = self._router_scores(x)
        biased = ttnn.add(scores, self.bias)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1)
        ttnn.deallocate(idx)
        kth = ttnn.min(vals, dim=-1, keepdim=True)  # [T, 1]: k-th largest biased score
        ttnn.deallocate(vals)
        sel = ttnn.multiply(scores, ttnn.ge(biased, kth))  # unbiased scores of the selected experts
        ttnn.deallocate(biased)
        ttnn.deallocate(kth)
        ttnn.deallocate(scores)
        denom = ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6)
        w = ttnn.div(sel, denom)
        if self.scale != 1.0:
            w = ttnn.multiply(w, self.scale)
        ttnn.deallocate(sel)
        return ttnn.typecast(w, ttnn.bfloat16)

    def _route_folded(self, x):
        """``MoE.route`` with sigmoid fused into the router matmul and the x1.0 scale dropped."""
        xf = ttnn.typecast(x, ttnn.float32)
        scores = ttnn.linear(
            xf, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32, activation="sigmoid"
        )
        ttnn.deallocate(xf)
        biased = ttnn.add(scores, self.bias)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1)
        ttnn.deallocate(vals)
        ttnn.deallocate(biased)
        sel = ttnn.gather(scores, dim=-1, index=idx)
        denom = ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6)
        w = ttnn.div(sel, denom)
        if self.scale != 1.0:
            w = ttnn.multiply(w, self.scale)
        rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        dense = ttnn.scatter(rm(ttnn.zeros_like(scores)), -1, rm(idx), rm(w))
        ttnn.deallocate(scores)
        ttnn.deallocate(sel)
        ttnn.deallocate(idx)
        return ttnn.typecast(ttnn.to_layout(dense, ttnn.TILE_LAYOUT), ttnn.bfloat16)

    def _split_gate_up_act(self, x, sparsity, R, groups, prefill_sparse):
        """Two half-width sparse matmuls (no [.., 2I] intermediate, no slices); returns
        act = up * silu(gate) in the down-proj layout.  (SiLU as the gate matmul's
        ``fused_activation`` gave wrong results: MoE output PCC 0.984; see work_log.)"""
        I, E = self.I, self.E
        outs = []
        for w in (self.w_up, self.w_gate):
            cfg = _sparse_matmul_config(R, I)
            o = ttnn.sparse_matmul(
                x,
                w,
                sparsity=sparsity,
                nnz=None,
                memory_config=DRAM,
                output_tile=ttnn.Tile([32, 32]),
                program_config=cfg,
                compute_kernel_config=self.ckc,
                dtype=ttnn.bfloat16,
            )  # [1, G, 1, E, R, I]
            if groups == 1 or prefill_sparse:
                o = ttnn.reshape(o, (groups, E, R, I))
            else:
                o = ttnn.permute(ttnn.reshape(o, (groups, E, R, I)), (1, 0, 2, 3))
            outs.append(o)
        act = ttnn.multiply(outs[0], outs[1], memory_config=DRAM, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        for o in outs:
            ttnn.deallocate(o)
        return act

    def _experts(self, x, routing, sparsity_gate_up, rows_per_group, groups):
        T = groups * rows_per_group
        I, H, E = self.I, self.H, self.E
        if "moe_split_gate_up" in self.f:
            prefill_sparse = groups > 1 and "moe_prefill_down_sparse" in self.f
            act = self._split_gate_up_act(x, sparsity_gate_up, rows_per_group, groups, prefill_sparse)
            return self._down_and_combine(act, routing, sparsity_gate_up, rows_per_group, groups, prefill_sparse)
        up_gate = ttnn.sparse_matmul(
            x,
            self.w_up_gate,
            sparsity=sparsity_gate_up,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._mm_cfg("gate_up", rows_per_group, 2 * I, groups),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )  # [1, G, 1, E, R, 2I]
        prefill_sparse = groups > 1 and "moe_prefill_down_sparse" in self.f
        if groups == 1 and "moe_decode_no_permute" in self.f:
            up_gate = ttnn.reshape(up_gate, (1, E, rows_per_group, 2 * I))
        elif prefill_sparse:
            up_gate = ttnn.reshape(up_gate, (groups, E, rows_per_group, 2 * I))  # [G, E, R, 2I], no permute
        else:
            up_gate = ttnn.reshape(up_gate, (groups, E, rows_per_group, 2 * I))
            up_gate = ttnn.permute(up_gate, (1, 0, 2, 3))  # [E, G, R, 2I]
        if "moe_swiglu_mul_act" in self.f:
            shp = up_gate.shape
            up = ttnn.slice(up_gate, (0, 0, 0, 0), (shp[0], shp[1], shp[2], I))
            gate = ttnn.slice(up_gate, (0, 0, 0, I), (shp[0], shp[1], shp[2], 2 * I))
            ttnn.deallocate(up_gate)
            act = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
        else:
            act = ttnn.swiglu(up_gate, dim=-1)
            ttnn.deallocate(up_gate)
        return self._down_and_combine(act, routing, sparsity_gate_up, rows_per_group, groups, prefill_sparse)

    def _down_and_combine(self, act, routing, sparsity_gate_up, rows_per_group, groups, prefill_sparse):
        T = groups * rows_per_group
        I, H, E = self.I, self.H, self.E
        if prefill_sparse:
            return self._down_prefill_sparse(act, routing, sparsity_gate_up, rows_per_group, groups)
        act = ttnn.reshape(act, (1, E, T, I))
        if groups == 1 and "moe_down_union_sparsity" in self.f:
            down_sparsity, nnz = sparsity_gate_up, None
        else:
            down_sparsity, nnz = self.all_experts, E
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            sparsity=down_sparsity,
            nnz=nnz,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._mm_cfg("down", T, H, 1),
            compute_kernel_config=self.ckc,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )  # [1, E, T, H]
        ttnn.deallocate(act)
        down = ttnn.reshape(down, (1, E, T, H))
        weights = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, T, 1]
        down = ttnn.multiply(down, weights)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(out, (1, 1, T, H))


def _down_prefill_sparse_impl(self, act, routing, tile_mask, R, G):
    """``act`` [G, E, R, I]: down projection only for (tile, expert) pairs the tile routes to
    (``tile_mask`` [1, 1, G, E] row-major), weighted by routing and summed over experts."""
    E, H = self.E, self.H
    down = ttnn.sparse_matmul(
        act,
        self.w_down,
        sparsity=tile_mask,
        nnz=None,
        memory_config=DRAM,
        output_tile=ttnn.Tile([32, 32]),
        program_config=self._mm_cfg("down_prefill", R, H, G),
        compute_kernel_config=self.ckc,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        dtype=ttnn.bfloat16,
    )  # [G, E, R, H]; skipped (tile, expert) blocks are zero-filled
    ttnn.deallocate(act)
    down = ttnn.reshape(down, (G, E, R, H))
    w = ttnn.permute(ttnn.reshape(routing, (G, R, E, 1)), (0, 2, 1, 3))  # [G, E, R, 1]
    down_w = ttnn.multiply(down, w)
    ttnn.deallocate(down)
    out = ttnn.experimental.fast_reduce_nc(down_w, dims=[1])  # [G, 1, R, H]
    ttnn.deallocate(down_w)
    return ttnn.reshape(out, (1, 1, G * R, H))


FusedMoE._down_prefill_sparse = _down_prefill_sparse_impl


class FusedDecoder(FunctionalDecoder):
    """``FunctionalDecoder`` with the graph rewrites in ``fusions`` (default ``DEFAULT_FUSIONS``)."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=TILE,
        weight_dtype=ttnn.bfloat16,
        fusions=DEFAULT_FUSIONS,
        **kwargs,
    ):
        return cls(state_dict, hf_config, layer_idx, mesh_device, max_batch_size, weight_dtype, fusions, **kwargs)

    def __init__(self, sd, hf_config, layer_idx, mesh_device, max_batch_size, weight_dtype, fusions, sdpa_chunk=256):
        fusions = frozenset(fusions)
        unknown = fusions - set(ALL_FUSIONS)
        assert not unknown, f"unknown fusions {unknown}"
        self.fusions = fusions
        args = Lfm2MoeArgs.from_hf_config(hf_config)
        self.args = args
        self.layer_idx = layer_idx
        self.kind = args.layer_kind(layer_idx)
        self.mesh_device = mesh_device
        self.eps = args.norm_eps
        self.ckc_norm = _hifi4(mesh_device)
        self.operator_norm = _as_tt(
            sd["operator_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.ffn_norm = _as_tt(sd["ffn_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.is_attention = args.is_attention_layer(layer_idx)
        if self.is_attention:
            self.mixer = FusedAttention(
                mesh_device, args, sd, max_batch_size, weight_dtype, fusions, sdpa_chunk=sdpa_chunk
            )
        else:
            self.mixer = FusedShortConv(mesh_device, args, sd, max_batch_size, weight_dtype, fusions)
        if args.is_moe_layer(layer_idx):
            self.ffn = FusedMoE(mesh_device, args, sd, weight_dtype, fusions)
        else:
            self.ffn = FusedDenseMLP(mesh_device, sd, weight_dtype, fusions)

    def _ffn_block(self, h):
        hn = self._norm(h, self.ffn_norm, DRAM)
        if isinstance(self.ffn, FusedDenseMLP):
            out = self.ffn.forward(hn, residual=h)  # residual add folded (or applied) inside
            ttnn.deallocate(hn)
            return out
        f = self.ffn(hn)
        ttnn.deallocate(hn)
        out = ttnn.add(h, f, memory_config=DRAM)
        ttnn.deallocate(f)
        return out

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
            h = self.mixer.prefill_forward(
                xn,
                rope=rope,
                chunk_start=chunk_start,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_tensor=chunk_start_tensor,
                residual=x,
            )
        else:
            h = self.mixer.prefill_forward(xn, conv_inputs, residual=x)
        ttnn.deallocate(xn)
        out = self._ffn_block(h)
        ttnn.deallocate(h)
        return out

    def decode_forward(
        self, x, *, rope=None, rope_cos=None, rope_sin=None, cur_pos=None, page_table=None, slot_active=None
    ):
        xn = self._norm(x, self.operator_norm, L1)
        if self.is_attention:
            h = self.mixer.decode_forward(
                xn, rope=rope, rope_cos=rope_cos, rope_sin=rope_sin, cur_pos=cur_pos, page_table=page_table, residual=x
            )
        else:
            h = self.mixer.decode_forward(xn, slot_active, residual=x)
        ttnn.deallocate(xn)
        out = self._ffn_block(h)
        ttnn.deallocate(h)
        return out


def make_rope(mesh_device, args, max_seq_len, fusions=DEFAULT_FUSIONS):
    """Rope setup matching ``fusions`` (sharded decode cos/sin when ``rope_hf`` is on)."""
    if "rope_hf" in fusions:
        return FusedRope(mesh_device, args, max_seq_len=max_seq_len)
    return RopeSetup(mesh_device, args, max_seq_len=max_seq_len)
