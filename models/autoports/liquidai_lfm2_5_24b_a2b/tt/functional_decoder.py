# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for LiquidAI/LFM2.5-24B-A2B (HF ``Lfm2MoeDecoderLayer``).

Layer = ``h = x + mixer(operator_norm(x)); out = h + ffn(ffn_norm(h))`` where

* mixer is ``ShortConv`` (``layer_types[i] == "conv"``) or GQA ``Attention`` with per-head
  q/k RMSNorm and HF rotate-half RoPE (``"full_attention"``);
* ffn is a dense SwiGLU MLP for ``i < num_dense_layers`` and otherwise a sigmoid-routed
  top-k MoE with selection-only expert bias (``Lfm2MoeSparseMoeBlock``).

Prefill contract (one request per call, one physical chunk per call)::

    prefill_forward(x, *, rope, chunk_start, page_table, chunk_page_table,
                    chunk_start_tensor, conv_inputs) -> [1, 1, S_pad, H]

  ``x`` is ``[1, 1, S_pad, H]`` (``S_pad % 32 == 0``) holding the chunk's tokens at absolute
  positions ``chunk_start ... chunk_start + S_pad - 1``; rows past the logical length are
  padding.  Padding never influences valid rows (every op is causal or row-local).  The caller
  (generator) owns padding, chunk planning and last-token selection.

  * attention (uses ``rope``/``chunk_start``/page tables): K/V of the chunk are written with
    ``paged_fill_cache`` through ``chunk_page_table`` (the request's blocks starting at
    ``chunk_start // block_size``), then ``chunked_scaled_dot_product_attention`` reads the
    accumulated prefix through the full ``page_table`` using ``chunk_start_tensor`` (int32
    ``[1]``).  ``chunk_start`` must be a multiple of the KV block size (asserted); the SDPA
    chunk is chosen internally to divide ``chunk_start`` and ``S_pad``.  Page tables are int32
    rows padded to a multiple of 8 entries (32 bytes) and must cover ``S_pad`` rows (padding
    rows land in blocks the request owns or in a scratch block); ``paged_update_cache`` also
    needs ``num_blocks >= page_table width``.
  * ShortConv (uses ``conv_inputs``): conv state is owned by the layer as two persistent
    ``[1, 1, 32, H]`` taps (row = decode batch slot).  ``conv_inputs`` (``ConvPrefillInputs``)
    are small host-built selection tensors: ``carry_sel_{0,1}`` pull the slot's previous two
    rows (zero for a fresh request, which is how a new request resets its slot) into the
    32-row prefix tile, ``tail_sel_{0,1}`` pick the last two logical rows of the chunk into the
    slot row, and ``keep_mask`` preserves every other slot.  All of this runs on device, so
    prefill writes the per-slot state without host round trips.

Decode contract (all slots at once, trace-safe)::

    decode_forward(x, *, rope, rope_cos, rope_sin, cur_pos, page_table, slot_active) -> [1, 1, B, H]

  ``x`` is ``[1, 1, B, H]`` with ``B == max_batch_size`` (32) rows, one per slot.  Attention
  layers also take ``cur_pos`` int32 ``[B]`` (absolute position of the new token; ``-1`` skips
  a slot's KV write and attention), ``page_table`` int32 ``[B, max_blocks_per_seq]`` and
  ``rope_cos/rope_sin`` from ``RopeSetup.decode_cos_sin`` (``rope`` supplies the rotation);
  ShortConv layers take ``slot_active`` (bf16 ``[1, 1, B, 1]``, 1 where the slot decodes this
  step, i.e. ``cur_pos >= 0``): active rows shift their taps, idle rows keep their state
  exactly (``ttnn.where``), so a request may be mid-prefill or waiting while others decode.

Every tensor used at runtime is created in ``from_state_dict``/``allocate_*``; the forward
passes contain only device ops (no torch / from_torch / to_torch).
"""

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .model_config import TILE, Lfm2MoeArgs
from .weights import stacked_expert_weights

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def _hifi4(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _hifi2(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _as_tt(t, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=DRAM):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=_mapper(mesh_device),
    )


# --------------------------------------------------------------------------------------------
# RoPE (HF default rope, rotate-half convention)
# --------------------------------------------------------------------------------------------


class RopeSetup(LightweightModule):
    """Device-resident cos/sin tables for all ``max_position_embeddings`` positions.

    HF computes ``cos/sin`` in fp32 and casts to the activation dtype (bf16) before use; the
    tables here are the same bf16 values.  ``rotate_half(x) == x @ rot`` with a signed
    permutation matrix, so ``apply`` is ``x * cos + (x @ rot) * sin``.
    """

    def __init__(self, mesh_device, args: Lfm2MoeArgs, max_seq_len=None):
        self.mesh_device = mesh_device
        self.head_dim = hd = args.head_dim
        # tile-padded prefill chunks read up to the next tile boundary (padding rows are discarded)
        self.max_seq_len = math.ceil((max_seq_len or args.max_position_embeddings) / TILE) * TILE
        inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
        pos = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)
        # [max_seq, hd] row-major tables for ttnn.embedding (decode) + tile [1,1,max_seq,hd] (prefill)
        self.cos_rm = _as_tt(cos, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin_rm = _as_tt(sin, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.cos_tile = _as_tt(cos.reshape(1, 1, self.max_seq_len, hd), mesh_device)
        self.sin_tile = _as_tt(sin.reshape(1, 1, self.max_seq_len, hd), mesh_device)
        rot = torch.zeros(hd, hd)
        half = hd // 2
        for j in range(half):
            rot[j + half, j] = -1.0  # out[j] = -x[j + half]
            rot[j, j + half] = 1.0  # out[j + half] = x[j]
        self.rot = _as_tt(rot.reshape(1, 1, hd, hd), mesh_device)

    def prefill_cos_sin(self, start, length):
        """cos/sin ``[1, 1, length, hd]`` for absolute positions ``start .. start+length-1``."""
        end = start + length
        assert end <= self.max_seq_len, (start, length, self.max_seq_len)
        cos = ttnn.slice(self.cos_tile, (0, 0, start, 0), (1, 1, end, self.head_dim))
        sin = ttnn.slice(self.sin_tile, (0, 0, start, 0), (1, 1, end, self.head_dim))
        return cos, sin

    def decode_cos_sin(self, rope_pos):
        """``rope_pos``: uint32 ``[1, B]`` row-major device tensor -> cos/sin ``[1, B, 1, hd]``."""
        b = rope_pos.shape[-1]
        cos = ttnn.embedding(rope_pos, self.cos_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        sin = ttnn.embedding(rope_pos, self.sin_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        cos = ttnn.reshape(cos, (1, b, 1, self.head_dim))
        sin = ttnn.reshape(sin, (1, b, 1, self.head_dim))
        return cos, sin

    def apply(self, x, cos, sin, ckc):
        rotated = ttnn.matmul(x, self.rot, compute_kernel_config=ckc)
        out = ttnn.multiply(x, cos)
        out = ttnn.mac(rotated, sin, out)
        ttnn.deallocate(rotated)
        return out


# --------------------------------------------------------------------------------------------
# Mixers
# --------------------------------------------------------------------------------------------


@dataclass
class ConvPrefillInputs:
    """Host-built selection tensors for one ShortConv prefill chunk (shared by all conv layers).

    carry_sel_{0,1}: ``[1, 1, 32, B]`` -- row 30 (resp. 31) is one-hot at ``slot`` for a
                     continuation chunk, all-zero for a fresh request.
    tail_sel_{0,1}:  ``[1, 1, B, 32 + S_pad]`` -- row ``slot`` is one-hot at extended row
                     ``32 + valid_len - 2`` (resp. ``- 1``); extended rows 30/31 are the carry.
    keep_mask:       ``[1, 1, B, 1]`` -- 0 at ``slot``, 1 elsewhere.
    tail_window:     ``None``, or ``(start, end)`` extended rows when built with ``windowed=True``:
                     ``tail_sel_*`` then index only that tile-aligned window (<= 64 rows) holding
                     the last two logical rows (consumers that support it slice ``ext`` first).
    tap_sel_{0,1}:   optional ``[1, 1, 32, 32 + S_pad]`` -- row ``r`` is one-hot at extended row
                     ``32 + b_r - 2`` (resp. ``- 1``) for each inner snapshot boundary ``b_r`` of the
                     chunk (serving prefix caching; consumers with tap buffers persist those rows).
    """

    carry_sel_0: ttnn.Tensor
    carry_sel_1: ttnn.Tensor
    tail_sel_0: ttnn.Tensor
    tail_sel_1: ttnn.Tensor
    keep_mask: ttnn.Tensor
    tail_window: tuple = None
    tap_sel_0: ttnn.Tensor = None
    tap_sel_1: ttnn.Tensor = None

    @staticmethod
    def tail_window_for(valid_len, padded_len):
        start = (TILE + valid_len - 2) // TILE * TILE
        return start, min(start + 2 * TILE, TILE + padded_len)

    @staticmethod
    def host_tensors(*, slot, valid_len, padded_len, continuation, batch=TILE, windowed=False):
        assert 0 < valid_len <= padded_len and padded_len % TILE == 0, (valid_len, padded_len)
        carry0 = torch.zeros(1, 1, TILE, batch)
        carry1 = torch.zeros(1, 1, TILE, batch)
        if continuation:
            carry0[0, 0, TILE - 2, slot] = 1.0
            carry1[0, 0, TILE - 1, slot] = 1.0
        start, end = ConvPrefillInputs.tail_window_for(valid_len, padded_len) if windowed else (0, TILE + padded_len)
        tail0 = torch.zeros(1, 1, batch, end - start)
        tail1 = torch.zeros(1, 1, batch, end - start)
        tail0[0, 0, slot, TILE + valid_len - 2 - start] = 1.0
        tail1[0, 0, slot, TILE + valid_len - 1 - start] = 1.0
        keep = torch.ones(1, 1, batch, 1)
        keep[0, 0, slot, 0] = 0.0
        return carry0, carry1, tail0, tail1, keep

    @classmethod
    def create(cls, mesh_device, **kw):
        tensors = cls.host_tensors(**kw)
        window = cls.tail_window_for(kw["valid_len"], kw["padded_len"]) if kw.get("windowed") else None
        return cls(*[_as_tt(t, mesh_device) for t in tensors], tail_window=window)


class ShortConv(LightweightModule):
    """``in_proj -> B*x -> depthwise causal conv(K=3) -> C*conv -> out_proj`` with device state."""

    def __init__(self, mesh_device, args: Lfm2MoeArgs, sd, max_batch_size, dtype):
        assert args.conv_kernel == 3 and not args.conv_bias, "kernel layout below assumes K=3, no bias"
        assert max_batch_size == TILE, "decode slots are the 32 rows of one tile"
        self.mesh_device = mesh_device
        self.H = args.hidden_size
        self.B = max_batch_size
        self.ckc = _hifi2(mesh_device)
        self.in_proj = _as_tt(sd["conv.in_proj.weight"].T, mesh_device, dtype=dtype)  # [H, 3H]
        self.out_proj = _as_tt(sd["conv.out_proj.weight"].T, mesh_device, dtype=dtype)  # [H, H]
        w = sd["conv.conv.weight"][:, 0, :]  # [H, K]; tap k multiplies x[t - (K-1-k)]
        self.taps = [_as_tt(w[:, k].reshape(1, 1, 1, -1), mesh_device) for k in range(3)]
        zeros = torch.zeros(1, 1, self.B, self.H)
        self.state = [_as_tt(zeros, mesh_device) for _ in range(2)]  # x[t-2], x[t-1] per slot row

    def _gated_input(self, x, mem):
        bcx = ttnn.linear(x, self.in_proj, compute_kernel_config=self.ckc, memory_config=mem)
        rows = x.shape[-2]
        H = self.H
        b = ttnn.slice(bcx, (0, 0, 0, 0), (1, 1, rows, H), memory_config=mem)
        c = ttnn.slice(bcx, (0, 0, 0, H), (1, 1, rows, 2 * H), memory_config=mem)
        xx = ttnn.slice(bcx, (0, 0, 0, 2 * H), (1, 1, rows, 3 * H), memory_config=mem)
        ttnn.deallocate(bcx)
        bx = ttnn.multiply(b, xx, memory_config=mem)
        ttnn.deallocate(b)
        ttnn.deallocate(xx)
        return bx, c

    def prefill_forward(self, x, inputs: ConvPrefillInputs):
        S = x.shape[-2]
        H = self.H
        bx, c = self._gated_input(x, DRAM)
        # 32-row prefix tile: rows 30/31 = the slot's previous two inputs (zeros if fresh)
        carry = ttnn.matmul(inputs.carry_sel_0, self.state[0], compute_kernel_config=self.ckc)
        carry = ttnn.add(carry, ttnn.matmul(inputs.carry_sel_1, self.state[1], compute_kernel_config=self.ckc))
        ext = ttnn.concat([carry, bx], dim=2)  # [1,1,32+S,H]
        ttnn.deallocate(carry)
        ttnn.deallocate(bx)
        ext_rm = ttnn.to_layout(ext, ttnn.ROW_MAJOR_LAYOUT)
        conv = None
        for k in range(3):
            start = TILE - 2 + k  # tap k reads x[t - (2 - k)]
            shifted = ttnn.to_layout(ttnn.slice(ext_rm, (0, 0, start, 0), (1, 1, start + S, H)), ttnn.TILE_LAYOUT)
            conv = ttnn.multiply(shifted, self.taps[k]) if conv is None else ttnn.mac(shifted, self.taps[k], conv)
            ttnn.deallocate(shifted)
        ttnn.deallocate(ext_rm)
        # persist the slot's last two logical rows (on device; other slots untouched)
        for j, sel in enumerate((inputs.tail_sel_0, inputs.tail_sel_1)):
            row = ttnn.matmul(sel, ext, compute_kernel_config=self.ckc)  # [1,1,B,H], zero except slot
            new_state = ttnn.mac(self.state[j], inputs.keep_mask, row)
            ttnn.copy(new_state, self.state[j])
            ttnn.deallocate(row)
            ttnn.deallocate(new_state)
        ttnn.deallocate(ext)
        y = ttnn.multiply(c, conv)
        ttnn.deallocate(c)
        ttnn.deallocate(conv)
        out = ttnn.linear(y, self.out_proj, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(y)
        return out

    def decode_forward(self, x, slot_active):
        """``slot_active`` ``[1, 1, B, 1]`` (1 = slot decodes this step): only active slots shift
        their taps, so idle slots (mid-prefill, waiting, free) keep their state exactly."""
        bx, c = self._gated_input(x, L1)
        conv = ttnn.multiply(self.state[0], self.taps[0], memory_config=L1)
        conv = ttnn.mac(self.state[1], self.taps[1], conv)
        conv = ttnn.mac(bx, self.taps[2], conv)
        new0 = ttnn.where(slot_active, self.state[1], self.state[0], memory_config=L1)
        new1 = ttnn.where(slot_active, bx, self.state[1], memory_config=L1)
        ttnn.deallocate(bx)
        ttnn.copy(new0, self.state[0])
        ttnn.copy(new1, self.state[1])
        ttnn.deallocate(new0)
        ttnn.deallocate(new1)
        y = ttnn.multiply(c, conv, memory_config=L1)
        ttnn.deallocate(c)
        ttnn.deallocate(conv)
        out = ttnn.linear(y, self.out_proj, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(y)
        return out


class Attention(LightweightModule):
    """GQA attention with per-head q/k RMSNorm, rotate-half RoPE and a paged KV cache."""

    def __init__(self, mesh_device, args: Lfm2MoeArgs, sd, max_batch_size, dtype, sdpa_chunk=256):
        self.mesh_device = mesh_device
        self.args = args
        self.NH, self.NKV, self.HD = args.num_heads, args.num_kv_heads, args.head_dim
        self.H = args.hidden_size
        self.B = max_batch_size
        self.eps = args.norm_eps
        self.scale = self.HD**-0.5
        self.ckc = _hifi2(mesh_device)
        self.ckc_sdpa = _hifi4(mesh_device)
        wqkv = torch.cat(
            [sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"], sd["self_attn.v_proj.weight"]], dim=0
        )
        self.wqkv = _as_tt(wqkv.T, mesh_device, dtype=dtype)  # [H, (NH+2NKV)*HD]
        self.wo = _as_tt(sd["self_attn.out_proj.weight"].T, mesh_device, dtype=dtype)
        self.q_norm = _as_tt(sd["self_attn.q_layernorm.weight"].reshape(1, 1, 1, -1), mesh_device)
        self.k_norm = _as_tt(sd["self_attn.k_layernorm.weight"].reshape(1, 1, 1, -1), mesh_device)
        self.sdpa_chunk = sdpa_chunk
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.kv_update_mem = ttnn.create_sharded_memory_config(
            shape=(TILE, self.HD),
            core_grid=ttnn.CoreGrid(x=8, y=self.B // 8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.k_cache = None
        self.v_cache = None

    def allocate_kv_cache(self, num_blocks, block_size, dtype=ttnn.bfloat16):
        shape = (num_blocks, self.NKV, block_size, self.HD)
        self.k_cache = ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=DRAM
        )
        self.v_cache = ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=DRAM
        )

    def set_kv_cache(self, k_cache, v_cache):
        self.k_cache, self.v_cache = k_cache, v_cache

    def prefill_sdpa_chunk(self, chunk_start, S):
        """Largest power-of-two SDPA chunk <= ``sdpa_chunk`` dividing both the chunk start (the
        chunked-SDPA offset contract) and the padded chunk length; at least one tile."""
        c = self.sdpa_chunk
        while c > TILE and (chunk_start % c or S % c):
            c //= 2
        return c

    def _qk_norm(self, t, weight, mem):
        t = ttnn.rms_norm(t, epsilon=self.eps, memory_config=mem, compute_kernel_config=self.ckc_sdpa)
        return ttnn.multiply(t, weight, memory_config=mem)

    def prefill_forward(self, x, *, rope, chunk_start, page_table, chunk_page_table, chunk_start_tensor):
        S = x.shape[-2]
        block_size = self.k_cache.shape[2]
        assert S % TILE == 0 and chunk_start % block_size == 0, (
            f"chunk_start {chunk_start} must be a multiple of the KV block size {block_size} "
            f"(paged_fill_cache writes from block chunk_start // block_size); S_pad {S} a multiple of {TILE}"
        )
        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.ckc, memory_config=DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.NH, num_kv_heads=self.NKV, transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        q = self._qk_norm(q, self.q_norm, DRAM)
        k = self._qk_norm(k, self.k_norm, DRAM)
        cos, sin = rope.prefill_cos_sin(chunk_start, S)
        q = rope.apply(q, cos, sin, self.ckc_sdpa)
        k = rope.apply(k, cos, sin, self.ckc_sdpa)
        if k.dtype != self.k_cache.dtype:
            k = ttnn.typecast(k, self.k_cache.dtype)
            v = ttnn.typecast(v, self.v_cache.dtype)
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx=0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q_chunk = self.prefill_sdpa_chunk(chunk_start, S)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=q_chunk,
            k_chunk_size=q_chunk,
            exp_approx_mode=False,
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
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(attn)
        return out

    def decode_forward(self, x, *, rope_cos, rope_sin, rope, cur_pos, page_table):
        B, NH, NKV, HD = x.shape[-2], self.NH, self.NKV, self.HD
        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.ckc, memory_config=L1)
        q = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, B, NH * HD)), (1, B, NH, HD))
        k = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, NH * HD), (1, 1, B, (NH + NKV) * HD)), (1, B, NKV, HD))
        v = ttnn.reshape(ttnn.slice(qkv, (0, 0, 0, (NH + NKV) * HD), (1, 1, B, (NH + 2 * NKV) * HD)), (1, B, NKV, HD))
        ttnn.deallocate(qkv)
        q = self._qk_norm(q, self.q_norm, L1)
        k = self._qk_norm(k, self.k_norm, L1)
        q = rope.apply(q, rope_cos, rope_sin, self.ckc_sdpa)
        k = rope.apply(k, rope_cos, rope_sin, self.ckc_sdpa)
        k_p = ttnn.pad(k, [1, B, TILE, HD], [0, 0, 0, 0], 0.0)
        v_p = ttnn.pad(v, [1, B, TILE, HD], [0, 0, 0, 0], 0.0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        k_sh = ttnn.to_memory_config(k_p, self.kv_update_mem)
        v_sh = ttnn.to_memory_config(v_p, self.kv_update_mem)
        ttnn.deallocate(k_p)
        ttnn.deallocate(v_p)
        ttnn.experimental.paged_update_cache(self.k_cache, k_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(self.v_cache, v_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(k_sh)
        ttnn.deallocate(v_sh)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
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
        ttnn.deallocate(q)
        attn = ttnn.reshape(attn, (1, 1, B, NH * HD))
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(attn)
        return out


# --------------------------------------------------------------------------------------------
# Feed-forward
# --------------------------------------------------------------------------------------------


class DenseMLP(LightweightModule):
    def __init__(self, mesh_device, sd, dtype):
        self.ckc = _hifi2(mesh_device)
        self.w1 = _as_tt(sd["feed_forward.w1.weight"].T, mesh_device, dtype=dtype)
        self.w3 = _as_tt(sd["feed_forward.w3.weight"].T, mesh_device, dtype=dtype)
        self.w2 = _as_tt(sd["feed_forward.w2.weight"].T, mesh_device, dtype=dtype)

    def forward(self, x):
        g = ttnn.linear(x, self.w1, compute_kernel_config=self.ckc, memory_config=DRAM)
        u = ttnn.linear(x, self.w3, compute_kernel_config=self.ckc, memory_config=DRAM)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, self.w2, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(h)
        return out


def _sparse_matmul_config(m, n):
    """1D mcast program config for ``sparse_matmul`` (largest divisor of N tiles on an 8x8 grid)."""
    n_tiles = (n + TILE - 1) // TILE
    best, best_xy = 1, (1, 1)
    for cores in range(1, min(64, n_tiles) + 1):
        if n_tiles % cores:
            continue
        for cy in range(1, 9):
            if cores % cy == 0 and cores // cy <= 8:
                if cores > best:
                    best, best_xy = cores, (cores // cy, cy)
                break
    per_core_n = n_tiles // best
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*best_xy),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class MoE(LightweightModule):
    """Sigmoid router + selection-only expert bias + active-expert ``sparse_matmul`` experts.

    Routing (HF ``Lfm2MoeTopKRouter``)::

        s = sigmoid(x @ Wg^T); idx = topk(s + bias, k); w = s[idx] / (sum(s[idx]) + 1e-6) * scale

    The dense routing tensor ``[1, 1, T, E]`` (zeros off the selected experts) is both the
    sparsity mask for ``sparse_matmul`` and the per-expert combine weight.
    """

    PREFILL_GROUP = 512  # tokens per grouped sparse_matmul in prefill

    def __init__(self, mesh_device, args: Lfm2MoeArgs, sd, dtype):
        self.mesh_device = mesh_device
        self.E = args.num_experts
        self.K = args.num_experts_per_tok
        self.I = args.moe_intermediate_size
        self.H = args.hidden_size
        self.scale = args.routed_scaling_factor
        assert args.norm_topk_prob and args.use_expert_bias
        self.ckc_router = _hifi4(mesh_device)
        self.ckc = _hifi2(mesh_device)
        self.router = _as_tt(sd["feed_forward.gate.weight"].T, mesh_device, dtype=ttnn.float32)  # [H, E]
        self.bias = _as_tt(sd["feed_forward.expert_bias"].float().reshape(1, 1, 1, -1), mesh_device, dtype=ttnn.float32)
        gate, up, down = stacked_expert_weights(sd, self.E)  # [E,I,H], [E,I,H], [E,H,I]
        # sparse_matmul weights: [1, E, K, N]; gate/up fused as [up | gate] for ttnn.swiglu
        up_gate = torch.cat([up.transpose(1, 2), gate.transpose(1, 2)], dim=-1).unsqueeze(0)  # [1,E,H,2I]
        self.w_up_gate = _as_tt(up_gate, mesh_device, dtype=dtype)
        self.w_down = _as_tt(down.transpose(1, 2).unsqueeze(0), mesh_device, dtype=dtype)  # [1,E,I,H]
        self.all_experts = _as_tt(torch.ones(1, 1, 1, self.E), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    def route(self, x):
        """x ``[1,1,T,H]`` -> dense routing weights ``[1,1,T,E]`` (bf16)."""
        xf = ttnn.typecast(x, ttnn.float32)
        logits = ttnn.linear(xf, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32)
        ttnn.deallocate(xf)
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        biased = ttnn.add(scores, self.bias)
        # fp32 topk: bf16-rounded scores pick a different expert set for ~5% of tokens (near ties)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1)
        ttnn.deallocate(vals)
        ttnn.deallocate(biased)
        sel = ttnn.gather(scores, dim=-1, index=idx)
        denom = ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6)
        w = ttnn.multiply(ttnn.div(sel, denom), self.scale)
        # fp32 scatter is only supported in row-major layout
        rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        dense = ttnn.scatter(rm(ttnn.zeros_like(scores)), -1, rm(idx), rm(w))
        ttnn.deallocate(scores)
        ttnn.deallocate(sel)
        ttnn.deallocate(idx)
        return ttnn.typecast(ttnn.to_layout(dense, ttnn.TILE_LAYOUT), ttnn.bfloat16)

    def _experts(self, x, routing, sparsity_gate_up, rows_per_group, groups):
        """x ``[1,G,R,H]`` grouped tokens; returns ``[1,1,G*R,H]``."""
        T = groups * rows_per_group
        I, H, E = self.I, self.H, self.E
        up_gate = ttnn.sparse_matmul(
            x,
            self.w_up_gate,
            sparsity=sparsity_gate_up,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(rows_per_group, 2 * I),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )  # [1, G, 1, E, R, 2I]
        up_gate = ttnn.reshape(up_gate, (groups, E, rows_per_group, 2 * I))
        up_gate = ttnn.permute(up_gate, (1, 0, 2, 3))  # [E, G, R, 2I]
        act = ttnn.swiglu(up_gate, dim=-1)
        ttnn.deallocate(up_gate)
        act = ttnn.reshape(act, (1, E, T, I))
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            sparsity=self.all_experts,
            nnz=E,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(T, H),
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

    def forward(self, x):
        T = x.shape[-2]
        routing = self.route(x)
        if T <= TILE:
            # decode: union of the rows' experts gates gate/up; down is weighted per row
            sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
            out = self._experts(ttnn.reshape(x, (1, 1, T, self.H)), routing, sparsity, T, 1)
            ttnn.deallocate(routing)
            return out
        outs = []
        for s0 in range(0, T, self.PREFILL_GROUP):
            n = min(self.PREFILL_GROUP, T - s0)
            g = n // TILE
            xc = ttnn.slice(x, (0, 0, s0, 0), (1, 1, s0 + n, self.H))
            rc = ttnn.slice(routing, (0, 0, s0, 0), (1, 1, s0 + n, self.E))
            tile_mask = ttnn.max(ttnn.reshape(rc, (1, g, TILE, self.E)), dim=2, keepdim=True)
            sparsity = ttnn.reshape(ttnn.to_layout(tile_mask, ttnn.ROW_MAJOR_LAYOUT), (1, 1, g, self.E))
            outs.append(self._experts(ttnn.reshape(xc, (1, g, TILE, self.H)), rc, sparsity, TILE, g))
            ttnn.deallocate(rc)
        ttnn.deallocate(routing)
        if len(outs) == 1:
            return outs[0]
        out = ttnn.concat(outs, dim=2)
        for o in outs:
            ttnn.deallocate(o)
        return out


# --------------------------------------------------------------------------------------------
# Decoder layer
# --------------------------------------------------------------------------------------------


class FunctionalDecoder(LightweightModule):
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
        **kwargs,
    ):
        return cls(state_dict, hf_config, layer_idx, mesh_device, max_batch_size, weight_dtype, **kwargs)

    def __init__(self, sd, hf_config, layer_idx, mesh_device, max_batch_size, weight_dtype, sdpa_chunk=256):
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
            self.mixer = Attention(mesh_device, args, sd, max_batch_size, weight_dtype, sdpa_chunk=sdpa_chunk)
        else:
            self.mixer = ShortConv(mesh_device, args, sd, max_batch_size, weight_dtype)
        if args.is_moe_layer(layer_idx):
            self.ffn = MoE(mesh_device, args, sd, weight_dtype)
        else:
            self.ffn = DenseMLP(mesh_device, sd, weight_dtype)

    def _norm(self, x, w, mem):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, memory_config=mem, compute_kernel_config=self.ckc_norm)

    def _ffn_block(self, h):
        hn = self._norm(h, self.ffn_norm, DRAM)
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
            m = self.mixer.prefill_forward(
                xn,
                rope=rope,
                chunk_start=chunk_start,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_tensor=chunk_start_tensor,
            )
        else:
            m = self.mixer.prefill_forward(xn, conv_inputs)
        ttnn.deallocate(xn)
        h = ttnn.add(x, m, memory_config=DRAM)
        ttnn.deallocate(m)
        out = self._ffn_block(h)
        ttnn.deallocate(h)
        return out

    def decode_forward(
        self, x, *, rope=None, rope_cos=None, rope_sin=None, cur_pos=None, page_table=None, slot_active=None
    ):
        xn = self._norm(x, self.operator_norm, L1)
        if self.is_attention:
            m = self.mixer.decode_forward(
                xn, rope=rope, rope_cos=rope_cos, rope_sin=rope_sin, cur_pos=cur_pos, page_table=page_table
            )
        else:
            m = self.mixer.decode_forward(xn, slot_active)
        ttnn.deallocate(xn)
        h = ttnn.add(x, m, memory_config=DRAM)
        ttnn.deallocate(m)
        out = self._ffn_block(h)
        ttnn.deallocate(h)
        return out
