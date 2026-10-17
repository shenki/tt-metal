# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full LFM2.5-24B-A2B model on the 1x4 Blackhole ring (TP = 4), stage 6.

Device-only compute around the stage-5 ``MultichipDecoder`` stack (``MC_POLICY`` / ``DEFAULT_MESH_PLAN`` unchanged):

* token embedding: the tied ``embed_tokens`` table, replicated bf16 ROW_MAJOR ``[128000, 2048]`` per chip; the decode
  residual enters the stack once, converted to the decoder's width-sharded ``resid_mem`` (the inter-layer contract);
* 40 decoder layers (2 conv_dense, 28 conv_moe, 10 attn_moe), one shared ``FusedRope``; decode cos/sin computed once per
  step (b-user for bucketed decode);
* final norm = HF ``embedding_norm`` (applied after the stack, ``modeling_lfm2_moe.py`` ``Lfm2MoeModel.forward``):
  decode ``ShardedNorm`` on ``resid_mem`` (= the LM-head input layout), prefill interleaved ``rms_norm``;
* LM head = ``embed_tokens.T`` (``tie_word_embeddings``), vocab-parallel: chip d holds vocab rows ``32256 d .. 32256 d +
  32255`` (vocab zero-padded at its tail to 4 x 32256, ``HEAD_COL_UNIT``) as a DRAM-sharded ``[2048, 32256]`` matmul;
  logits stay vocab-sharded (the sampler's input contract; the sampler masks the padded tail).

The model owns the persistent decode inputs (token, positions, RoPE index, page table, slot masks): host code refreshes
them with ``copy_host_to_device_tensor`` outside traces; ``decode_forward_device`` only reads them and advances
``cur_pos`` / ``rope_pos`` on device.  Prefill runs in chunks of at most ``PREFILL_CHUNK`` rows padded to a physical
bucket (``PREFILL_BUCKETS``); positions, page tables, chunk start and the ShortConv tail selection are runtime tensors,
so the compiled program set is finite (bucket x {first, continuation} chunk).
"""

import contextlib
import math
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

from .functional_decoder import DRAM, TILE, ConvPrefillInputs, _as_tt, _hifi4
from .fused_decoder import DEFAULT_FUSIONS, make_rope
from .model_config import Lfm2MoeArgs, load_hf_config
from .multichip_decoder import DEFAULT_MESH_PLAN, MC_POLICY, MultichipDecoder, _ds_host, _shard_like
from .optimized_decoder import DTYPES, DramShardedLinear, ShardedNorm, _ckc
from .weights import WeightCache, load_global_state_dict, load_layer_state_dict

BLOCK_SIZE = 64  # paged KV block (tokens)
PAGE_TABLE_ALIGN = 8  # page-table rows are a multiple of 32 bytes
PREFILL_CHUNK = 2048  # = Policy.moe_dispatch_max_rows; chunk starts are multiples of it (and of BLOCK_SIZE)
PREFILL_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)  # physical padded chunk lengths
# LM head decode DRAM-sharded config (cores, in0_block_w, readers per bank), doc/full_model/repro/head_sweep.py
# (logs/head_sweep_bfp8*.log): with the 32256-column shard 175.6 us / call; the unpadded 32000-column shard (125 tiles
# per DRAM bank) only admits 1 reader per bank: 393.6 us
HEAD_CFG = (64, 1, 3)
# per-chip LM-head columns are a multiple of TILE * dram banks * 3 readers (768): the global vocab is padded at its TAIL
# (128000 -> 4 x 32256 = 129024, zero weight rows) so chip d's column j is still token 32256 d + j and the sampler's
# invalid-vocab mask (padded_vocab_size) removes the tail
HEAD_COL_UNIT = 768
# generators built with build_generator(..., shared=True), keyed per mesh / configuration: lives here because the
# readiness runners load tt/generator.py as a standalone file module (a second copy of its globals)
SHARED_GENERATORS = {}
# bump when this file changes how the embedding / final norm / LM head weights are built (weight-cache key)
WEIGHT_LAYOUT_VERSION = 1


def prefill_bucket(n):
    """Smallest physical prefill bucket holding ``n`` rows (1 <= n <= PREFILL_CHUNK)."""
    assert 1 <= n <= PREFILL_CHUNK, n
    return next(b for b in PREFILL_BUCKETS if b >= n)


def blocks_for(n_tokens, block_size=BLOCK_SIZE):
    return math.ceil(n_tokens / block_size)


def _replicated(mesh, t, dtype, layout, mem=DRAM):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def host_tensor(mesh, t, dtype, layout):
    """Host-side mesh tensor (replicated) for ``copy_host_to_device_tensor``."""
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


class _ChunkRope:
    """The shared ``FusedRope`` with prefill cos/sin replaced by the chunk's runtime-gathered tables (one gather per
    chunk for all attention layers, no compile-time slice offsets)."""

    def __init__(self, rope, cos, sin):
        self._rope, self._cos, self._sin = rope, cos, sin

    def prefill_cos_sin(self, start, length):
        assert length == self._cos.shape[-2], (length, tuple(self._cos.shape))
        return self._cos, self._sin

    def __getattr__(self, name):
        return getattr(self._rope, name)


class DecodeState:
    """Persistent decode inputs (stable addresses for trace capture), replicated on every chip."""

    def __init__(self, mesh, batch, max_blocks, slot_mask_width, slot_mask_mem):
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        self.batch, self.max_blocks = batch, max_blocks
        self.tokens = _replicated(mesh, torch.zeros(1, 1, 1, batch, dtype=torch.int32), ttnn.uint32, rm)
        self.cur_pos = _replicated(mesh, torch.full((batch,), -1, dtype=torch.int32), ttnn.int32, rm)
        self.rope_pos = _replicated(mesh, torch.zeros(1, batch, dtype=torch.int32), ttnn.uint32, rm)
        self.page_table = _replicated(mesh, torch.zeros(batch, max_blocks, dtype=torch.int32), ttnn.int32, rm)
        self.slot_active = _replicated(mesh, torch.zeros(1, 1, batch, 1), ttnn.bfloat16, tile)
        mask = _replicated(mesh, torch.zeros(1, 1, batch, slot_mask_width), ttnn.bfloat16, tile)
        self.slot_mask = mask if slot_mask_mem is None else ttnn.to_memory_config(mask, slot_mask_mem)
        if slot_mask_mem is not None:
            ttnn.deallocate(mask)


class Lfm2Model:
    """Embedding + 40 x ``MultichipDecoder`` + final norm + vocab-parallel LM head (see module docstring)."""

    def __init__(
        self,
        mesh_device,
        *,
        model_path=None,
        hf_config=None,
        layer_indices=None,
        policy=MC_POLICY,
        plan=DEFAULT_MESH_PLAN,
        fusions=DEFAULT_FUSIONS,
        max_batch_size=TILE,
        max_seq_len=None,
        head_dtype="bfp8",
        head_cfg=HEAD_CFG,
        head_fidelity="hifi2",
        weight_cache=None,
    ):
        """``weight_cache``: directory of the device-weight cache (``weights.WeightCache``); default
        ``$LFM2_WEIGHT_CACHE`` or ``~/.cache/tt_autoport_weights/lfm2_5_24b_a2b``; ``LFM2_WEIGHT_CACHE=0`` / ``False``
        disables it."""
        from .model_config import HF_REVISION, resolve_model_path

        assert tuple(mesh_device.shape) == (1, plan.tp), f"mesh {tuple(mesh_device.shape)} != (1, {plan.tp})"
        assert max_batch_size == TILE, "the decoder stack is built for 32 physical slots"
        t0 = time.perf_counter()
        self.mesh = mesh_device
        self.model_path = model_path or resolve_model_path()
        self.hf_config = hf_config or load_hf_config(self.model_path)
        self.args = args = Lfm2MoeArgs.from_hf_config(self.hf_config)
        self.policy, self.plan, self.fusions = policy, plan, frozenset(fusions)
        self.batch = max_batch_size
        self.context = args.max_position_embeddings
        self.max_seq_len = max_seq_len or self.context
        self.layer_indices = tuple(range(args.num_layers)) if layer_indices is None else tuple(layer_indices)
        self.vocab = args.vocab_size
        self.tp = plan.tp
        self.vocab_local = math.ceil(self.vocab / self.tp / HEAD_COL_UNIT) * HEAD_COL_UNIT
        self.padded_vocab = self.vocab_local * self.tp
        H = self.H = args.hidden_size
        cache = self._weight_cache(weight_cache, HF_REVISION, head_dtype, head_cfg, head_fidelity, max_batch_size)
        with cache if cache is not None else contextlib.nullcontext():
            self._build(cache, mesh_device, args, policy, plan, max_batch_size, head_dtype, head_cfg, head_fidelity, t0)
        self.weight_cache = cache.summary() if cache is not None else None
        logger.info(
            f"Lfm2Model: {len(self.layers)} layers {self.layer_indices if len(self.layers) < 40 else '0..39'}, "
            f"build {self.build_s['globals_head']:.1f} s globals/head + {self.build_s['layers']:.1f} s layers, "
            f"weight cache {self.weight_cache}"
        )

    def _weight_cache(self, root, revision, head_dtype, head_cfg, head_fidelity, batch):
        if root is None:
            root = os.environ.get(
                "LFM2_WEIGHT_CACHE", str(Path.home() / ".cache" / "tt_autoport_weights" / "lfm2_5_24b_a2b")
            )
        if root in (False, "0", ""):
            return None
        key = WeightCache.key_for(
            revision=revision,
            policy=repr(self.policy),
            plan=repr(self.plan),
            fusions=sorted(self.fusions),
            head=(head_dtype, tuple(head_cfg), head_fidelity),
            mesh=tuple(self.mesh.shape),
            batch=batch,
            max_seq_len=self.max_seq_len,
            model_weights=(WEIGHT_LAYOUT_VERSION, HEAD_COL_UNIT),
            checkpoint=WeightCache.checkpoint_fingerprint(self.model_path),
        )
        return WeightCache(root, key)

    def _build(self, cache, mesh_device, args, policy, plan, max_batch_size, head_dtype, head_cfg, head_fidelity, t0):
        scope = cache.scope if cache is not None else (lambda name: None)
        H = self.H
        scope("globals")
        g = load_global_state_dict(self.model_path)
        emb = g["embed_tokens.weight"]  # [V, H] bf16 (tied LM head)
        assert tuple(emb.shape) == (self.vocab, H)
        self.embed = _replicated(mesh_device, emb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.norm_w = _as_tt(
            g["embedding_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.ckc_norm = _hifi4(mesh_device)
        self.eps = args.norm_eps
        # LM head: chip d multiplies by E_pad[32256 d : 32256 (d + 1)].T (E_pad = E with zero rows up to padded_vocab)
        emb_pad = torch.cat([emb, torch.zeros(self.padded_vocab - self.vocab, H, dtype=emb.dtype)])
        nl = self.vocab_local
        shards = [emb_pad[d * nl : (d + 1) * nl].T.contiguous() for d in range(self.tp)]
        self.head = DramShardedLinear(mesh_device, shards[0], DTYPES[head_dtype], head_cfg, policy.prefill_mm_v2)
        self.head.weight = _shard_like(self.head.weight, [_ds_host(self.head, w) for w in shards], dim=-1)
        self.head_ckc = _ckc(mesh_device, head_fidelity)
        self.head_dtype, self.head_cfg = head_dtype, head_cfg
        del g, emb, emb_pad, shards
        t_glob = time.perf_counter()

        scope("rope")
        self.rope = make_rope(mesh_device, args, self.max_seq_len, self.fusions)
        self.layers = []
        for i in self.layer_indices:
            scope(f"L{i:02d}")
            sd = load_layer_state_dict(self.model_path, i)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    sd,
                    hf_config=self.hf_config,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    max_batch_size=max_batch_size,
                    policy=policy,
                    fusions=self.fusions,
                    plan=plan,
                )
            )
            del sd
        if cache is not None:
            cache.stop()  # everything below (final-norm config, decode state) is not a weight: never cached
        self.resid_mem = self.layers[0].resid_mem
        assert all(l.resid_mem == self.resid_mem for l in self.layers)
        conv = [l for l in self.layers if not l.is_attention]
        self.attn_layers = [l for l in self.layers if l.is_attention]
        self.conv_layers = conv
        mask_mem = conv[0].slot_mask_mem if conv else None
        mask_width = conv[0].slot_mask_width if conv else 1
        assert all(l.slot_mask_mem == mask_mem and l.slot_mask_width == mask_width for l in conv)
        self.final_norm = ShardedNorm(mesh_device, H, plan.resid_cores, self.eps, self.ckc_norm)
        assert self.final_norm.mem == self.resid_mem
        self.bucketed_rope = any(l.bucketed_rope for l in self.attn_layers)
        self.max_blocks = math.ceil(blocks_for(self.max_seq_len) / PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN
        self.state = DecodeState(mesh_device, self.batch, self.max_blocks, mask_width, mask_mem)
        self.kv = None
        self.hoist_bucket_slices = os.environ.get("LFM2_HOIST_SLICES", "1") == "1"  # stage 7 D1 (A/B switch)
        self.build_s = {"globals_head": t_glob - t0, "layers": time.perf_counter() - t_glob}

    # ------------------------------------------------------------------------------------------
    # state: KV cache, ShortConv state
    # ------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, num_blocks, block_size=BLOCK_SIZE):
        """One ``[K, V]`` pair per attention layer (per chip ``[num_blocks, 2, block_size, 64]`` in the policy cache
        dtype, BFP8).  Block 0 is the scratch block (padded page-table entries and padded prefill rows)."""
        for l in self.attn_layers:
            l.mixer.allocate_kv_cache(num_blocks, block_size)
        self.kv = [[l.mixer.k_cache, l.mixer.v_cache] for l in self.attn_layers]
        self.num_blocks, self.block_size = num_blocks, block_size
        return self.kv

    def set_kv_cache(self, kv):
        """Bind an externally owned cache (same layout as ``allocate_kv_cache``)."""
        assert len(kv) == len(self.attn_layers)
        for l, (k, v) in zip(self.attn_layers, kv):
            l.mixer.set_kv_cache(k, v)
        self.kv = kv
        self.num_blocks, self.block_size = kv[0][0].shape[0], kv[0][0].shape[2]

    def zero_kv_cache(self):
        for k, v in self.kv:
            ttnn.fill(k, 0.0, output_tensor=k)
            ttnn.fill(v, 0.0, output_tensor=v)

    def conv_states(self):
        return [s for l in self.conv_layers for s in l.mixer.state]

    def zero_conv_state(self):
        for s in self.conv_states():
            ttnn.fill(s, 0.0, output_tensor=s)

    def _slot_tensors(self, slot):
        """(one-hot column [1,1,32,1], keep column [1,1,32,1], row-select [1,1,32,32] all rows -> slot), on device."""
        col = torch.zeros(1, 1, self.batch, 1)
        col[0, 0, slot, 0] = 1.0
        sel = torch.zeros(1, 1, self.batch, self.batch)
        sel[0, 0, :, slot] = 1.0
        return [_replicated(self.mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT) for t in (col, 1.0 - col, sel)]

    def reset_conv_slots(self, slots):
        keep = torch.ones(1, 1, self.batch, 1)
        keep[0, 0, list(slots), 0] = 0.0
        keep_tt = _replicated(self.mesh, keep, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        for s in self.conv_states():
            ttnn.multiply(s, keep_tt, output_tensor=s)
        ttnn.deallocate(keep_tt)

    def snapshot_conv_state(self, slot):
        """Prefix-caching hook: the ShortConv state rows of ``slot`` (every row of each returned tensor = that slot's
        row), one tensor per state tensor; exact (one-hot fp32-accumulated matmul)."""
        _, _, sel = self._slot_tensors(slot)
        snap = [ttnn.matmul(sel, s, compute_kernel_config=self.ckc_norm) for s in self.conv_states()]
        ttnn.deallocate(sel)
        return snap

    def restore_conv_state(self, slot, snap):
        """Write a ``snapshot_conv_state`` result into row ``slot`` (other rows unchanged)."""
        col, keep, sel = self._slot_tensors(slot)
        for s, r in zip(self.conv_states(), snap):
            t = ttnn.multiply(r, col)
            new = ttnn.mac(s, keep, t)
            ttnn.copy(new, s)
            ttnn.deallocate(t)
            ttnn.deallocate(new)
        for t in (col, keep, sel):
            ttnn.deallocate(t)

    # ------------------------------------------------------------------------------------------
    # decode (device only: warm-up and trace capture)
    # ------------------------------------------------------------------------------------------
    def decode_forward_device(self, bucket=None, advance=True):
        """Token ``state.tokens`` -> sampler-ready vocab-sharded logits ``[1, 1, 32, 32256]`` bf16 DRAM per chip; then
        ``cur_pos`` (idle -1 kept) and ``rope_pos`` advance on device.  ``bucket``: static decode bucket (slots < b)."""
        st = self.state
        tok = ttnn.reshape(st.tokens, (1, self.batch))
        e = ttnn.embedding(tok, self.embed, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, 32, H]
        x = ttnn.to_memory_config(ttnn.reshape(e, (1, 1, self.batch, self.H)), self.resid_mem)
        ttnn.deallocate(e)
        cos = sin = None
        if self.attn_layers:
            b = bucket if (bucket is not None and self.bucketed_rope) else None
            cos, sin = self.rope.decode_cos_sin(st.rope_pos, batch=b)
        # bucketed decode: the b-row page table / positions are sliced once per step for the 10 attention layers (the
        # attention decode consumes b-row tensors directly: OptimizedAttention._decode_r3 `sliced` / b-user update path)
        pt, cp = st.page_table, st.cur_pos
        hoist = self.hoist_bucket_slices and bucket is not None and bucket < self.batch and self.attn_layers
        if hoist:
            pt = ttnn.slice(st.page_table, (0, 0), (bucket, st.page_table.shape[-1]))
            cp = ttnn.slice(st.cur_pos, (0,), (bucket,))
        for l in self.layers:
            kw = dict(slot_active=st.slot_active, slot_mask=st.slot_mask, active_slots=bucket)
            if l.is_attention:
                kw.update(rope=self.rope, rope_cos=cos, rope_sin=sin, cur_pos=cp, page_table=pt)
            y = l.decode_forward(x, **kw)
            ttnn.deallocate(x)
            x = y
        if cos is not None:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        if hoist:
            ttnn.deallocate(pt)
            ttnn.deallocate(cp)
        logits = self._head_decode(x)
        ttnn.deallocate(x)
        if advance:
            ttnn.plus_one(st.cur_pos, skip_negative_entries=True)
            # idle rows' rope_pos also advance; bounded because every request boundary rewrites rope_pos (idle rows 0)
            # and a step budget never exceeds the active rows' remaining context (generator: pos_host capacity check)
            ttnn.plus_one(st.rope_pos)
        return logits

    def _head_decode(self, x_resid):
        """[1, 1, 32, H] in ``resid_mem`` -> final norm -> vocab-sharded logits (DRAM interleaved bf16)."""
        xn = self.final_norm(x_resid, self.norm_w)
        logits = self.head.decode(xn, self.head_ckc, DRAM)
        ttnn.deallocate(xn)
        return logits

    # ------------------------------------------------------------------------------------------
    # prefill (eager; one chunk)
    # ------------------------------------------------------------------------------------------
    def prefill_host_inputs(self, tokens, *, chunk_start, valid_len, slot, page_table_row, out_row=None):
        """Host values of one chunk's runtime inputs (bucket ``S = prefill_bucket(valid_len)``): token ids padded with
        ``pad_token_id``, positions, the sequence's page-table row, the chunk's block row, chunk start, ShortConv carry /
        tail / keep selections and the last-row selector (row ``valid_len - 1`` -> output row ``out_row``, default
        ``slot``).  Every chunk-dependent quantity is data, so one program set serves all chunks of a bucket."""
        n = int(valid_len)
        S = prefill_bucket(n)
        assert chunk_start % PREFILL_CHUNK == 0 and chunk_start + n <= self.max_seq_len, (chunk_start, n)
        ids = torch.full((1, S), self.args.pad_token_id, dtype=torch.int32)
        ids[0, :n] = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)[:n]
        pos = torch.arange(chunk_start, chunk_start + S, dtype=torch.int32).clamp(max=self.rope.max_seq_len - 1)
        h = {"ids": ids, "pos": pos.reshape(1, S)}
        if self.attn_layers:
            row = torch.zeros(1, self.max_blocks, dtype=torch.int32)
            src_row = torch.as_tensor(page_table_row, dtype=torch.int32).reshape(-1)[: self.max_blocks]
            row[0, : src_row.numel()] = src_row
            first, nblk = chunk_start // self.block_size, blocks_for(S, self.block_size)
            chunk_row = torch.zeros(1, math.ceil(nblk / PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN, dtype=torch.int32)
            src = row[0, first : first + nblk]
            chunk_row[0, : src.numel()] = src
            h.update(pt=row, cpt=chunk_row, cst=torch.tensor([chunk_start], dtype=torch.int32))
        if self.conv_layers:
            c0, c1, t0, t1, keep = ConvPrefillInputs.host_tensors(
                slot=slot, valid_len=n, padded_len=S, continuation=chunk_start > 0, windowed=False
            )
            h.update(carry0=c0, carry1=c1, tail0=t0, tail1=t1, keep=keep)
        sel = torch.zeros(1, 1, TILE, S)
        sel[0, 0, slot if out_row is None else out_row, n - 1] = 1.0
        h["sel"] = sel
        return S, h

    _PF_DTYPES = {
        "ids": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "pos": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "pt": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "cpt": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        "cst": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    }

    def prefill_inputs(self, S):
        """Persistent device inputs of prefill bucket ``S`` (stable addresses for the prefill traces; allocated by
        ``alloc_prefill_inputs`` before any capture)."""
        return self._pf_inputs[S]

    def alloc_prefill_inputs(self):
        """Allocate the persistent per-bucket prefill inputs (< 2 MB per chip for all buckets).  Call before the first
        trace capture: an allocation made later and kept across replays would be trace-unsafe."""
        if getattr(self, "_pf_inputs", None):
            return
        self._pf_inputs = {}
        for S in PREFILL_BUCKETS:
            _, h = self.prefill_host_inputs(
                torch.zeros(S, dtype=torch.int32), chunk_start=0, valid_len=S, slot=0, page_table_row=[0]
            )
            self._pf_inputs[S] = {
                k: _replicated(self.mesh, v, *self._PF_DTYPES.get(k, (ttnn.bfloat16, ttnn.TILE_LAYOUT)))
                for k, v in h.items()
            }

    def load_prefill_inputs(self, S, host):
        """Host refresh of bucket ``S``'s persistent inputs (ordered after earlier replays on the command queue)."""
        dev = self._pf_inputs[S]
        for k, v in host.items():
            dt, lay = self._PF_DTYPES.get(k, (ttnn.bfloat16, ttnn.TILE_LAYOUT))
            ttnn.copy_host_to_device_tensor(host_tensor(self.mesh, v, dt, lay), dev[k])
        return len(host)

    def prefill_chunk_device(self, S, *, continuation, out="last", accumulate_into=None):
        """Device-only prefill of one chunk of bucket ``S`` over the persistent inputs (eager or inside a trace).

        ``continuation`` selects the program family (first chunk / chunk at a 2048-aligned start > 0); the actual start,
        positions, blocks, slot and valid length are the runtime inputs.  ``out``: "none" (no head), "last" (last valid
        row's logits in row ``out_row`` of ``[1, 1, 32, 32256]``, other rows exactly 0; with ``accumulate_into`` they are
        added into that persistent tensor and nothing is returned) or "all" (``[1, 1, S, 32256]``, eager readiness path).
        """
        inp = self._pf_inputs[S]
        # the representative start (0 / PREFILL_CHUNK) selects the same SDPA programs as any PREFILL_CHUNK-aligned start
        # only while the decoder's q / k chunk caps are powers of two dividing PREFILL_CHUNK (prefill_sdpa_chunk and the
        # k_chunk loop depend on the start only through divisibility); the runtime start is inp["cst"]
        for l in self.attn_layers:
            q_cap, k_cap = l.mixer.sdpa_chunk, l.policy.sdpa_prefill_k_chunk or l.mixer.sdpa_chunk
            assert all(c & (c - 1) == 0 and PREFILL_CHUNK % c == 0 for c in (q_cap, k_cap)), (q_cap, k_cap)
        e = ttnn.embedding(inp["ids"], self.embed, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, S, H]
        x = ttnn.reshape(e, (1, 1, S, self.H))
        attn_kw, cs = {}, []
        if self.attn_layers:
            for table in (self.rope.cos_rm, self.rope.sin_rm):
                t = ttnn.embedding(inp["pos"], table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, S, hd]
                cs.append(ttnn.reshape(t, (1, 1, S, t.shape[-1])))
            # chunk_start only selects the SDPA chunk program (first vs 2048-aligned continuation): runtime start in cst
            attn_kw = dict(
                rope=_ChunkRope(self.rope, *cs),
                chunk_start=PREFILL_CHUNK if continuation else 0,
                page_table=inp["pt"],
                chunk_page_table=inp["cpt"],
                chunk_start_tensor=inp["cst"],
            )
        conv_in = None
        if self.conv_layers:
            conv_in = ConvPrefillInputs(inp["carry0"], inp["carry1"], inp["tail0"], inp["tail1"], inp["keep"])
        for l in self.layers:
            y = l.prefill_forward(x, **attn_kw) if l.is_attention else l.prefill_forward(x, conv_inputs=conv_in)
            ttnn.deallocate(x)
            x = y
        for t in cs:
            ttnn.deallocate(t)
        logits = None
        if out == "all":
            xn = ttnn.rms_norm(
                x, epsilon=self.eps, weight=self.norm_w, memory_config=DRAM, compute_kernel_config=self.ckc_norm
            )
            logits = self.head.prefill(xn, self.head_ckc, DRAM)
            ttnn.deallocate(xn)
        elif out == "last":
            h = ttnn.matmul(inp["sel"], x, compute_kernel_config=self.ckc_norm, memory_config=ttnn.L1_MEMORY_CONFIG)
            logits = self._head_decode(h)
            ttnn.deallocate(h)
            if accumulate_into is not None:
                ttnn.add(accumulate_into, logits, output_tensor=accumulate_into)  # rows of other slots are exactly 0
                ttnn.deallocate(logits)
                logits = None
        ttnn.deallocate(x)
        return logits

    def prefill_chunk(self, tokens, *, chunk_start, valid_len, slot, page_table_row, out="last", out_row=None):
        """One eager prefill chunk for one sequence (see ``prefill_host_inputs`` / ``prefill_chunk_device``): chunks of
        at most PREFILL_CHUNK rows at PREFILL_CHUNK-aligned starts, padded to ``prefill_bucket(n)`` with
        ``pad_token_id``; padded rows write K/V only into this sequence's own blocks past its logical end or the scratch
        block, never feed valid rows, and are dropped.  Returns the device logits (``out`` "last" / "all") or None."""
        self.alloc_prefill_inputs()
        S, host = self.prefill_host_inputs(
            tokens,
            chunk_start=chunk_start,
            valid_len=valid_len,
            slot=slot,
            page_table_row=page_table_row,
            out_row=out_row,
        )
        self.load_prefill_inputs(S, host)
        return self.prefill_chunk_device(S, continuation=chunk_start > 0, out=out)

    def logits_to_host(self, logits, rows=None):
        """Vocab-sharded device logits -> torch fp32 ``[..., rows, 128000]`` (full-logits readback; padded tail dropped)."""
        t = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))[..., : self.vocab].float()
        return t if rows is None else t[..., :rows, :]
