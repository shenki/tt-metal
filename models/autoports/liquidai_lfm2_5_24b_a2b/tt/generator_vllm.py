# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for LFM2.5-24B-A2B (tenstorrent/vllm-tt-plugin model interface, decode reload contract v1).

Thin translation layer over ``tt/generator.py`` (``Lfm2Generator``): the model, the selected precision policy
(``tt/precision.py`` ``resolve()``), the traced chunked prefill, the split-sampling decode traces and the token feedback
all live there.  This module only maps plugin keyword arguments onto the generator's low-level calls:

* cache ownership: vLLM sizes the attention KV pool (``get_max_tokens_all_users`` = max_num_seqs x max_model_len) and
  the adapter allocates it once in ``allocate_kv_cache`` and binds it (``Lfm2Generator.bind_cache``); the page tables
  come from vLLM on every reload.  Block 0 is vLLM's null block and the model's scratch block.
* slots: plugin state slot ``i`` = generator slot ``i`` (ShortConv state row, sampling row, decode row; decode rows are
  front-packed after ``slot_remap``).
* ShortConv state across prefill calls: the plugin may move a request to another slot between prefill chunks and a
  prefix-cache hit starts a request mid-prompt, so conv state is carried by content, not by slot: after every chunk that
  ends on the 2048-token grid the slot's state is snapshotted into a preallocated pool keyed by a hash of the token
  history (``_ConvSnapshots``).  A chunk starting at ``P > 0`` uses the highest grid boundary ``Q <= P`` whose state is
  either already in its slot or in the pool, and recomputes ``[Q, P)`` (worst case ``Q = 0``).  The recompute writes
  its K/V into adapter-owned scratch blocks past the vLLM pool (a full page-table row: any walk-down fits) (the chunk's page-table entries for ``[Q, P)`` point
  there), never into the cached blocks vLLM handed the request: those may be shared with a live request, and a
  recompute through a different prefill bucket is not bitwise identical.  Prefill chunk programs only exist for
  2048-aligned starts.
* sampling: device sampling uses the generator's traced sampler (k <= 32, top-p, temperature); per-row seeds are
  position-derived (``base(seed) + position``), so a seeded stream is independent of slot, batch and async timing.
  Host sampling steps (selected by the plugin for logprobs, penalties, min_p, structured output, steps with
  intermediate prefill chunks) get last-position logits: prefill from the traced chunks' persistent logits buffer,
  decode from the model trace alone (sampler not replayed).  That compat path is never used for greedy / top-k
  benchmarks.
"""

import hashlib
import json
import os
import random
import time
from collections import OrderedDict

import torch
from loguru import logger

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import TILE
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import Lfm2Generator
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model import BLOCK_SIZE, PREFILL_BUCKETS, PREFILL_CHUNK, Lfm2Model
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DECODE_BUCKETS

DEFAULT_SNAPSHOTS = 128  # ShortConv snapshot pool entries (1.9 MB per entry per chip)


def _fabric_capability():
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192  # multichip_decoder.mesh_device_params
    return {"config": ttnn.FabricConfig.FABRIC_1D_RING, "router_config": router}


def _seed_base(seed):
    """31-bit stream base of a request seed (a fresh random base for unseeded requests)."""
    if seed is None:
        return random.getrandbits(31)
    return (int(seed) * 2654435761 + 0x5BD1E995) & 0x7FFFFFFF


class _ConvSnapshots:
    """Host bookkeeping of the ShortConv snapshot pool: LRU map ``prefix hash -> pool entry`` and, per slot, the
    ``(end, hash)`` of the token history its live state belongs to (None once decode advanced it)."""

    def __init__(self, entries, batch):
        self.entries = entries
        self.lru = OrderedDict()  # hash -> entry
        self.free = list(range(entries))
        self.slot_key = [None] * batch
        self.stats = dict(
            stores=0, store_hits=0, restores=0, in_slot=0, recompute_tokens=0, evictions=0, scratch_rows=0
        )

    @staticmethod
    def chain(tokens, n_blocks):
        """Hashes of the token history ending at each 2048-token boundary (``out[k]`` covers ``tokens[:2048 (k + 1)]``)."""
        out, h = [], b""
        t = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)
        for k in range(n_blocks):
            h = hashlib.blake2b(
                h + t[k * PREFILL_CHUNK : (k + 1) * PREFILL_CHUNK].numpy().tobytes(), digest_size=16
            ).digest()
            out.append(h)
        return out

    def lookup(self, h):
        e = self.lru.get(h)
        if e is not None:
            self.lru.move_to_end(h)
        return e

    def reserve(self, h):
        """Entry to store ``h`` into (None if already stored)."""
        if h in self.lru:
            self.lru.move_to_end(h)
            self.stats["store_hits"] += 1
            return None
        if self.free:
            e = self.free.pop()
        else:
            _, e = self.lru.popitem(last=False)
            self.stats["evictions"] += 1
        self.lru[h] = e
        self.stats["stores"] += 1
        return e

    def permute(self, remap):
        self.slot_key = [self.slot_key[int(j)] for j in remap]


class _DecodeOut:
    """Device output of one decode submission (sampled tokens or vocab-sharded logits) and its valid row count."""

    def __init__(self, kind, tensor, rows):
        self.kind, self.tensor, self.rows = kind, tensor, rows


class TTLfm2MoeForCausalLM:
    """Registered as ``TTLfm2MoeForCausalLM`` (``TT_MODEL_CLASS_OVERRIDES``); HF architecture ``Lfm2MoeForCausalLM``."""

    decode_input_update_contract = 1
    model_capabilities = {
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "fabric_config": _fabric_capability(),
    }
    tt_supported_decode_batch_sizes = tuple(DECODE_BUCKETS)

    def __init__(self, generator: Lfm2Generator):
        self.gen = generator
        self.model = generator.model
        self.mesh = generator.mesh
        self.model_capabilities = dict(type(self).model_capabilities)
        self.B = generator.B
        n = int(os.environ.get("LFM2_CONV_SNAPSHOTS", DEFAULT_SNAPSHOTS)) if self.model.conv_layers else 0
        generator.snapshot_entries = n
        self.snaps = _ConvSnapshots(n, self.B)
        self.seed_base = [None] * self.B  # per slot: (request seed, stream base)
        self.events_path = os.environ.get("LFM2_VLLM_EVENTS")
        self.steps = 0

    # ------------------------------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError(
                f"LFM2.5-24B-A2B serves on a 1x4 mesh (--mesh-device P300x2), got {tuple(mesh_device.shape)}"
            )
        if tt_data_parallel != 1:
            raise ValueError("tt_data_parallel must be 1 (the model is tensor-parallel over the 4 chips)")
        if max_batch_size > TILE:
            raise ValueError(f"max_num_seqs {max_batch_size} > {TILE} decode slots")
        layers = os.environ.get("LFM2_VLLM_LAYERS")  # reduced serving target for adapter bring-up only
        layer_indices = [int(x) for x in layers.split(",")] if layers else None
        context = int(hf_config.max_position_embeddings)
        if max_seq_len is not None and int(max_seq_len) > context:
            raise ValueError(f"max_model_len {max_seq_len} exceeds the trained context {context}")
        # the model is always built for the full trained context (RoPE tables, page-table width, weight-cache key); vLLM
        # enforces the served max_model_len
        model = Lfm2Model(mesh_device, layer_indices=layer_indices)
        from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

        logger.info(
            f"TTLfm2MoeForCausalLM: precision {model.precision.config_id} ({model.precision_source}), "
            f"layers {len(model.layers)}, max_model_len {max_seq_len}, optimizations {optimizations!r} (unused), "
            f"trace allocation tracking {'ON' if TRACE_ALLOC_TRACKING else 'off'}"
        )
        return cls(Lfm2Generator(model))

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=4, tt_data_parallel=1, max_model_len=None, max_num_seqs=TILE, **kwargs
    ):
        """KV capacity: every slot can hold a full-length request (32 x 131072 BFP8 fits: doc/context_contract.json)."""
        return int(max_num_seqs) * int(max_model_len or 131072)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """vLLM-owned attention KV pool: per chip ``[num_blocks, 2, 64, 64]`` per attention layer in the policy KV dtype
        (BFP8; vLLM's ``dtype`` describes its byte accounting only)."""
        num_blocks, heads, block, head_dim = (int(x) for x in kv_cache_shape)
        if (heads, block, head_dim) != (2, BLOCK_SIZE, 64):
            raise ValueError(
                f"unexpected KV block shape {tuple(kv_cache_shape)} (want [*, 2, {BLOCK_SIZE}, 64] per chip)"
            )
        if num_layers != 10:
            raise ValueError(f"vLLM requested {num_layers} attention layers, the checkpoint has 10")
        # vLLM addresses blocks [0, num_blocks); the blocks after them belong to the adapter: the recompute target of a
        # prefix hit's [Q, P).  The walk-down is bounded only by the snapshot pool (an evicted boundary sends Q further
        # back, up to 0), and later recompute chunks attend to earlier ones, so scratch covers a full page-table row.
        n_scratch = self.model.max_blocks
        kv = self.model.allocate_kv_cache(num_blocks + n_scratch, block)  # only the built attention layers
        self.gen.bind_cache(kv)
        self.kv_handle = kv
        self.scratch_blocks = torch.arange(num_blocks, num_blocks + n_scratch, dtype=torch.int32)
        logger.info(
            f"TTLfm2MoeForCausalLM: KV pool {num_blocks} blocks x {block} tokens (+{n_scratch} adapter scratch), "
            f"{len(kv)} attention layers"
        )
        return kv

    def _check_cache(self, kv_cache):
        if kv_cache is not None and kv_cache is not self.kv_handle:
            raise ValueError("kv_cache is not the pool returned by allocate_kv_cache")

    # ------------------------------------------------------------------------------------------
    # warm-up: phase 1 (compile everything, no capture) then phase 2 (capture every reachable trace once)
    # ------------------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, can_sample_on_device=True, **kwargs):
        self._check_cache(kv_cache)
        self.gen.prepare()
        if enable_trace:
            t0 = time.perf_counter()
            # reachable prefill keys: final chunk of any bucket (first / continuation), full non-final chunks
            keys = [(S, cont, "last") for S in PREFILL_BUCKETS for cont in (False, True)]
            keys += [(PREFILL_CHUNK, cont, "none") for cont in (False, True)]
            for key in keys:
                if key not in self.gen.prefill_traces:
                    self.gen._capture_prefill(key)
            self._event("warmup_prefill", capture_s=time.perf_counter() - t0, keys=[str(k) for k in keys])

    def warmup_model_decode(
        self,
        kv_cache=None,
        max_batch_size=TILE,
        num_blocks=None,
        can_sample_on_device=True,
        enable_trace=True,
        **kwargs,
    ):
        self._check_cache(kv_cache)
        self.gen.prepare()
        if enable_trace:
            t0 = time.perf_counter()
            for b in DECODE_BUCKETS:
                self.gen._trace(None if b >= TILE else b)
            self._event("warmup_decode", capture_s=time.perf_counter() - t0)
            self._event("traces_ready", **self.trace_inventory())

    def trace_inventory(self):
        g = self.gen
        return dict(
            decode={str(b): [int(t[0]), int(t[2]), int(g.sample_hist_traces[b])] for b, t in g.traces.items()},
            prefill={str(k): int(v) for k, v in g.prefill_traces.items()},
            prefill_sampler=None if g.pf_sample_trace is None else int(g.pf_sample_trace),
            counters={k: int(v) for k, v in g.counters.items() if "capture" in k},
            trace_bytes_per_bank=int(g._trace_bytes()),
        )

    # ------------------------------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------------------------------
    def _set_row_sampling(self, rows_slots, sampling_params, positions):
        """Per-slot k / p / temperature and position-derived seeds of ``rows_slots`` [(row, slot)]."""
        sp = sampling_params
        seeds_rows, seeds_vals = [], []
        for row, slot in rows_slots:
            t = float(sp.temperature[row])
            k = int(sp.top_k[row])
            p = float(sp.top_p[row])
            if t == 0.0 or k == 1:
                self.gen.set_sampling_params(slots=[slot], top_k=1)
            else:
                if not 1 <= k <= 32:
                    raise ValueError(f"device sampling row {row}: top_k {k} outside 1..32 (max_device_top_k)")
                self.gen.set_sampling_params(slots=[slot], top_k=k, top_p=p, temperature=t)
            seed = None if sp.seed is None else sp.seed[row]
            prev = self.seed_base[slot]
            if prev is None or prev[0] != seed or seed is None:
                self.seed_base[slot] = (seed, _seed_base(seed))
            seeds_rows.append(slot)
            seeds_vals.append(self.seed_base[slot][1] + int(positions[row]))
        if seeds_rows:
            self.gen.set_row_seeds(seeds_rows, seeds_vals)

    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache=None,
        enable_trace=True,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """Rows ``i``: compute ``tokens[i, start_pos[i]:prompt_lens[i]]`` into slot ``empty_slots[i]`` (default ``i``).
        Returns sampled ids ``[B, 1]`` (device sampling) or last-position logits ``[B, 1, 128000]`` fp32 (host
        sampling; intermediate chunks' rows are produced too and dropped by the plugin)."""
        self._check_cache(kv_cache)
        for k in ("page_tables_per_layer", "stage_inputs", "pixel_values"):
            if kwargs.get(k) is not None:
                raise ValueError(f"unsupported prefill input {k}")
        tokens = torch.as_tensor(tokens)
        B = tokens.shape[0]
        starts = [int(x) for x in torch.as_tensor(start_pos).reshape(-1)]
        ends = [int(x) for x in torch.as_tensor(prompt_lens).reshape(-1)]
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        page_table = torch.as_tensor(page_table, dtype=torch.int32).clone()
        C = PREFILL_CHUNK
        eff, chains, rows_info = [], [], []
        for i in range(B):
            P, E, s = starts[i], ends[i], slots[i]
            if not 0 <= P < E <= self.model.max_seq_len:
                raise ValueError(f"prefill row {i}: start {P} end {E}")
            ch = self.snaps.chain(tokens[i, :E], E // C) if self.model.conv_layers else []
            chains.append(ch)
            Q = 0
            for q in range((P // C) * C, 0, -C):
                if not self.model.conv_layers:
                    Q = q  # attention-only target: no recurrent state to carry
                    break
                h = ch[q // C - 1]
                if self.snaps.slot_key[s] == (q, h):
                    Q = q
                    self.snaps.stats["in_slot"] += 1
                    break
                e = self.snaps.lookup(h)
                if e is not None:
                    self.model.restore_conv_state_from(s, e)
                    self.snaps.stats["restores"] += 1
                    Q = q
                    break
            self.snaps.stats["recompute_tokens"] += P - Q
            if Q < P:
                # [Q, P) already has cached K/V in the request's blocks (possibly shared): recompute into scratch
                if P % BLOCK_SIZE or Q % BLOCK_SIZE:
                    raise ValueError(f"prefill row {i}: recompute range [{Q}, {P}) is not block aligned")
                n_blk = (P - Q) // BLOCK_SIZE
                if n_blk > len(self.scratch_blocks):
                    raise ValueError(
                        f"prefill row {i}: recompute [{Q}, {P}) needs {n_blk} scratch blocks > {len(self.scratch_blocks)}"
                    )
                page_table[i, Q // BLOCK_SIZE : P // BLOCK_SIZE] = self.scratch_blocks[:n_blk]
                self.snaps.stats["scratch_rows"] += 1
            eff.append(Q)
            rows_info.append([P, E, s, Q])

        def on_chunk(b, end):
            if end % C == 0 and self.model.conv_layers:
                h = chains[b][end // C - 1]
                e = self.snaps.reserve(h)
                if e is not None:
                    self.model.snapshot_conv_state_into(slots[b], e)
                self.snaps.slot_key[slots[b]] = (end, h)
            else:
                self.snaps.slot_key[slots[b]] = None

        L = max(E - Q for (_, E, _, Q) in rows_info)
        chunk_tokens = torch.zeros(B, L, dtype=torch.int64)
        for i, (_, E, _, Q) in enumerate(rows_info):
            chunk_tokens[i, : E - Q] = tokens[i, Q:E]
        device = sampling_params is not None
        if device:
            self._set_row_sampling(list(enumerate(slots)), sampling_params, ends)
        out = self.gen.prefill_forward(
            chunk_tokens,
            page_table=page_table,
            prompt_lens=[E - Q for (_, E, _, Q) in rows_info],
            slots=slots,
            start_pos=eff,
            sample=device,
            host_logits=not device,
            on_chunk=on_chunk,
        )
        self._event("prefill", rows=rows_info, device_sampling=device)
        if device:
            return torch.as_tensor(out, dtype=torch.int64).reshape(B, 1)
        return out

    # ------------------------------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=False,
        slot_remap=None,
        sampling_params=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """One decode step of ``B = tokens.shape[0]`` front-packed rows (row = slot; ``start_pos`` -1 = padding).

        Reload commands (contract v1) are executed exactly: ``slot_remap`` first (ShortConv rows, sampling rows,
        RNG seeds unless reset on this step, and the resident token / position / page-table rows unless reloaded),
        then the host inputs (``reload_inputs``: tokens, positions, page tables; ``reload_page_table``: page tables
        only; neither: device-resident token feedback and positions), then sampling params / RNG state."""
        self._check_cache(kv_cache)
        if "reset_batch" in kwargs:
            raise ValueError(
                "legacy reset_batch is not accepted: this adapter implements decode_input_update_contract=1"
            )
        for k in ("spec_mode", "defer_device_sampling", "page_tables_per_layer", "prompt_tokens", "output_tokens"):
            if kwargs.get(k) is not None and kwargs.get(k) is not False:
                raise ValueError(f"unsupported decode input {k}")
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already includes the page table")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state requires reload_inputs")
        if not enable_trace:
            raise ValueError("serving decode requires decode tracing (tt.trace_mode all / decode_only)")
        tokens = torch.as_tensor(tokens)
        B = int(tokens.shape[0])
        device = sampling_params is not None
        if slot_remap is not None:
            self.gen.permute_slots(slot_remap, resident_inputs=not reload_inputs, seeds=not reset_sampling_state)
            self.snaps.permute(torch.as_tensor(slot_remap).tolist())
            self.seed_base = [self.seed_base[int(j)] for j in torch.as_tensor(slot_remap).tolist()]
        pos = None
        if reload_inputs:
            pos = torch.as_tensor(start_pos, dtype=torch.int64).reshape(-1)
            tokens = torch.where(pos.reshape(tokens.shape[0], *([1] * (tokens.dim() - 1))) >= 0, tokens, 0)
        if device and (reload_sampling_params or reset_sampling_state):
            # positions of the sampled tokens: consumed position + 1 (host-authoritative on reload, else the generator's
            # mirror, which advances exactly once per submitted decode)
            cur = pos if pos is not None else self.gen.pos_host[:B]
            rows = [r for r in range(B) if int(cur[r]) >= 0]
            sp = sampling_params
            if reset_sampling_state:
                for r in rows:
                    seed = None if sp.seed is None else sp.seed[r]
                    self.seed_base[r] = (seed, _seed_base(seed))
            self._set_row_sampling([(r, r) for r in rows], sp, [int(c) + 1 for c in cur.tolist()])
        out = self.gen.decode_forward(
            tokens.reshape(-1) if reload_inputs else None,
            pos if reload_inputs else None,
            page_table=page_table if (reload_inputs or reload_page_table) else None,
            read_from_device=False,
            active_slots=B,
            return_logits=not device,
        )
        for r in range(B):
            self.snaps.slot_key[r] = None  # decode advanced (or idled) the rows' ShortConv state
        self.steps += 1
        self._event(
            "decode",
            rows=B,
            device_sampling=device,
            reload_inputs=bool(reload_inputs),
            reload_page_table=bool(reload_page_table),
            reload_sampling_params=bool(reload_sampling_params),
            reset_sampling_state=bool(reset_sampling_state),
            remap=slot_remap is not None,
        )
        res = _DecodeOut("tokens" if device else "logits", self.model.state.tokens if device else out, B)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(res), is_tokens=device)
        return res

    def read_decode_output(self, tt_out, async_read=False, sample_rows=None):
        """Start the (non-blocking) host copy of a decode submission; ``(handle, [event])`` when ``async_read``.  The
        copy is enqueued before the next submission, so the persistent token buffer is read for this step."""
        if sample_rows is not None:
            raise ValueError("selective host readback is not declared")
        host = tt_out.tensor.cpu(blocking=not async_read)
        handle = _DecodeOut(tt_out.kind, host, tt_out.rows)
        if async_read:
            return handle, [ttnn.record_event(self.mesh, 0)]
        return handle

    def process_decode_output_host(self, tt_out, is_tokens=False, sample_rows=None):
        if sample_rows is not None:
            raise ValueError("compact host logits are not declared")
        if (tt_out.kind == "tokens") != bool(is_tokens):
            raise ValueError(f"decode output holds {tt_out.kind}, plugin expects {'tokens' if is_tokens else 'logits'}")
        if is_tokens:
            t = ttnn.to_torch(tt_out.tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            return t[0].reshape(-1)[: tt_out.rows].to(torch.int64).reshape(-1, 1)
        t = ttnn.to_torch(tt_out.tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
        return t[0, 0, : tt_out.rows, : self.model.vocab].float().unsqueeze(1)

    # ------------------------------------------------------------------------------------------
    # request state / lifecycle
    # ------------------------------------------------------------------------------------------
    def release_request(self, slot):
        """A finished / preempted request's slot: idle it (no KV writes through its stale page-table row) and zero its
        ShortConv rows."""
        self.gen.reset_slots([int(slot)])
        self.snaps.slot_key[int(slot)] = None
        self.seed_base[int(slot)] = None
        self._event("release", slot=int(slot))

    def release_persistent_capture(self):
        self._event("release_traces", **self.trace_inventory())
        self.gen.release_traces()

    # ------------------------------------------------------------------------------------------
    # instrumentation (LFM2_VLLM_EVENTS=<jsonl path>): trace keys / captures / chunk starts for the serving checks
    # ------------------------------------------------------------------------------------------
    def _event(self, kind, **data):
        if not self.events_path:
            return
        c = self.gen.counters
        rec = dict(
            t=time.time(),
            kind=kind,
            step=self.steps,
            captures=int(c["trace_captures"]),
            prefill_captures=int(c["prefill_trace_captures"]),
            model_replays=int(c["model_replays"]),
            sampling_replays=int(c["sampling_replays"]),
            prefill_replays=int(c["prefill_replays"]),
            eager_chunks=int(c["prefill_eager_chunks"]),
            token_refreshes=int(c["token_refreshes"]),
            position_refreshes=int(c["position_refreshes"]),
            page_table_refreshes=int(c["page_table_refreshes"]),
            snapshots=dict(self.snaps.stats),
            **data,
        )
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
