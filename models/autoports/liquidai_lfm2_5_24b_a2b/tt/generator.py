# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generator for LFM2.5-24B-A2B on the 1x4 Blackhole ring: readiness ``Generator`` contract + serving-ready low-level API.

Token-out decode = canonical split sampling (two traces per decode bucket, captured once and kept for the generator's
lifetime):

* model trace (``Lfm2Model.decode_forward_device``): persistent token input -> 40 layers -> final norm -> vocab-sharded
  logits; ``cur_pos`` / ``rope_pos`` advance on device (``plus_one``);
* sampling trace: ``TTSampling(logits, tt_out_tok=model.state.tokens)`` (local top-32 per chip, all-gather of the
  candidates, ``ttnn.sampling`` with per-row k / p / temperature) writes the sampled token straight into the token input
  of the next model replay; ``plus_one(seeds)``.

Greedy = the same split path with k=1, p=0, temp=1 (semantically greedy; ``sampling_strategy="argmax"`` builds the
force-argmax variant for benchmarking only).  The steady-state loop is replay + replay (+ a tiny history-append trace when
the caller defers readback): no host token / position / RoPE / page-table / mask writes and no synchronization; host
refreshes happen only at request boundaries (new prompt, changed active-slot set, changed page table, teacher-forced
token) and are counted in ``self.counters``.

Prefill is eager and chunked (``Lfm2Model.prefill_chunk``: chunks of 2048 at 2048-aligned starts, the last padded to a
physical bucket); its first token is sampled by a third traced sampler over a persistent prefill-logits buffer.  All
prefill / decode / sampling programs are compiled by ``prepare()`` (phase 1) before the first capture (phase 2: lazily
per bucket on first use, never recaptured); warm-up mutations of tokens / positions / seeds / ShortConv state are undone.

Cache ownership: the standalone generator allocates the full cache (32 slots x 131072 tokens, BFP8) and a fixed
slot -> block mapping; serving callers bind their own cache / page table (``bind_cache``) and drive ``prefill_forward`` /
``decode_forward`` with explicit slots, positions and page tables.
"""

import contextlib
import os
import time
from collections import Counter
from types import SimpleNamespace

import torch
from loguru import logger
from readiness_check.contract import Generator

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import DRAM, TILE
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.model import (
    BLOCK_SIZE,
    PREFILL_BUCKETS,
    PREFILL_CHUNK,
    SHARED_GENERATORS,
    Lfm2Model,
    blocks_for,
    host_tensor,
)
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import (
    DECODE_BUCKETS,
    decode_bucket,
    validate_decode_bucket,
)

HISTORY_ROWS = 512  # device token-history rows (deferred readback window; read and rewound when full)


def configure_fabric(payload_bytes=8192):
    """Fabric for the 1x4 ring; call before opening the mesh (``multichip_decoder.mesh_device_params``)."""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload_bytes
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)


def open_mesh(trace_region_size=768 << 20):
    """Open the 1x4 mesh with the parameters the model needs (CCL semaphores in L1_SMALL, trace region)."""
    from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import mesh_device_params

    p = mesh_device_params(trace_region_size)
    ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
    return ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=p["l1_small_size"], trace_region_size=p["trace_region_size"]
    )


def _acknowledge(t):
    """Exempt one reviewed trace output from the allocation tracker (lifetime invariant in doc/full_model/README.md)."""
    try:
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(t)
    except ImportError:  # pragma: no cover
        pass


class Lfm2Generator(Generator):
    def __init__(
        self, model: Lfm2Model, *, host_sampling=False, sampling_strategy="split", pad_logits_to_power_of_2=True
    ):
        from transformers import AutoTokenizer

        from models.common.modules.tt_ccl import get_tt_ccl
        from models.common.sampling.tt_sampling import TTSampling

        self.model, self.mesh = model, model.mesh
        self.counters = Counter()
        self.last_perf = {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model.model_path))
        self.B = model.batch
        self.host_sampling = host_sampling
        if sampling_strategy not in ("split", "argmax"):
            raise ValueError(f"unknown sampling strategy {sampling_strategy!r}")
        self.sampling_strategy = sampling_strategy
        args = SimpleNamespace(
            vocab_size=model.vocab,
            padded_vocab_size=model.padded_vocab,
            cluster_shape=(1, model.tp),
            max_batch_size=self.B,
            max_top_k=32,
            pad_logits_to_power_of_2=pad_logits_to_power_of_2,
            model_config={
                "SAMPLING_AG_CONFIG": dict(
                    allow_force_argmax=sampling_strategy == "argmax", num_links=2, topology=ttnn.Topology.Ring
                )
            },
        )
        # greedy rows: k=1, p=0, temp=1 (split path) or the force-argmax params when that path is benchmarked
        self._greedy = (1, 1.0, 1.0) if sampling_strategy == "argmax" else (1, 0.0, 1.0)
        k, p, t = self._greedy
        self.k_host = torch.full((self.B,), k, dtype=torch.int64)
        self.p_host = torch.full((self.B,), p)
        self.t_host = torch.full((self.B,), t)
        self.sampler = TTSampling(
            self.mesh, get_tt_ccl(self.mesh), args, k=self.k_host, p=self.p_host, temp=self.t_host
        )
        # stable local top-k (TTSampling's own switch, `_topk_stable`; this [32, 32768] call is not on the routed
        # large-index path): 99.5 vs 124.2 us per call on this shard (doc/optimized_full_model/logs/topk_stable_bench.log);
        # the greedy tie-break pass is kept, so greedy exactness does not depend on it
        self.sampler._topk_stable = os.environ.get("LFM2_STABLE_TOPK", "1") == "1"
        self._push_sampling_params()
        rm = ttnn.ROW_MAJOR_LAYOUT
        mk = lambda t_, dt, lay, mapper: ttnn.from_torch(
            t_, dtype=dt, layout=lay, device=self.mesh, memory_config=DRAM, mesh_mapper=mapper
        )
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        # prefill first-token sampling: persistent logits (vocab-sharded) and token output
        self.pf_logits = mk(
            torch.zeros(1, 1, TILE, model.padded_vocab),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.ShardTensorToMesh(self.mesh, dim=-1),
        )
        self.pf_tokens = mk(torch.zeros(1, 1, 1, self.B, dtype=torch.int32), ttnn.uint32, rm, rep)
        # device token history (free-running deferred readback)
        self.history = mk(torch.zeros(HISTORY_ROWS, 1, 1, self.B, dtype=torch.int32), ttnn.uint32, rm, rep)
        self.history_cursor = mk(torch.zeros(1, dtype=torch.int32), ttnn.uint32, rm, rep)
        self.history_count = 0
        self._history_out = []  # host rows of earlier (full) windows
        self.seed0 = ttnn.to_torch(ttnn.get_device_tensors(self.sampler.seeds_tt_tensor)[0]).to(torch.int64)
        # cache / request state (host mirrors of the device inputs)
        self.kv = None
        self.owns_cache = False
        self.page_host = None
        self.pos_host = torch.full((self.B,), -1, dtype=torch.int64)  # next position per slot, -1 idle
        self.active_host = None  # active-slot tuple whose masks are on device
        # traces
        self.prepared = False
        self.traces = {}  # bucket -> (model trace, logits, sampling trace)
        self.pf_sample_trace = None
        self.append_trace = None
        self.prefill_traces = {}  # (bucket, continuation, out) -> trace id
        self.sample_hist_traces = {}  # bucket -> sampling + history-append trace id
        self.traced_prefill = os.environ.get("LFM2_EAGER_PREFILL") != "1"

    # ------------------------------------------------------------------------------------------
    # host -> device refresh helpers (request boundaries only)
    # ------------------------------------------------------------------------------------------
    def _copy(self, src, dst, counter):
        ttnn.copy_host_to_device_tensor(host_tensor(self.mesh, src, dst.dtype, dst.layout), dst)
        self.counters[counter] += 1

    def _push_sampling_params(self):
        self.sampler.reset_params(self.k_host.clone(), self.p_host.tolist(), self.t_host.clone())
        self.counters["sampling_param_refreshes"] += 1

    def set_sampling_params(self, *, slots=None, top_k=1, top_p=0.0, temperature=1.0, seed=None):
        """Per-row sampling params (device tensors refreshed by host copy; the captured graphs do not change).
        Greedy = k=1, p=0, temp=1.  ``seed``: per-row RNG seed (int, applied to every given row)."""
        rows = list(range(self.B)) if slots is None else list(slots)
        greedy = top_k == 1 or temperature == 0
        if greedy:
            k, p, t = self._greedy
        else:
            if not 1 <= top_k <= 32 or temperature <= 0:
                raise ValueError("device sampling supports top_k 1..32 and temperature > 0")
            k, p, t = top_k, top_p, temperature
        changed = False
        for r in rows:
            changed |= (self.k_host[r], self.p_host[r], self.t_host[r]) != (k, p, 1.0 / t)
            self.k_host[r], self.p_host[r], self.t_host[r] = k, p, 1.0 / t
        if changed:
            self._push_sampling_params()
        if seed is not None:
            # read-modify-write: other slots keep their device-advanced RNG state
            s = ttnn.to_torch(ttnn.get_device_tensors(self.sampler.seeds_tt_tensor)[0]).to(torch.int64).reshape(-1)
            self.counters["seed_readbacks"] += 1
            s[rows] = torch.tensor([seed + r for r in rows], dtype=s.dtype)
            self._copy(s.to(torch.int32), self.sampler.seeds_tt_tensor, "seed_refreshes")

    def _refresh_table(self, table):
        """Copy the decode page table only when it differs from what is on device."""
        if table is None:
            return
        if isinstance(table, ttnn.Tensor):
            raise ValueError("pass the page table as a host tensor; the bound device copy is refreshed on change")
        t = torch.zeros(self.B, self.model.max_blocks, dtype=torch.int32)
        src = torch.as_tensor(table, dtype=torch.int32)
        t[: src.shape[0], : src.shape[1]] = src
        if self.page_host is not None and torch.equal(t, self.page_host):
            return
        self._copy(t, self.model.state.page_table, "page_table_refreshes")
        self.page_host = t

    def _refresh_positions(self, pos):
        """``pos`` [32] next position per slot (-1 idle) -> cur_pos, rope_pos (idle rows 0) and, when the active set
        changed, slot_active / slot_mask."""
        st = self.model.state
        pos = torch.as_tensor(pos, dtype=torch.int64).reshape(-1)
        if pos.numel() != self.B or (pos < -1).any() or (pos >= self.model.max_seq_len).any():
            raise ValueError("decode positions must be [32] with -1 (idle) or 0 <= p < max_seq_len")
        self._copy(pos.to(torch.int32), st.cur_pos, "position_refreshes")
        self._copy(pos.clamp(min=0).to(torch.int32).reshape(1, -1), st.rope_pos, "rope_refreshes")
        self.pos_host = pos.clone()
        active = tuple(int(i) for i in torch.nonzero(pos >= 0).flatten())
        if active != self.active_host:
            a = (pos >= 0).float().reshape(1, 1, -1, 1)
            self._copy(a, st.slot_active, "mask_refreshes")
            self._copy(a.expand(1, 1, self.B, st.slot_mask.shape[-1]), st.slot_mask, "mask_refreshes")
            self.active_host = active

    def _read_tokens(self, t=None):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.model.state.tokens if t is None else t)[0]).reshape(-1).long()

    def _write_tokens(self, ids):
        self._copy(
            torch.as_tensor(ids, dtype=torch.int32).reshape(1, 1, 1, self.B), self.model.state.tokens, "token_refreshes"
        )

    def _set_slot_tokens(self, slots, values):
        """Request boundary: write tokens of ``slots`` without disturbing the device-resident tokens of other active
        slots (one 128-byte readback when another slot is active)."""
        others = [s for s in (self.active_host or ()) if s not in slots]
        cur = self._read_tokens() if others else torch.zeros(self.B, dtype=torch.int64)
        for s, v in zip(slots, values):
            cur[s] = int(v)
        self._write_tokens(cur)

    # ------------------------------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------------------------------
    def allocate_cache(self, num_blocks=None, block_size=BLOCK_SIZE):
        """Standalone cache: 32 slots x max_seq_len, slot s owns blocks 1 + s * bps .. (block 0 = scratch)."""
        bps = blocks_for(self.model.max_seq_len, block_size)
        num_blocks = num_blocks or 1 + self.B * bps
        self.kv = self.model.allocate_kv_cache(num_blocks, block_size)
        self.owns_cache = True
        self.bps = bps
        table = torch.zeros(self.B, self.model.max_blocks, dtype=torch.int32)
        if num_blocks >= 1 + self.B * bps:
            for s in range(self.B):
                table[s, :bps] = torch.arange(1 + s * bps, 1 + (s + 1) * bps, dtype=torch.int32)
        self.slot_table = table
        return self.kv

    def bind_cache(self, kv_cache, page_table=None):
        """Bind an externally allocated cache (same layout as ``Lfm2Model.allocate_kv_cache``) before ``prepare``."""
        if self.traces:
            raise ValueError("bind the cache before the first trace capture (traces hold the cache addresses)")
        self.model.set_kv_cache(kv_cache)
        self.kv, self.owns_cache = kv_cache, False
        if page_table is not None:
            self._refresh_table(page_table)

    # ------------------------------------------------------------------------------------------
    # phase 1: compile every variant; phase 2: capture on first use
    # ------------------------------------------------------------------------------------------
    def prepare(self):
        """Phase 1 warm-up (no capture): every prefill bucket x {first, continuation} chunk with both output heads,
        the prefill sampler, every decode bucket + sampler, the history append, slot-reset / zero / snapshot ops.  Runs
        on the scratch block / idle slots; restores tokens, positions, seeds, history cursor and ShortConv state."""
        if self.prepared:
            return
        assert self.kv is not None, "allocate or bind the cache first"
        m, st = self.model, self.model.state
        t0 = time.perf_counter()
        m.alloc_prefill_inputs()  # persistent per-bucket prefill inputs: before any capture
        conv_backup = [ttnn.clone(s) for s in m.conv_states()]
        backup = [(ttnn.clone(t), t) for t in (st.tokens, st.cur_pos, st.rope_pos, self.sampler.seeds_tt_tensor)]
        zero_row = torch.zeros(m.max_blocks, dtype=torch.int32)  # every block -> scratch block 0
        scratch_slot = self.B - 1
        for S in PREFILL_BUCKETS:
            ids = torch.full((S,), m.args.pad_token_id)
            lg = m.prefill_chunk(ids, chunk_start=0, valid_len=S, slot=scratch_slot, page_table_row=zero_row, out="all")
            ttnn.deallocate(lg)
            if m.max_seq_len >= PREFILL_CHUNK + S:
                lg = m.prefill_chunk(
                    ids, chunk_start=PREFILL_CHUNK, valid_len=S, slot=scratch_slot, page_table_row=zero_row, out="last"
                )
            else:
                lg = m.prefill_chunk(
                    ids, chunk_start=0, valid_len=S, slot=scratch_slot, page_table_row=zero_row, out="last"
                )
            ttnn.add(self.pf_logits, lg, output_tensor=self.pf_logits)  # the prefill-sampling accumulation
            ttnn.deallocate(lg)
        ttnn.fill(self.pf_logits, 0.0, output_tensor=self.pf_logits)
        self._sample(self.pf_logits, self.pf_tokens)
        for b in DECODE_BUCKETS:
            bucket = None if b >= TILE else b
            lg = m.decode_forward_device(bucket)
            self._sample(lg, st.tokens)
            ttnn.deallocate(lg)
        self._append_history()
        # request-boundary ops that may run between replays (reset() zeroes the whole KV cache / ShortConv state: the
        # cache is zeroed here once, the ShortConv state is restored from the backup below)
        m.zero_kv_cache()
        m.zero_conv_state()
        m.reset_conv_slots([scratch_slot])
        snap = m.snapshot_conv_state(scratch_slot)
        m.restore_conv_state(scratch_slot, snap)
        for t in snap:
            ttnn.deallocate(t)
        for s, b in zip(m.conv_states(), conv_backup):
            ttnn.copy(b, s)
            ttnn.deallocate(b)
        for b, t in backup:
            ttnn.copy(b, t)
            ttnn.deallocate(b)
        self._copy(torch.zeros(1, dtype=torch.int32), self.history_cursor, "history_cursor_refreshes")
        ttnn.synchronize_device(self.mesh)
        self.prepared = True
        self.last_perf["prepare_s"] = time.perf_counter() - t0
        logger.info(f"Lfm2Generator.prepare: {self.last_perf['prepare_s']:.1f} s")

    def _sample(self, logits, out_tok):
        self.sampler(logits, tt_out_tok=out_tok)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)

    def _append_history(self):
        updated = ttnn.indexed_fill(self.history_cursor, self.history, self.model.state.tokens, dim=0)
        ttnn.copy(updated, self.history)
        ttnn.deallocate(updated)
        ttnn.plus_one(self.history_cursor)

    def _capture(self, bucket):
        """Phase 2: capture the model + sampling traces of ``bucket`` (once), plus the prefill-sampling and history
        traces on the first call.  Capture records without executing: device state is unchanged."""
        assert self.prepared
        ttnn.synchronize_device(self.mesh)
        t0 = time.perf_counter()
        with self._no_cache_misses():  # every capture below: a program-cache miss fails instead of compiling
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            logits = self.model.decode_forward_device(bucket)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            # logits[bucket] is written by model trace b and read only by sampling trace b, replayed right after it with
            # nothing in between; other traces may reuse its address as scratch (README "Trace allocation")
            _acknowledge(logits)
            sid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._sample(logits, self.model.state.tokens)
            ttnn.end_trace_capture(self.mesh, sid, cq_id=0)
            # free-running variant: sampling + history append in one trace (one replay fewer per step)
            sid_h = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._sample(logits, self.model.state.tokens)
            self._append_history()
            ttnn.end_trace_capture(self.mesh, sid_h, cq_id=0)
            self.traces[bucket] = (tid, logits, sid)
            self.sample_hist_traces[bucket] = sid_h
            self.counters["trace_captures"] += 3
            if self.pf_sample_trace is None:
                self.pf_sample_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._sample(self.pf_logits, self.pf_tokens)
                ttnn.end_trace_capture(self.mesh, self.pf_sample_trace, cq_id=0)
                self.append_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._append_history()
                ttnn.end_trace_capture(self.mesh, self.append_trace, cq_id=0)
                self.counters["trace_captures"] += 2
        self.last_perf.setdefault("capture_s", {})[str(bucket)] = time.perf_counter() - t0

    def _trace(self, bucket):
        if bucket not in self.traces:
            self._capture(bucket)
        return self.traces[bucket]

    # ------------------------------------------------------------------------------------------
    # low level: prefill
    # ------------------------------------------------------------------------------------------
    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache=None,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        start_pos=None,
        sample=True,
        **kwargs,
    ):
        """Prefill ``B`` sequences (each in its own fixed slot) into the bound cache.

        ``tokens`` [B, L_pad] (logical lengths ``prompt_lens``), ``page_table`` [B, max_blocks] (row b = sequence b),
        ``slots`` the ShortConv state rows (default 0..B-1), ``start_pos`` [B] absolute start of these tokens (default 0;
        a continuation must start at a multiple of 2048 and its slot must hold the state of the preceding tokens).
        Returns torch fp32 logits [B, L, 128000] (``return_all_logits``), else the first sampled token per sequence
        [B] (device sampling, ``sample=True``), else last-position logits [B, 1, 128000] (compat)."""
        if kv_cache is not None and kv_cache is not self.kv:
            raise ValueError("kv_cache is not the bound cache (bind_cache first)")
        if not self.prepared:
            self.prepare()
        tokens = torch.as_tensor(tokens)
        page_table = torch.as_tensor(page_table, dtype=torch.int32)
        B = tokens.shape[0]
        slots = list(range(B)) if slots is None else [int(s) for s in slots]
        starts = [0] * B if start_pos is None else [int(s) for s in torch.as_tensor(start_pos).reshape(-1)]
        all_logits, last = [], []
        if not return_all_logits and sample:
            ttnn.fill(self.pf_logits, 0.0, output_tensor=self.pf_logits)
        for b in range(B):
            n = int(prompt_lens[b])
            ids = tokens[b, :n]
            row = page_table[b]
            pos, outs = starts[b], []
            if pos % PREFILL_CHUNK:
                raise ValueError(f"prefill start {pos} must be a multiple of {PREFILL_CHUNK}")
            if pos + n > self.model.max_seq_len:
                raise ValueError(f"prompt end {pos + n} exceeds max_seq_len {self.model.max_seq_len}")
            done = 0
            while done < n:
                c = min(PREFILL_CHUNK, n - done)
                final = done + c == n
                mode = "all" if return_all_logits else ("last" if final else "none")
                lg = self._prefill_chunk(ids[done : done + c], pos + done, c, slots[b], row, mode, sample)
                if mode == "all":
                    outs.append(self.model.logits_to_host(lg)[0, 0, :c])
                    self.counters["full_logits_readbacks"] += 1
                    ttnn.deallocate(lg)
                elif mode == "last" and not sample:
                    last.append(self.model.logits_to_host(lg)[0, 0, slots[b]].reshape(1, -1))
                    self.counters["full_logits_readbacks"] += 1
                    ttnn.deallocate(lg)
                done += c
            self.counters["prefill_chunks"] += (n + PREFILL_CHUNK - 1) // PREFILL_CHUNK
            if return_all_logits:
                all_logits.append(torch.cat(outs, dim=0))
        if return_all_logits:
            L = max(int(x) for x in prompt_lens)
            out = torch.zeros(B, L, self.model.vocab)
            for b, lg in enumerate(all_logits):
                out[b, : lg.shape[0]] = lg
            return out
        if not sample:
            return torch.stack(last)
        if self.pf_sample_trace is None:
            self._capture(decode_bucket(slots))
        ttnn.execute_trace(self.mesh, self.pf_sample_trace, cq_id=0, blocking=False)
        self.counters["prefill_sampling_replays"] += 1
        toks = self._read_tokens(self.pf_tokens)
        return toks[slots]

    def _prefill_chunk(self, ids, chunk_start, n, slot, row, mode, sample):
        """One chunk: host refresh of the bucket's persistent inputs, then the traced chunk (non-blocking replay; the
        sampled-mode last chunk accumulates its logits into ``pf_logits`` inside the trace) or the eager path (all-logits
        readiness / host-compat / ``LFM2_EAGER_PREFILL=1``).  Returns device logits for the eager "all" / unsampled "last"
        modes, else None."""
        m = self.model
        S, host = m.prefill_host_inputs(ids, chunk_start=chunk_start, valid_len=n, slot=slot, page_table_row=row)
        self.counters["prefill_input_refreshes"] += m.load_prefill_inputs(S, host)
        cont = chunk_start > 0
        acc = self.pf_logits if (mode == "last" and sample) else None
        if mode == "all" or (mode == "last" and not sample) or not self.traced_prefill:
            self.counters["prefill_eager_chunks"] += 1
            return m.prefill_chunk_device(S, continuation=cont, out=mode, accumulate_into=acc)
        key = (S, cont, mode)
        tid = self.prefill_traces.get(key)
        if tid is None:
            tid = self._capture_prefill(key)
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        self.counters["prefill_replays"] += 1
        return None

    def _capture_prefill(self, key):
        """Capture the prefill trace of ``key = (bucket, continuation, out)`` once (every program was compiled by
        ``prepare``; capture records without executing)."""
        assert self.prepared
        S, cont, mode = key
        ttnn.synchronize_device(self.mesh)
        t0 = time.perf_counter()
        before = self._trace_bytes()
        with self._no_cache_misses():
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.model.prefill_chunk_device(
                S, continuation=cont, out=mode, accumulate_into=self.pf_logits if mode == "last" else None
            )
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.prefill_traces[key] = tid
        self.counters["prefill_trace_captures"] += 1
        self.last_perf.setdefault("prefill_capture_s", {})[str(key)] = time.perf_counter() - t0
        self.last_perf.setdefault("prefill_trace_bytes_per_bank", {})[str(key)] = self._trace_bytes() - before
        return tid

    def _trace_bytes(self):
        try:
            return ttnn.get_memory_view(self.mesh, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank
        except Exception:  # pragma: no cover
            return 0

    @contextlib.contextmanager
    def _no_cache_misses(self):
        """A program-cache miss inside a capture fails loudly instead of compiling (prepare must have compiled it)."""
        f = getattr(self.mesh, "set_program_cache_misses_allowed", None)
        if f is not None:
            f(False)
        try:
            yield
        finally:
            if f is not None:
                f(True)

    # ------------------------------------------------------------------------------------------
    # low level: decode
    # ------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        *,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        host_sampling=None,
        active_slots=None,
        record_history=False,
        **kwargs,
    ):
        """One decode step for all 32 fixed slots.

        ``tokens`` [B] / [B, 1] (B <= 32, row = slot) and ``start_pos`` [B] (-1 = idle slot) are host refreshes at a
        request boundary; ``None`` keeps the device-resident state (sampled token feedback, device-advanced positions).
        ``page_table`` [B, max_blocks] is copied only when it changed.  ``active_slots`` (static bucket b: only slots < b
        active) defaults to the smallest bucket covering the active slots.  Returns the sampled tokens [32] (torch,
        ``read_from_device``) or None; ``host_sampling`` (compat): host argmax of the full logits, returns [32] tokens.
        """
        if kv_cache is not None and kv_cache is not self.kv:
            raise ValueError("kv_cache is not the bound cache (bind_cache first)")
        if not enable_trace and os.environ.get("LFM2_EAGER_DEBUG") != "1":
            raise ValueError("decode requires tracing (enable_trace=True); LFM2_EAGER_DEBUG=1 for an eager debug step")
        if not self.prepared:
            self.prepare()
        compat = self.host_sampling if host_sampling is None else host_sampling
        self._refresh_table(page_table)
        if start_pos is not None:
            pos = torch.full((self.B,), -1, dtype=torch.int64)
            sp = torch.as_tensor(start_pos, dtype=torch.int64).reshape(-1)
            pos[: sp.numel()] = sp
            self._refresh_positions(pos)
        if tokens is not None:
            tk = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
            if tk.numel() > self.B or (tk < 0).any() or (tk >= self.model.vocab).any():
                raise ValueError("tokens: at most 32 ids (row = slot) in range")
            n = tk.numel()
            if n < self.B and any(s >= n for s in (self.active_host or ())):
                # rows >= n are not given: keep the device-resident tokens of slots still active there
                ids = self._read_tokens()
            else:
                ids = torch.zeros(self.B, dtype=torch.int64)
            ids[:n] = tk
            self._write_tokens(ids)
        active = [int(i) for i in torch.nonzero(self.pos_host >= 0).flatten()]
        if not active:
            raise ValueError("no active slot (start_pos binds the request positions)")
        if (self.pos_host[active] >= self.model.max_seq_len).any():
            raise ValueError("cache capacity exhausted for an active slot")
        bucket = decode_bucket(active) if active_slots is None else (None if active_slots >= TILE else active_slots)
        validate_decode_bucket(self.pos_host, bucket)
        if not enable_trace:  # debug only
            lg = self.model.decode_forward_device(bucket)
            self._sample(lg, self.model.state.tokens)
            ttnn.deallocate(lg)
            self.counters["eager_decode_steps"] += 1
        else:
            tid, logits, sid = self._trace(bucket)
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            self.counters["model_replays"] += 1
            if compat:
                full = self.model.logits_to_host(logits)[0, 0]
                self.counters["full_logits_readbacks"] += 1
                pred = full.argmax(-1)
                self._write_tokens(pred)
            elif record_history:  # sampling + history append (one trace)
                if self.history_count == HISTORY_ROWS:
                    self._flush_history()
                ttnn.execute_trace(self.mesh, self.sample_hist_traces[bucket], cq_id=0, blocking=False)
                self.history_count += 1
                self.counters["sampling_replays"] += 1
                self.counters["history_appends"] += 1
            else:
                ttnn.execute_trace(self.mesh, sid, cq_id=0, blocking=False)
                self.counters["sampling_replays"] += 1
            if record_history and compat:
                self._record()
        self.pos_host[active] += 1
        if compat:
            return pred
        return self._read_tokens() if read_from_device else None

    def _record(self):
        if self.history_count == HISTORY_ROWS:
            self._flush_history()
        ttnn.execute_trace(self.mesh, self.append_trace, cq_id=0, blocking=False)
        self.history_count += 1
        self.counters["history_replays"] += 1

    def _flush_history(self):
        """Read the device token history (one readback per window of up to HISTORY_ROWS steps) and rewind it."""
        if self.history_count:
            h = ttnn.to_torch(ttnn.get_device_tensors(self.history)[0])[: self.history_count, 0, 0, :].long()
            self.counters["history_readbacks"] += 1
            self._history_out.append(h)
        self._copy(torch.zeros(1, dtype=torch.int32), self.history_cursor, "history_cursor_refreshes")
        self.history_count = 0

    def _take_history(self):
        self._flush_history()
        out = torch.cat(self._history_out) if self._history_out else torch.empty(0, self.B, dtype=torch.int64)
        self._history_out = []
        return out

    # ------------------------------------------------------------------------------------------
    # high level
    # ------------------------------------------------------------------------------------------
    def _ensure_cache(self):
        if self.kv is None:
            self.allocate_cache()
        if not self.prepared:
            self.prepare()

    def prefill_logits(self, prompt_token_ids):
        """Readiness prefill: reset, chunked prefill of one prompt in slot 0, logits [1, len, 128000] (fp32)."""
        self._ensure_cache()
        self.reset()
        ids = torch.as_tensor(prompt_token_ids, dtype=torch.int64).reshape(1, -1)
        n = ids.shape[1]
        if n > self.model.max_seq_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_seq_len {self.model.max_seq_len}")
        return self.prefill_forward(
            ids, page_table=self.slot_table[:1], prompt_lens=[n], return_all_logits=True, slots=[0]
        )

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=None,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=None,
        stop_on_eos=None,
        slot=0,
        **kwargs,
    ):
        """Prefill ``prompt_token_ids`` in ``slot`` then decode; returns the model's own predictions (``max_new_tokens``
        tokens, fewer if ``stop_on_eos``).  ``next_input(i, p_i)`` (teacher forcing) chooses the next input each step
        (one token readback + one token write per step: inherent to teacher forcing).  Free-running: the sampled token
        feeds the next replay on device; tokens are appended to a device history read once per window (or every 16
        steps with ``stop_on_eos``).  ``stop_on_eos`` defaults to True when free-running (HF ``generate`` behaviour: the
        returned list ends with the EOS token) and False under teacher forcing (every step is scored)."""
        if stop_on_eos is None:
            stop_on_eos = next_input is None
        if not enable_trace and os.environ.get("LFM2_EAGER_DEBUG") != "1":
            raise ValueError("generate requires traced decode (enable_trace=True)")
        if self.kv is not None and not self.owns_cache:
            raise ValueError(
                "generate() owns the slot -> block mapping: use prefill_forward / decode_forward with a bound cache"
            )
        self._ensure_cache()
        compat = self.host_sampling if host_sampling is None else host_sampling
        if compat and not (top_k == 1 or temperature == 0):
            raise ValueError("host-sampling compatibility mode is greedy only")
        prompt = torch.as_tensor(prompt_token_ids, dtype=torch.int64).reshape(-1)
        n = prompt.numel()
        if n + max_new_tokens > self.model.max_seq_len:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens exceeds max_seq_len {self.model.max_seq_len}")
        # the request takes `slot` (released from any previous request)
        pos = self.pos_host.clone()
        pos[slot] = -1
        self.set_sampling_params(slots=[slot], top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        self._refresh_table(self.slot_table if self.owns_cache else None)
        perf = {}
        t0 = time.perf_counter()
        sample_dev = not compat
        out = self.prefill_forward(
            prompt.reshape(1, -1),
            page_table=self.slot_table[slot : slot + 1],
            prompt_lens=[n],
            slots=[slot],
            sample=sample_dev,
        )
        p0 = int(out[0]) if sample_dev else int(out[0, 0].argmax())
        perf["ttft_s"] = time.perf_counter() - t0
        preds = [p0]
        feed = p0 if next_input is None else int(next_input(0, p0))
        pos[slot] = n
        self._refresh_positions(pos)
        self._set_slot_tokens([slot], [feed])
        eos = self.model.args.eos_token_id
        defer = next_input is None and not compat
        t1 = time.perf_counter()
        steps = 0
        for i in range(1, 1 if (stop_on_eos and p0 == eos) else max_new_tokens):
            if defer:
                self.decode_forward(read_from_device=False, record_history=True)
                steps += 1
                if stop_on_eos and i % 16 == 0:
                    if (self._peek_history(slot) == eos).any():
                        break
                continue
            toks = self.decode_forward(read_from_device=True, host_sampling=compat)
            steps += 1
            p = int(toks[slot])
            preds.append(p)
            if next_input is not None:
                feed = int(next_input(i, p))
                if feed != p:
                    toks = toks.clone()
                    toks[slot] = feed
                    self._write_tokens(toks)
            if stop_on_eos and p == eos:
                break
        if defer:
            hist = self._take_history()
            preds += [int(t) for t in hist[:, slot]]
            if stop_on_eos and eos in preds:
                preds = preds[: preds.index(eos) + 1]
        ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        perf.update(
            decode_steps=steps,
            decode_s=t2 - t1,
            decode_tok_s_u=steps / (t2 - t1) if steps else 0.0,
            mode="teacher_forcing" if next_input is not None else ("host_sampling" if compat else "token_out"),
        )
        self.last_perf.update(perf)
        return preds[:max_new_tokens]

    def _peek_history(self, slot):
        self._flush_history()
        return self._history_out[-1][:, slot] if self._history_out else torch.empty(0)

    def reset(self):
        """Zero the KV cache and ShortConv state, idle every slot, rewind the token history, greedy params, initial
        seeds.  Device buffers and captured traces are kept."""
        if self.kv is None:
            return
        self.model.zero_kv_cache()
        self.model.zero_conv_state()
        self._refresh_positions(torch.full((self.B,), -1, dtype=torch.int64))
        self._write_tokens(torch.zeros(self.B, dtype=torch.int64))
        self._history_out = []
        self.history_count = 0
        self._copy(torch.zeros(1, dtype=torch.int32), self.history_cursor, "history_cursor_refreshes")
        self.set_sampling_params(top_k=1)
        self._copy(self.seed0.to(torch.int32), self.sampler.seeds_tt_tensor, "seed_refreshes")
        self.counters["resets"] += 1

    def reset_slots(self, slots):
        """Free request slots: idle positions, zero their ShortConv state rows (KV pages are owned by the page table)."""
        pos = self.pos_host.clone()
        pos[list(slots)] = -1
        self._refresh_positions(pos)
        self.model.reset_conv_slots(slots)

    def release_traces(self):
        for tid, logits, sid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
            ttnn.release_trace(self.mesh, sid)
        for t in (
            self.pf_sample_trace,
            self.append_trace,
            *self.prefill_traces.values(),
            *self.sample_hist_traces.values(),
        ):
            if t is not None:
                ttnn.release_trace(self.mesh, t)
        self.traces, self.pf_sample_trace, self.append_trace, self.prefill_traces = {}, None, None, {}
        self.sample_hist_traces = {}

    def teardown(self):
        """Explicit end of the trace lifetime (the readiness runners call it after a run).  A shared generator
        (``build_generator(..., shared=True)``) keeps its traces until ``close()``."""
        if not getattr(self, "shared", False):
            self.release_traces()

    def close(self):
        self.release_traces()


def build_generator(
    model_dir=None,
    mesh_device=None,
    *,
    layer_indices=None,
    host_sampling=False,
    sampling_strategy="split",
    pad_logits_to_power_of_2=True,
    max_seq_len=None,
    kv_blocks=None,
    shared=False,
    prepare=True,
    **kwargs,
):
    """Readiness factory.  ``shared=True`` returns one generator per mesh (built once: the driver reuses the 40-layer
    model across the prefill / teacher-forcing / autoregressive runners)."""
    if mesh_device is None:
        raise ValueError("mesh_device is required")
    if tuple(mesh_device.shape) != (1, 4):
        raise ValueError(f"LFM2.5-24B-A2B runs on a 1x4 mesh, got {tuple(mesh_device.shape)}")
    key = (id(mesh_device), tuple(layer_indices or ()), host_sampling, sampling_strategy, max_seq_len)
    if shared and key in SHARED_GENERATORS:
        return SHARED_GENERATORS[key]
    model = Lfm2Model(mesh_device, layer_indices=layer_indices, max_seq_len=max_seq_len, **kwargs)
    gen = Lfm2Generator(
        model,
        host_sampling=host_sampling,
        sampling_strategy=sampling_strategy,
        pad_logits_to_power_of_2=pad_logits_to_power_of_2,
    )
    gen.shared = shared
    gen.allocate_cache(kv_blocks)
    if prepare:
        gen.prepare()
    if shared:
        SHARED_GENERATORS[key] = gen
    return gen
