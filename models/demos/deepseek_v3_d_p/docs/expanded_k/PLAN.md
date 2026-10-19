# Plan: expanded K on arrival in ring_joint SDPA (Kimi K3 / K2.7, Galaxy 8×4)

Status: **rev 2** after review round 1 (Claude Code, Codex gpt-6-astra; §8) · Author: Pavle Josipovic (with Claude) ·
2026-10-17
Branch: `pjosipovic/unrolled-v-tp-sharded-kv`, on top of Phase 3 of the parent plan. It follows
[../unrolled_v_tp_sharded_kv/PLAN.md](../unrolled_v_tp_sharded_kv/PLAN.md) §6 and keeps its constraints:
- the latent stays K's wire format;
- the TP latent buffers keep all 576 columns;
- SP-local row order, the extents and the single-slot read path are unchanged.

## 0. Goal and exit criteria

**Goal:** run QK un-absorbed at 192 wide (128 no-PE + 64 PE per head) instead of 576 wide (absorbed latent). K is
expanded per head inside `ring_joint` SDPA, from the latent it already receives. Only the latent travels on the SP
ring, and V stays unrolled as today.

**Done when every item holds** on the Galaxy 8×4 TorusXY, release build:

| # | Criterion | How it's checked |
|---|---|---|
| E1 | `ring_joint` with `k_expand_weight` matches a torch reference of un-absorbed MLA attention: per-head PCC ≥ 0.999 vs fp32, finite outputs, max error reported. Coverage: K3 and K2.7 shapes, causal, chunk 0 (diagonal), rotated starts, partial chunks, short → long → short program-cache hits, metadata and scalar paths, `sp_only` and `tp_sharded` (single-slot) K | Op tests (§5) |
| E2 | MLA with expanded K: output PCC vs the CPU reference ≥ the absorbed path's PCC − 0.001 on every `test_mla_chunked_prefill` cpu scenario; K3 golden-trace PCC within 0.001 of absorbed. `tp_sharded` == `sp_only` bitwise with expanded K, as parent X2 | Paired tests |
| E3 | Steady-state traced MLA forward (parent X5 method) ≥ 15% faster than absorbed unrolled-V at K3 512k and K2.7 256k, ≥ 8% at the 50k points, both layouts, prefetch on. The realtime profiler shows the SDPA compute-bound (the ring gather below the compute) | Perf test + profiler split |
| E4 | Multi-layer captured trace with distinct per-layer weights, prefetch on and off, replays bitwise equal to eager; allocation tracking clean | Trace test |
| E5 | The absorbed path is unchanged and selectable (`ttMLA(expanded_k=False)`); all existing MLA, op and parent X2–X6 tests pass | Regression runs |
| E6 | Every commit reviewed by Claude Code and Codex gpt-6-astra; findings fixed or answered | `reviews/<sha>.md` |

**Stop and re-plan if** Phase 1's in-situ expansion throughput (MAC/s, from the profiler) is below 60% of the QK
matmul's MAC/s. At that floor the projected SDPA saving is about 26% (§1), the least that still clears E3.
**Waived (author, 2026-10-18):** the ratio could not be pinned down inside the fused kernel (estimate 0.43–0.71),
and E3's timing is met at the end of Phase 2 (−20.7% to −25.1%). The expansion's utilisation was then measured as an
optimisation input, not a gate: 0.74–0.81 of QK's rate (STATUS).

## 1. Why (measured) and what it buys

- **The SDPA is the forward.** Realtime profiler, one 5k chunk: `ring_joint` takes 5.9 of 7.8 ms at K3 50k, 54.4 of
  57.8 ms at K3 512k, and 19.7 of 21.6 ms at K2.7 256k.
- **Compute ceiling.** Causal `flash_mla_prefill` on every chip at once, traced, HiFi2, bf8, V 128 per head
  (`tests/scratch_expand_k_ceiling.py`, 2026-10-17). Per-head 192-wide K vs shared 576-wide latent K:

  | Heads | Chunks | 4k | 8k |
  |---|---|---|---|
  | 24 | q160 k512 | −51% | −53% |
  | 24 | q128 k256 | −51% | −53% |
  | 16 | — | −49% | −52% |

- **The expansion adds 32% of the expanded SDPA's MACs** when done once per device per (head, K row): 512 × 128 MACs
  against 640 local query rows × (192 + 128).
- **Projection:** 0.48 × (1 + 0.32/e) of today's SDPA, where e is the expansion's MAC/s relative to the SDPA's. That is
  0.63 at e = 1, i.e. −37% SDPA (−20 ms of the 57.8 ms K3 512k forward). At the stop floor e = 0.6 it is 0.74, i.e.
  −26%, before exchange overhead.
- **Per core per K chunk** at K3 (k512, q160, 4 cores per head): expansion 8.4M + QK 15.7M + PV 10.5M = 34.6M MACs,
  against 57.7M today (ratio 0.60). The 0.63 above is a different base: the whole SDPA, from the measured ceiling.
- **Done redundantly on every core of a head, it loses at K3:** 33.5M MACs of expansion per core per K chunk, against a
  31.5M QK saving. K2.7 (q320) still gains about 2× on QK even redundantly. Hence Phase 1 (redundant, simple) and
  Phase 2 (shared).
- **The ring today:** the fused ring gather carries the latent (576) and unrolled V (24 × 128 at K3), and takes 43–48%
  of the HiFi2 SDPA time. At 0.63–0.74 compute it would be 61–76% of the compute: probably still hidden, but E3 checks
  it with the profiler.
- **Rejected alternatives:**
  - Expanding K ahead of the SDPA and ringing per-head K: 1.7–2.1× the ring bytes.
  - Expanding all rows up front per device: 2.6 GB per chip at K3 512k.
  - Expanding V on arrival too: each chip would expand 8× more V than today's local expansion.

## 2. Math (per model)

- **Absorbed (today):** score = (q_nope · W_UK) · c_kv + q_pe · k_pe. Q is 576 wide; K is the cached latent
  [c_kv (512) | k_pe (64)].
- **Expanded:** score = q_nope · (c_kv · W_UKᵀ) + q_pe · k_pe. Q = [q_nope (128) | q_pe (64)]; K_h = [c_kv · W_UK_hᵀ
  (128) | k_pe (64)].
- **The weight is `wkv_b1` as stored:** W_UK, `[1, NH_local, 128, 512]` (out 128, in 512), bf8, head-sharded on TP
  (`mla.py`, reference `mla_reference.py`). No host transpose (§3.2).
- **Scale:** `self.scale` unchanged. That is 192^-0.5 for K3; for K2.7 it is 192^-0.5 × YaRN mscale² ≈ 2.005.
- **RoPE:**
  - K2.7 rotates q_pe once (as today, on its own 64 channels), and the cache's k_pe is already rotated and shared by
    all heads.
  - K3 is NoPE: its 64 PE channels stay, and neither side is rotated.
  - `_apply_rope_padded(concat_prefix=q_nope)` now gets a 128-wide prefix; Phase 3 checks that the indexed RoPE op
    accepts it.

## 3. The design

### 3.1 Q width and K width separated (Phase 1)

Today one head-dim tile count, `DHt`, sizes everything. The work is to split it into `DHt_qk` (6 for expanded K) for
Q, the QK `in0_block_w` and the QK K operand, and `DHt_k` (18) for the latent K. The places that use it:

| Layer | Uses |
|---|---|
| Validation | `k_shape[3] == DH` |
| Factory | q_tiles, k_tiles, `qk_in0_block_w`, granularity |
| Reader | Q / K slices |
| Compute | `sdpa_ring_v2` drains |
| Writer | — |
| Placeholders | padded and dummy multicast pushes |

The latent's wire, ring-gather and K^T multicast paths keep `DHt_k`.

### 3.2 Expansion, and tile orientation

`cb_k_in` holds plain latent tiles placed in K^T tile order: tile (d, kt) at index d·Sk_t + kt. `fetch_block` with
`transpose=true` only reorders tiles. QK finishes the transpose in-tile, through the matmul's in1-transpose flag.

The expansion keeps that contract: compute plain K_h tiles in K^T tile order. For each output tile (kt, h) with
h ∈ the core's slice:

```
acc = Σ_d  matmul_tiles(in0 = cb_k_in[d·Sk_t + kt],   in1 = cb_wuk[h·16 + d],   transpose_in1 = true)
```

Here in1 is the stored W_UK tile (h, d), transposed in-tile, i.e. tile (d, h) of W_UKᵀ: the same transpose flag the QK
already uses, so one matmul init serves both. The result is packed to `cb_kexp` index h·Sk_t + kt, bf8.

The 2 PE tile rows (d = 16, 17) are copied from `cb_k_in` into `cb_kexp` rows 4–5, as plain tiles in the same order.
QK then runs unchanged, with `in0_block_w = DHt_qk = 6` and transpose on, over a homogeneous 6 × Sk_t block. No
in-tile transpose, no mixed transpose flags.

### 3.3 Phase 1: every core expands its whole head (no exchange)

- **Per K chunk:** wait `cb_k_in`, expand 4 × Sk_t tiles, copy 2 × Sk_t PE tiles into `cb_kexp`, pop `cb_k_in` early
  (it is no longer needed), then QK from `cb_kexp`.
- **Skipped chunks:** a core whose causal mask skips a chunk skips its expansion too. No coupling.
- **Every schedule is supported,** with redundant work wherever a head spans several cores (K3).

### 3.4 Phase 2: head groups share the expansion (as built: pairs)

**The group.** A head group is the cores of `head_segments[band·NH + head]`, the V head chain's existing groups
(factory).
- **Built:** pairs. That is, n = `num_q_chunks` = 2 with `max_q_per_core == 1`, per-head V chains and B = 1, which
  covers both Kimi table schedules.
- **Not built yet:** groups of 4 (e.g. K3 at q160), which are Phase 3 work.
- **Everything else** keeps Phase 1's independent path; the factory selects it, never fatally.
- **All participants** are inside the SDPA sub-device (x 0–10). The Phase 3 prefetch cores (x 11, y 4–9) are never
  in a group.
- **Pairs may straddle rows.** The exchange is unicast, and every row receives the same K chunks.

**Participation.** Both cores of a pair walk the same live K chunks. The reader's skips are per device and K-split
band; the per-Q causal skip happens only in compute. So each core expands its half of every K chunk the reader
delivers, and only then may its compute skip QK/PV for a chunk its Q chunk masks out entirely. The skip then drains
the assembled `cb_kexp`.

**Protocol per K chunk (as built), all in the reader after the chunk's V push:**
1. Reserve this chunk's `cb_kexp_peer` slot. Increment the peer's credit semaphore (the slot is free).
2. Wait for compute's own rows in `cb_kexp_own`. Compute packs them into its `cb_kexp` and into `cb_kexp_own`.
3. Wait credit ≥ chunks exchanged + 1. NoC-write own rows into the peer's `cb_kexp_peer` slot, which is at the same
   L1 address as this core's slot: identical CB layout, and both cores push and pop in lockstep. Write barrier, then
   increment the peer's data semaphore. Pop `cb_kexp_own`.
4. Wait data ≥ chunks exchanged + 1. Push `cb_kexp_peer`. Compute copies the peer's rows into `cb_kexp`.

**Semaphores and ordering:**
- Counts are cumulative; every increment is consumed before the kernel ends, and the program launch resets them.
- The atomic barrier at reader exit flushes the increments.
- Credit is sent before any wait, so no core waits on its peer in a cycle.

**Difference from rev 2:** rev 2 had the writer send. As built, the reader both sends and receives, so it cannot fetch
or receive K^T j+1 until the pair has exchanged chunk j. That limits K prefetch to about one chunk, and it couples a
row's multicast to its pairs' exchanges.
- **Result:** Phase 2 measured −20.7% to −25.1% vs absorbed (STATUS).
- **Next lever:** moving the send to the writer, or deferring the exchange by one chunk, which only matters if the
  profiler shows the exchange exposed.

### 3.5 Op interface, hashing, trace

- **New optional input:** `k_expand_weight` `[1, NH_local, 128, 512]`, bf8, TILE, DRAM interleaved. Validated to
  exactly this spec.
- **When it is present:** Q is `[1, NH, S, 192]`, K is the 576-wide latent (validated as 512 + 64), and V is as today.
- **Hash:**
  - the weight's presence and spec (dtype, layout, memory config, shape), following the existing optional tensors;
  - the mode (independent or group), n, and the expanded-K pack format.
- **Address:** the weight's buffer address is a runtime argument, patched on cache hits by the override path.
- **Trace:** each layer passes its own persistent `wkv_b1`, so captured addresses stay valid. The test uses distinct
  weights per layer (E4).

### 3.6 L1 budget (K3, k512, n = 4; Phase 1 needs the whole weight)

| CB | Today | Expanded, Phase 2 | Expanded, Phase 1 |
|---|---|---|---|
| Q (bf16) | 90 tiles, 184 KB | 30 tiles, 61 KB | 61 KB |
| K^T latent `cb_k_in` (bf8, 2 slots) | 627 KB | 627 KB | 627 KB |
| W_UK (bf8) | — | 4/n × 16 tiles, 17 KB | 64 tiles, 70 KB |
| `cb_kexp` (bf8, assembled 6 × 16, 2 slots) | — | 209 KB | 209 KB |
| `cb_kexp_own` (1 × 16, 2 slots) | — | 35 KB | — |
| `cb_kexp_peer` (3 × 1 × 16, 2 slots) | — | 104 KB | — |
| Rest | unchanged | | |

- **Phase 1** is +152 KB, and **Phase 2** is +222 KB, on top of today's ≈1.38 MB.
- **Neither fits at k512.** The fallback is k256 for K3, which halves `cb_k_in` and `cb_kexp` (−418 KB).
- Phase 1 builds the full CB plan, prints it, and checks it against the kernel config buffer limit noted at the K-split
  epilogue. The chunk choice is re-measured (§3.7).
- A bf16 `cb_kexp` fallback (accuracy) adds +123 KB per 96 tiles.

### 3.7 MLA

- **`_q_stem`:** with `ttMLA(expanded_k=True)`, skip `wkv_b1`. Q = q_nope ++ (RoPE'd for K2.7) q_pe = 192. This also
  drops a 42–55 µs matmul.
- **`_unrolled_v_attn`:** passes `k_expand_weight=self.wkv_b1_weight`.
- **`_unrolled_v_sdpa_chunks`:** the table key gains the mode, `(576, heads, rows, expanded)`, with entries re-measured
  for the expanded kernel.

## 4. Phases

| Phase | Work | Exit |
|---|---|---|
| 1 | Q/K width split (§3.1); `k_expand_weight` with every core expanding its whole head (§3.2–3.3); L1 plan at k256/k512; in-situ expansion MAC/s from the profiler | E1 on both shapes (independent mode); throughput ≥ 60% of QK, else stop (waived, §0) |
| 2 | Head-group exchange (§3.4) | E1 in group mode; per-core work drops as §1 |
| 3 | MLA integration behind `ttMLA(expanded_k)` (default off); chunk retune; perf | E2, E3, E4 |
| 4 | Default on for Kimi if E3 holds; docs | E5, E6 |

Every commit gets the two reviews (E6).

## 5. Tests

**Op tests:** `ring_joint` with `k_expand_weight` vs a torch un-absorbed reference. Parametrized over:
- heads 16 / 24;
- (q, k) chunks, including K-split bands (K2.7, 3 bands, an empty band);
- chunk 0 (diagonal masking differs per Q chunk);
- rotated starts and partial chunks;
- short → long → short calls on one program (cache hits);
- groups straddling rows, and idle multicast participants;
- a schedule with `max_q_per_core > 1` (falls back to independent);
- a new weight at a new address on a cache hit;
- scalar and metadata;
- `tp_sharded` single-slot K;
- the Phase 3 prefetch on (SDPA inside the sub-device).

**MLA:** paired PCC (E2) on every cpu scenario × K3 / K2.7 × scalar / metadata × `sp_only` / `tp_sharded`, with
`tp_sharded` == `sp_only` bitwise; and the K3 golden trace.

**Trace:** a 3-layer, 2-user capture with distinct weights, prefetch on and off, poisoned latent buffers, and
allocation tracking (E4).

**Perf:** steady-state traced forward, absorbed vs expanded, both layouts, at the four points; a realtime-profiler
split of ring vs compute.

## 6. Risks

1. **L1 at k512** (§3.6): likely k256 for K3, which moves the (q, k) efficiency. Re-measured.
2. **Expansion throughput:** per-tile `matmul_tiles` (single inner tile per call) may be unpack-bound. That is the stop
   criterion. Mitigation: a blocked variant once the orientation is proven.
3. **Exchange exposure:** a NoC exchange per K chunk on the critical path. Measured in Phase 2.
4. **Accuracy:** bf8 expanded K. If E2 fails, pack bf16 (+123 KB of L1).
5. **The ring becomes the limit** at 0.63–0.74 compute, at 61–76%. The parent plan's §7 levers (the two-cycle latent
   ring) come next if the profiler shows it.

## 7. Out of scope

LoFi (Kimi runs HiFi2); Mistral 4 (different MLA dims); sparse MLA.

## 8. Changes from rev 1 (review round 1)

**Claude, majors:**
- Q/K width coupling (§3.1).
- Tile orientation (§3.2).
- Exchange flow control (§3.4).
- Which kernel exchanges, and in what order (§3.4).

**Claude, minors:**
- weight naming;
- scale;
- RoPE / NoPE;
- the RoPE prefix width;
- the chunk-table key;
- the projection range;
- the ring margin;
- the Phase 1 L1;
- the ring bytes;
- the cache-hit patching;
- the missing tests.

**Codex, blockers:**
- Orientation (§3.2).
- Q/K coupling (§3.1).
- The progress protocol, including causal participation (§3.4).

**Codex, majors:**
- Supported scheduling domain and row-straddling groups.
- Scale / RoPE.
- No compute overlap: comm-only double buffering.
- The full L1 budget.
- Throughput-conditional projection; the "<15% means ring" inference was dropped.
- Phase 3 invariants: participants inside the SDPA sub-device, per-layer weights in traces.
- Hashing.
- Scheduling-transition tests.

**Codex, minors:** the bf16 delta (123 KB); the chunk-table key.
