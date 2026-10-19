# Plan: expanded K on arrival in ring_joint SDPA (Kimi K3 / K2.7, Galaxy 8×4)

Author: Pavle Josipovic (with Claude) · 2026-10-17, updated 2026-10-19 to the final design. Status and evidence:
[STATUS.md](STATUS.md).

Branch `pjosipovic/unrolled-v-tp-sharded-kv`, on top of Phase 3 of the parent plan. It follows
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
| E5 | The absorbed path is unchanged and selectable (`ttMLA(expanded_k=False)`); all existing MLA, op and parent X2–X6 tests pass. Exception: the V expansion's fp32 accumulation (see [STATUS.md](STATUS.md), decisions) also changes absorbed numerics, deliberately | Regression runs |

Every change was reviewed by Claude Code and Codex.

## 1. Why, and what it buys

- **The SDPA is the forward.** Realtime profiler, one 5k chunk, absorbed: `ring_joint` takes 5.9 of 7.8 ms at K3 50k,
  54.4 of 57.8 ms at K3 512k, and 19.7 of 21.6 ms at K2.7 256k.
- **Compute ceiling.** Causal `flash_mla_prefill` on every chip at once, traced, HiFi2, bf8, V 128 per head:
  per-head 192-wide K takes 49–53% less time than a shared 576-wide latent K (16 and 24 heads, 4k and 8k).
- **The expansion adds 32% of the expanded SDPA's MACs** when done once per device per (head, K row): 512 × 128 MACs
  against 640 local query rows × (192 + 128).
- **Projection:** 0.48 × (1 + 0.32/e) of the absorbed SDPA, where e is the expansion's MAC/s relative to the SDPA's:
  −37% SDPA at e = 1.
- **Done redundantly on every core of a head, it loses at K3:** 33.5M MACs of expansion per core per K chunk, against a
  31.5M QK saving. Hence the head pairs (§3.4), which expand each K chunk once per pair.
- **Rejected alternatives:**
  - Expanding K ahead of the SDPA and ringing per-head K: 1.7–2.1× the ring bytes.
  - Expanding all rows up front per device: 2.6 GB per chip at K3 512k.
  - Expanding V on arrival too: each chip would expand 8× more V than today's local expansion.

## 2. Math (per model)

- **Absorbed:** score = (q_nope · W_UK) · c_kv + q_pe · k_pe. Q is 576 wide; K is the cached latent
  [c_kv (512) | k_pe (64)].
- **Expanded:** score = q_nope · (c_kv · W_UKᵀ) + q_pe · k_pe. Q = [q_nope (128) | q_pe (64)]; K_h = [c_kv · W_UK_hᵀ
  (128) | k_pe (64)].
- **The weight is `wkv_b1` as stored:** W_UK, `[1, NH_local, 128, 512]` (out 128, in 512), bf8, head-sharded on TP
  (`mla.py`, reference `mla_reference.py`). No host transpose (§3.2).
- **Scale:** `self.scale` unchanged. That is 192^-0.5 for K3; for K2.7 it is 192^-0.5 × YaRN mscale² ≈ 2.005.
- **RoPE:**
  - K2.7 rotates q_pe once (on its own 64 channels), and the cache's k_pe is already rotated and shared by all heads.
  - K3 is NoPE: its 64 PE channels stay, and neither side is rotated.
  - `_apply_rope_padded(concat_prefix=q_nope)` gets a 128-wide prefix.

## 3. The design

### 3.1 Q width and K width separated

One head-dim tile count, `DHt`, used to size everything. It is split into `DHt_qk` (6 for expanded K) for Q, the QK
`in0_block_w` and the QK K operand, and `DHt_k` (18) for the latent K. The places that use it:

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

The expansion (`sdpa_expand_k_chunk`, `compute_streaming.hpp`) keeps that contract: it computes plain K_h tiles in
K^T tile order.
- **W_UK layout in `cb_wuk`:** the reader stores the core's rows of W_UK latent-column-major, tile (r, d) at
  d · own_rows + r, so a block of W_UK rows for one latent column d is contiguous.
- **Matmul:** blocked and no-MOP (`sdpa_mm_init` / `sdpa_matmul_block_no_mop`), rt K tiles × ct W_UK rows per DST
  acquire, filling the DST (8 tiles with bf16 dest): 4 × 2 under pairs. in1 is the stored W_UK tile, transposed
  in-tile, i.e. tile (d, h) of W_UKᵀ: the same transpose flag the QK uses. The sum runs over the 16 latent tiles d.
- **Fidelity:** the no-MOP helpers take `SDPA_MATMUL_FIDELITY`, so the expansion runs at the SDPA's matmul fidelity
  (LoFi with QK and PV when `sdpa_matmul_fidelity=LoFi`).
- **Output:** packed bf8 into `cb_kexp` rows 0–3, in K^T tile order.

The 2 PE tile rows (d = 16, 17) are copied from `cb_k_in` into `cb_kexp` rows 4–5, as plain tiles in the same order.
QK then runs unchanged, with `in0_block_w = DHt_qk = 6` and transpose on, over a homogeneous 6 × Sk_t block.

### 3.3 Per K chunk, in compute

Wait `cb_k_in`; expand this core's 2 no-PE tile rows into `cb_kexp` and `cb_kexp_own`; copy the 2 PE tile rows; pop
`cb_k_in` early (it is no longer needed); copy the peer's 2 rows from `cb_kexp_peer` into `cb_kexp`; then QK from
`cb_kexp`. A core expands every K chunk the reader delivers, even one its own Q chunk skips (§3.4).

### 3.4 Head pairs share the expansion

**Pairs only.** Expanded K requires that the two Q chunks of each (K-split band, head) sit one per core: one Q chunk per
core (`max_q_per_core == 1`), two Q chunks per head, per-head V chains, batch 1. Both Kimi schedules
(`_EXPANDED_K_SDPA_CHUNKS`, §3.7) satisfy it. The op refuses any other schedule; there is no independent (unpaired)
expansion and no group of 4.

**The pair.** A pair is the cores of `head_segments[band·NH + head]`, the V head chain's existing groups (factory).
- Each core expands 2 of the 4 no-PE tile rows of every K chunk, with only those rows of W_UK resident.
- All participants are inside the SDPA sub-device (x 0–10). The parent plan's prefetch cores (x 11, y 4–9) are never in
  a pair.
- Pairs may straddle rows. The exchange is unicast, and every row receives the same K chunks.

**Participation.** Both cores of a pair walk the same live K chunks. The reader's skips are per device and K-split
band; the per-Q causal skip happens only in compute. So each core expands its half of every K chunk the reader
delivers, and only then may its compute skip QK/PV for a chunk its Q chunk masks out entirely. The skip then drains
the assembled `cb_kexp`.

**Protocol per K chunk, all in the reader after the chunk's V push:**
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

The reader both sends and receives, so it cannot fetch or receive K^T j+1 until the pair has exchanged chunk j. That
limits K prefetch to about one chunk. Moving the send to the writer, or deferring the exchange by one chunk, is the
lever if the profiler ever shows the exchange exposed.

### 3.5 Op interface, hashing, trace

- **New optional input:** `k_expand_weight` `[1, NH_local, nope, latent]` (`128 x 512` for Kimi), bf8, TILE, DRAM
  interleaved. Its shape sets the geometry: the no-PE width must give each head pair core a power-of-two count of whole
  tile rows (the expansion's DST blocks divide it), and the latent whole tiles.
- **When it is present:** Q is `[1, NH, S, nope + pe]` (192 for Kimi), K is the latent plus the same PE width (576),
  and V is as today.
- **Hash:** the weight's presence and spec (dtype, layout, memory config, shape), following the existing optional
  tensors. The pair layout follows from hashed inputs (grid, chunk sizes, shapes).
- **Address:** the weight's buffer address is a runtime argument, patched on cache hits by the override path.
- **Trace:** each layer passes its own persistent `wkv_b1`, so captured addresses stay valid. The test uses distinct
  weights per layer (E4).

### 3.6 L1 budget

Per core, under pairs, with Sk_t K tiles per K chunk (bf8 tiles):

| CB | Tiles |
|---|---|
| `cb_wuk` (W_UK, this core's 2 rows) | 2 × 16 (one buffer: one Q chunk per core) |
| `cb_kexp` (assembled 6 × Sk_t, 2 slots) | 12 · Sk_t |
| `cb_kexp_own` (2 × Sk_t, 2 slots) | 4 · Sk_t |
| `cb_kexp_peer` (2 × Sk_t, 2 slots) | 4 · Sk_t |

Q shrinks from 18 to 6 tiles wide. k512 exceeded the L1 budget beside the expanded K^T CB for both models, so every
expanded-K schedule runs k256 (Sk_t = 8).

### 3.7 MLA

- **`_q_stem`:** with expanded K, skip `wkv_b1`. Q = q_nope ++ (RoPE'd for K2.7) q_pe = 192. This also drops a
  42–55 µs matmul.
- **`_unrolled_v_attn`:** passes `k_expand_weight=self.wkv_b1_weight`.
- **Schedules:** a separate table, `_EXPANDED_K_SDPA_CHUNKS` (kvpe width, local heads, local rows) → (Q chunk, K chunk,
  K splits), beside the absorbed `_UNROLLED_V_SDPA_CHUNKS`. `_unrolled_v_sdpa_chunks` reads it when expanded K is on.
- **Selection:** `mla_config.MLA_ATTENTION_CONFIG` turns it on for Kimi K2.7 / K3, and `ttMLA(expanded_k=...)`
  overrides that. `_expanded_k_supported` keeps it where the schedule table has an entry (5k and 2k chunks) and the
  geometry fits (a power-of-two count of whole no-PE tile rows per head pair core, no llama4 query scale, no
  balanced attention); a model default elsewhere keeps absorbed K, and `expanded_k=True` raises ValueError. On a model without
  unrolled V, `expanded_k=True` is refused (it needs `unrolled_v`).

## 4. Phases

| Phase | Work | Exit |
|---|---|---|
| 1 | Q/K width split (§3.1); `k_expand_weight` and the in-kernel expansion (§3.2) | E1 |
| 2 | Head-pair exchange (§3.4) | E1 with pairs |
| 3 | MLA integration behind `ttMLA(expanded_k)`; chunk retune; perf | E2, E3, E4 |
| 4 | Default on for Kimi; docs | E5 |

All four phases are implemented. E1–E4 are only partly met (◐); [STATUS.md](STATUS.md) has the evidence and the open items.

## 5. Tests

**Op tests:** `ring_joint` with `k_expand_weight` vs a torch un-absorbed reference. Parametrized over:
- heads 16 / 24;
- (q, k) chunks, including K-split bands (K2.7, 3 bands, an empty band);
- chunk 0 (diagonal masking differs per Q chunk);
- rotated starts and partial chunks;
- short → long → short calls on one program (cache hits);
- pairs straddling rows, and idle multicast participants;
- a new weight at a new address on a cache hit;
- scalar and metadata;
- `tp_sharded` single-slot K;
- the parent's prefetch on (SDPA inside the sub-device).

**MLA:** paired PCC (E2) on every cpu scenario × K3 / K2.7 × scalar / metadata × `sp_only` / `tp_sharded`, with
`tp_sharded` == `sp_only` bitwise; and the K3 golden trace.

**Trace:** a 3-layer, 2-user capture with distinct weights, prefetch on and off, poisoned latent buffers, and
allocation tracking (E4).

**Perf:** steady-state traced forward, absorbed vs expanded, both layouts, at the four points; a realtime-profiler
split of ring vs compute.

## 6. Risks

1. **Accuracy:** bf8 expanded K. If E2 fails, pack bf16 (+123 KB of L1 per 96 tiles).
2. **The ring becomes the limit** once compute shrinks: at 2k chunks the SDPA is gather-bound
   (../unrolled_v_tp_sharded_kv/STATUS.md, SP ring gather).

## 7. Out of scope

Mistral 4 (different MLA dims); sparse MLA.
