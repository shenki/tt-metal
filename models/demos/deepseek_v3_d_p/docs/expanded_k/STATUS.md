# Expanded K on arrival: status

All phases of [PLAN.md](PLAN.md) are complete. Branch `pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-19.
Everything was measured on one Blackhole Galaxy 8×4 (TorusXY, high-power host).

**On by default** for Kimi K2.7 / K3 (`mla_config.MLA_ATTENTION_CONFIG`) wherever a measured schedule exists (5k and
2k chunks); other shapes keep absorbed K, and other models run `ring_mla`. `ttMLA(expanded_k=False)` turns it off, and
`expanded_k=True` raises when it cannot apply (ValueError; on a model without unrolled V, an assertion).

## What it does

`ring_joint` SDPA takes a new optional `k_expand_weight`, which is W_UK per head (`wkv_b1` as stored). K stays the
shared 576-wide latent on every data-movement path: DRAM, the SP ring gather and the row multicast. For each K chunk,
the compute kernel builds the per-head K^T chunk the QK reads (`sdpa_expand_k_chunk`, `compute_streaming.hpp`):
- **No-PE rows:** 4 tile rows of c_kv @ W_UK^T, split over the head's pair of cores (2 rows each). A blocked no-MOP
  matmul (`sdpa_mm_init` / `sdpa_matmul_block_no_mop`) in full DST blocks (4 × 2), with W_UK stored latent-column-major
  in `cb_wuk` (tile (r, d) at d · own_rows + r) and transposed in-tile (the QK's own flag). It runs at the SDPA's
  matmul fidelity (`SDPA_MATMUL_FIDELITY`), so LoFi together with QK and PV.
- **PE rows:** the 2 PE tile rows of the latent, copied.
- **Exchange:** the reader sends this core's rows to the peer's `cb_kexp_peer` slot (credit, NoC write, data count)
  and hands the peer's rows to compute, which copies them into the remaining rows. A core expands and exchanges every
  chunk the reader delivers, including chunks its own Q chunk then skips as fully masked, since the peer needs them.

The QK then runs unchanged at 192 wide (6 tiles), against 576 (18 tiles) absorbed.

The rest of the change:
- **Q:** un-absorbed, `[q_nope | q_pe]`. `_q_stem` skips `wkv_b1`.
- **W_UK delivery:** the reader pushes the core's rows of its head's W_UK with every Q push, and compute pops them
  with Q.
- **Validation:** `k_expand_weight` requires causal self-attention, one shared K head, a separate per-head V,
  streaming compute, batch 1, a pair schedule (one Q chunk per core, two per head, per-head V chains), no rotated Q
  split and no segmented accumulation, and no joint, sink, sliding, circular, split or balanced attention.
- **Hash:** the weight's spec is hashed; its address is a runtime argument.
- **Chunks** (`_EXPANDED_K_SDPA_CHUNKS`): 5k chunks run q320 / k256 in 2 K-split bands (K3) or 3 (K2.7); 2k chunks
  run q128 / k256 in 2 (K3) or 3 (K2.7). k512 does not fit L1 beside the expanded K^T CB.

## Decisions

- **On by default for Kimi** (author, 2026-10-18); Phase 4 closed when E5 passed on the new default.
- **Pairs only.** The op refuses any schedule that is not a head pair. Unpaired (per-core, redundant) expansion and
  forced schedules are removed. Groups of 4 were dropped: they only help 4-Q-chunk-per-head schedules, which neither
  Kimi model runs, and K3's 4-core schedule (q160) was 11.5% slower with per-core expansion.
- **Schedules** (chunk retune, steady-state traced `sp_only`, one run each):
  - K3 512k: q320 / k256 / 2 splits 47.09 ms; k128 / 2 splits 53.74; k256 / 1 split 82.73.
  - K2.7 256k: q320 / k256 / 3 splits 16.55 ms; k128 / 3 splits 18.72; k256 / 2 splits 22.61.
  - 2k chunks, 512k HiFi2 sweep: K3 q128 / k256 / 2 28.7 ms vs 46.2 for 1 band and 32.7 for q256 / 4 bands; K2.7
    19.9 vs 25.9 / 24.3.
- **Blocked no-MOP expansion at the SDPA's fidelity.** Under LoFi (traced, expanded KV): −7.9% at KV 512k with 5k
  chunks (K3 38.69 → 35.62 ms, K2.7 26.33 → 24.26 ms), −4.8 to −6.6% at 55k; −1.3 to −3.5% with 2k chunks, which are
  gather-bound. At HiFi2 unchanged within noise, bit-identical PCC. LoFi output PCC vs the CPU reference moves by at
  most −0.00005 (K3 5k 0.99819 → 0.99815, 2k 0.99908 → 0.99904; K2.7 5k 0.99732 → 0.99727, 2k 0.99826 → 0.99826),
  still at or above main (K3 5k 0.99796, K2.7 5k 0.99726).
- **V expansion accumulates in fp32, at HiFi2.** Accumulating in bf16 cost 0.001 (K2.7) to 0.005 (K3) output PCC
  against main's `ring_mla`. CPU reference, 10k prefix + two 5k chunks, HiFi2 SDPA, expanded KV: K3 0.99322 → 0.99828
  (main 0.99805), K2.7 0.99637 → 0.99751 (main 0.99738). HiFi4 gave the same PCC. 512k forward vs bf16
  accumulation: K3 47.06 → 47.15 ms (+0.2%), K2.7 31.77 → 31.78 ms. bf8 storage of V was not the cause, so
  `ring_joint` needs no mixed-dtype support. This applies to every unrolled-V path (both layouts, absorbed and
  expanded K).
- **Rejected, one line each:**
  - Blocking alone (MOP `matmul_block` at HiFi2): ~3% faster per tile in the profile (39.5 → 38.2 cycles at K3,
    43.8 → 42.4 at K2.7), all 16 traced cells within noise.
  - No-MOP at HiFi2: no faster (39.1 / 43.3 cycles); the no-MOP path pays only by following the SDPA to LoFi.
  - Expanding in place (no separate `cb_kexp`, peer writes into the latent chunk): the copies' ceiling is 1–1.5% end
    to end (timing-only kernel, K3 512k HiFi2 47.11 → 46.43 ms), and it would cut the reader's K prefetch to one
    chunk. Batching the copies per DST acquire left every traced cell within noise.
  - Ring CCL workers on row 0, or swapping the gather's NoCs: +11–27% at 2k
    ([../unrolled_v_tp_sharded_kv/STATUS.md](../unrolled_v_tp_sharded_kv/STATUS.md), SP ring gather).
  - V on arrival (ring only the latent, expand V per head in the SDPA): 4.6–6.3× fewer ring bytes, but each chip
    would expand 8× more V than the local expansion; the napkin model has it losing.

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| E1 | Op correctness vs un-absorbed reference | ◐ | `test_ring_joint_expanded_k_per_head_vs_reference`: one chunk at position 0, full or valid-rows-only × `sp_only` / `tp_sharded` (single-slot K) × scalar / metadata × K2.7 / K3 × absorbed / expanded. SDPA output per (query chip, head) vs fp32 attention on its own inputs: 32/32, min PCC 0.99951 (gate 0.999), max abs error ≤ 0.053, finite. `test_ring_joint_expanded_k_causal_vs_reference` (non-chunked causal, per-Q skip with pairs, no K split): 4/4, min PCC 0.99948. Rotated starts, multi-user, partial chunks and cache-hit sequences are checked only against absorbed on the device: per (device, head) PCC ≥ 0.99968 over 40 runs. Weight address on cache hits: `test_mla_expanded_k_distinct_layer_weights` |
| E2 | MLA PCC vs CPU reference ≥ absorbed − 0.001; `tp_sharded` == `sp_only` bitwise | ◐ | `test_mla_expanded_k_matches_absorbed` (rot-multislab, rot_partial, maxedge-1u × K2.7 / K3 × scalar / metadata × `sp_only` / `tp_sharded`): 24/24; over 40 user-runs expanded is at worst 3e-6 below absorbed. Expanded vs absorbed output PCC ≥ 0.99979, per (device, head) SDPA output ≥ 0.999. X2 `expanded_k`, `tp_sharded` == `sp_only` bitwise: 36/36. The full CPU matrix is not run. **Model level, with weights:** K3 golden trace, full output PCC 0.999579 expanded vs 0.999524 absorbed; X2 golden bitwise 4/4. K2.7 `test_kimi_prefill_transformer_chunked_padded` (L10, mid15k, `tp_sharded`, prefetch on, traced and untraced): pass; min per-layer PCC over 54 checks 0.994921 vs 0.995063 absorbed. K3 `kimi_k3/test_chunked_prefill.py` (11 chunks to 56k, `sp_only` and `tp_sharded`): L5 MLA layer 0.995014 vs 0.995034; L24 worst layer 0.980685 vs 0.980931, KV PCC within 2e-4. K3 `test_kimi_k3_prefill_transformer_chunked` (L24, traced): KV cache min PCC 0.994524 (KV only; no output PCC) |
| E3 | ≥ 15% faster at K3 512k / K2.7 256k, ≥ 8% at 50k; SDPA compute-bound | ◐ | **Timing met:** −20.7% to −25.1% at the four points (Performance, below). **Compute-bound, partly shown:** the SDPA program (realtime profiler) drops 17–18% when its matmuls go to LoFi (K3 512k 42.35 → 34.86 ms, K2.7 256k 14.74 → 12.18 ms), so compute is on the critical path. The ring K+V gather run alone takes 22.65 ms at K3 512k and 8.20 ms at K2.7 256k, 53% and 56% of the expanded SDPA program. Per-RISC wait time is not split out, so exposed exchange or gather waits are not separated from compute |
| E4 | Multi-layer trace with distinct weights | ◐ | X4 `expanded_k` (K2.7, K3): 3 layers with distinct per-layer `kv_b`, 2 users, segmented capture with the prefetch, allocation tracking; traced == eager == `sp_only` bitwise, 4/4. A stale weight address is caught by `test_mla_expanded_k_distinct_layer_weights`. Prefetch off is not traced |
| E5 | Absorbed path unchanged, selectable | ✅ | `ttMLA(expanded_k=False)`. Regression set on the default-on build, all pass: X2 full 204; expanded vs absorbed 5k 88, 2k 24, LoFi 4 (pairs), pairs without a K split 8; attention selection (Kimi defaults and overrides, Mistral on `ring_mla`) 9 and overrides (Kimi forced to `ring_mla`, `expanded_k=True` refused off-schedule) 1; accuracy floor 4; X4 8; prefetch 4; X5 4; ring split-KV 2; K2.7 transformer 2; K3 golden MLA 28; K3 L5 2; absorbed X2 rotated subset 24/24 bitwise. X2 (48), X4, prefetch, X5, K3 golden and the expanded-K suites were rerun on the final (blocked) expansion kernel. Perf gates re-baselined for TorusXY on a DDR 14000 host, one forward each: K3 realtime 6.368 ms; Tracy per forward K2.7 4.089 ms, K3 5.883 ms (main `ring_mla` 6.43 / 9.86 ms on the same host). The Tracy selectors pin every axis, and the merge requires each iteration to be exactly one MLA_START/MLA_END region |

## Accuracy (E2)

PCC vs the CPU reference, from the harness's own check:

| Scenario | Absorbed | Expanded |
|---|---|---|
| K2.7 rot-multislab, scalar, `sp_only` | 0.99613 | 0.99616 |
| K2.7 rot_partial, metadata, `tp_sharded` | 0.99660 | 0.99661 |
| K3 rot-multislab, scalar, `sp_only` (q320 / 2 bands) | 0.99584 | 0.99587 |

## Performance (E3)

Traced replays of a steady-state layer (layer 1 of 3), the parent plan's X5 method, HiFi2, one run each. Measured on
the pair kernel before the blocked expansion; the blocked expansion left every HiFi2 cell within noise. Main's
`ring_mla` was measured with an equivalent traced harness at `a7a3bc8f603`.

| Point | Absorbed (unrolled-V) | Expanded K, `sp_only` | vs absorbed | Expanded K, `tp_sharded` + prefetch | main `ring_mla` |
|---|---|---|---|---|---|
| K3 50k | 7.582 ms | 6.014 ms | −20.7% | 6.054 ms | 10.33 ms |
| K3 512k | 61.779 ms | 47.063 ms | −23.8% | 46.851 ms | 89.61 ms |
| K2.7 50k | 5.467 ms | 4.227 ms | −22.7% | 4.210 ms | 6.83 ms |
| K2.7 256k | 22.127 ms | 16.579 ms | −25.1% | 16.404 ms | 29.46 ms |

`tp_sharded` with the prefetch is within ±1% of `sp_only`. The K3 512k forward is 47% faster than main's `ring_mla`.

**Expansion utilisation, single-tile kernel (before blocking).** Device-profiler sum zones around the expansion's
matmul + pack and the QK's, one eager `sp_only` forward, 5k chunk, pairs; math-thread cycles per 32×32×32 tile matmul
over all SDPA cores:

| Point | Expansion | QK | Expansion share of SDPA kernel | QK share |
|---|---|---|---|---|
| K3 512k | 39.5 cycles | 32.1 | 18.2% | 27.7% |
| K2.7 256k | 43.8 cycles | 32.3 | 19.7% | 27.4% |

QK runs at about the nominal HiFi2 rate (two 16-cycle fidelity passes). The gap to it is not operand reloads (blocking
did not close it); its cause is untested: the in-tile transpose of W_UK, the 16-tile inner sums, or the packs.

## Open items

1. E3: the ring-vs-compute split (per-RISC wait time) for the profiler criterion.
2. E1: an fp32 reference for rotated starts and short → long → short sequences (today checked only against absorbed).
3. E2: the full `test_mla_chunked_prefill` CPU matrix with expanded K.
4. E4: trace the prefetch-off case.
5. CI perf gates: re-cut on a DDR 16000 host (today's are from a DDR 14000 host).
6. LoudBox / QuietBox: `kimi_k3/test_mla_perf.py` `PERF_TARGETS` are stale with expanded K on by default and need
   re-measuring on those boxes.
