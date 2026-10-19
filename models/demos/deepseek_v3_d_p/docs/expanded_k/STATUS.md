# Expanded K on arrival: status

Milestone: Phases 1 and 2 of [PLAN.md](PLAN.md). In Phase 1 every core expands its head's K itself; in Phase 2 the
two cores of each head pair split the expansion and exchange their halves. Branch
`pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-18. Everything was measured on one Blackhole Galaxy 8×4
(TorusXY, high-power host). The feature is off by default: `ttMLA(expanded_k=True)` turns it on.

## What it does

`ring_joint` SDPA takes a new optional `k_expand_weight`, which is W_UK per head (`wkv_b1` as stored). K stays the
shared 576-wide latent on every data-movement path: DRAM, the SP ring gather and the row multicast. For each K chunk,
the compute kernel builds the per-head K^T chunk the QK reads (`sdpa_expand_k_chunk`, `compute_streaming.hpp`):
- **No-PE rows:** 4 tile rows of c_kv @ W_UK^T. These are per-tile matmuls, in1 is transposed in-tile (the QK's own
  flag), and the result is packed in K^T tile order.
- **PE rows:** the 2 PE tile rows of the latent, copied.

The QK then runs unchanged at 192 wide (6 tiles), against 576 (18 tiles) absorbed.

The rest of the change:
- **Q:** un-absorbed, `[q_nope | q_pe]`. `_q_stem` skips `wkv_b1`.
- **W_UK delivery:** the reader pushes the core's head's W_UK with every Q push, and compute pops it with Q.
- **Validation:** `k_expand_weight` requires causal self-attention, one shared K head, a separate per-head V,
  streaming compute, batch 1, and no joint, sink, sliding, circular, split or balanced attention.
- **Hash:** the weight's spec is hashed; its address is a runtime argument.
- **Chunks:** K3 runs q320 / k256 in 2 K-split bands (k512 does not fit L1 beside the expanded K^T CB); K2.7 runs
  q320 / k256 in 3 bands (unchanged).

**Phase 2, head groups:**
- **When it applies:** with one Q chunk per core, per-head V chains, batch 1 and two Q chunks per head (both table
  schedules), the two cores of each (K-split band, head) form a pair.
- **What each core does:** it expands 2 of the 4 no-PE tile rows of every K chunk, with only those rows of W_UK
  resident. It packs them both into its K^T chunk and into `cb_kexp_own`.
- **The exchange:** the reader sends `cb_kexp_own` to the peer's `cb_kexp_peer` slot: a credit first (its own slot is
  free), then the NoC write, then the data count. It hands the peer's rows to compute, which copies them into the
  remaining rows.
- **Causal skips:** a core expands and exchanges every chunk the reader delivers, including chunks its own Q chunk
  then skips as fully masked, since the peer needs them.
- **Unchanged:** the factory's `head_segments` are the pairs. Any other schedule keeps Phase 1.

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| E1 | Op correctness vs un-absorbed reference | ◐ | `test_ring_joint_expanded_k_per_head_vs_reference`: one chunk at position 0, full or valid-rows-only (2560 tokens; at position 0 this adds the unwritten pad rows, not partial-chunk masking) × `sp_only` / `tp_sharded` (single-slot K) × scalar / metadata × K2.7 / K3 × absorbed / expanded. The SDPA output per (query chip, head) is compared with fp32 attention on its own inputs: 32/32, min PCC 0.99951 (gate 0.999), max abs error ≤ 0.053, finite. Rotated starts and cache-hit sequences are checked only against absorbed on the device: per (device, head) PCC ≥ 0.99962 over rotated, multi-user and partial scenarios and forced schedules. Weight address on cache hits: `test_mla_expanded_k_distinct_layer_weights`. **Phase 2** (pairs exchanging rows): E1 32/32, per (device, head) vs absorbed ≥ 0.99968 over 40 runs, schedules 16/16, distinct weights 2/2. Non-chunked causal (per-Q skip with pairs, no K split): `test_ring_joint_expanded_k_causal_vs_reference` 4/4, min PCC 0.99948. Forced pairs without a K split (q320/k256, 1 split) in the schedule test |
| E2 | MLA PCC vs CPU reference ≥ absorbed − 0.001; `tp_sharded` == `sp_only` bitwise | ◐ | Asserted directly (the harness's `return_pccs`); over 40 user-runs, expanded is at worst 3e-6 below absorbed and at best 3.4e-5 above. `test_mla_expanded_k_matches_absorbed` (rot-multislab, rot_partial, maxedge-1u × K2.7 / K3 × scalar / metadata × `sp_only` / `tp_sharded`): 24/24 pass. With `return_outputs`, the harness runs its per-iteration 0.98 checks but not the full-region and KV-cache comparisons; the full-region PCC is returned (`return_pccs`) and compared. Expanded vs absorbed: output PCC ≥ 0.99979 (gate 0.999), and per (device, head) SDPA output PCC ≥ 0.999. `tp_sharded` == `sp_only` bitwise with expanded K: 56/56 functional scenarios. The full CPU matrix is not run yet. **Model level, Phase 2 kernel, with weights:** K3 golden trace (KIMI_K3_HF_MODEL, 14 passed per mode), full measured output PCC 0.999579 expanded vs 0.999524 absorbed (within PLAN's 0.001; expanded is higher); X2 golden bitwise 4/4. K2.7 transformer `test_kimi_prefill_transformer_chunked_padded` (L10, mid15k, `tp_sharded`, prefetch on, traced and untraced): both pass; over 54 per-layer decoder checks min PCC 0.994921 expanded vs 0.995063 absorbed (−0.00014), min nPCC 0.997048 vs 0.996934, KV cache min PCC 0.994869 vs 0.994724. K3 `kimi_k3/test_chunked_prefill.py` L5 (11 chunks of 5k to 56k vs golden, mesh fabric, `sp_only` and `tp_sharded`): both pass in both modes; the MLA layer's PCC at 56k is 0.995014 expanded vs 0.995034 absorbed, the worst layer 0.993055 vs 0.992996, KV slot identical (lora 0.997860, rope 0.996174). Same test at L24 (6 MLA layers): both layouts pass in both modes; worst layer at 56k is L19 in both, 0.980685 expanded vs 0.980931 absorbed (−0.00025); per-MLA-layer KV PCC within 2e-4. K3 transformer `test_kimi_k3_prefill_transformer_chunked` (L24, 11 chunks, traced, expanded): passes; it compares only the KV cache with the golden (no output PCC), min 0.994524 (absorbed not rerun; CI's leg). In these model-level runs the mode is set by `ttMLA(expanded_k)` and recorded only in the log names; no "ignored" warning fired, and expanded and absorbed numbers differ. **Phase 2:** X2 `expanded_k` bitwise `tp_sharded` == `sp_only` 36/36 (rot, maxedge, fullchunk) |
| E3 | ≥ 15% faster at K3 512k / K2.7 256k, ≥ 8% at 50k; SDPA compute-bound | ◐ | **Timing met.** Phase 2, steady-state traced, `sp_only` vs absorbed: K3 512k −23.8%, K2.7 256k −25.1%, K3 50k −20.7%, K2.7 50k −22.7%; `tp_sharded` with the prefetch is within ±1% of `sp_only` (table below), one run each. **Ring vs compute, partly measured:** the ring K+V gather run alone (same ring and links, no compute, no contention) takes 22.65 ms at K3 512k and 8.20 ms at K2.7 256k, 53% and 56% of the expanded SDPA program (42.35 / 14.74 ms, below). The gather alone is about half the fused program's elapsed time (which itself includes any exposed waits), but this does not prove it is hidden inside the fused program; per-RISC wait time is still not split out, so this part of E3 stays open. The SDPA program (realtime profiler) drops 17–18% when only its matmuls go to LoFi (K3 512k 42.35 → 34.86 ms, K2.7 256k 14.74 → 12.18 ms, one run each), so compute is on the critical path and the program is not ring-gather-bound. It does not separate exposed exchange or gather waits from compute (LoFi also speeds the expansion). Method: `test_mla.py`'s profile mode with `num_cache_layers=3`, `ttMLA.sdpa_matmul_fidelity` set to LoFi after construction (a local scratch harness) |
| E4 | Multi-layer trace with distinct weights | ◐ | X4 `expanded_k` (K2.7, K3): 3 layers with distinct per-layer `kv_b`, 2 users, a segmented capture with the prefetch, allocation tracking. Traced == eager == `sp_only`, bitwise: 4/4 with absorbed. All three paths are expanded, so X4 would not see a stale weight address; `test_mla_expanded_k_distinct_layer_weights` checks that. Prefetch off is not traced yet. **Phase 2:** X4 4/4 with allocation tracking |
| E5 | Absorbed path unchanged, selectable | ◐ | Off by default. Absorbed X2 subset (rotated scenarios, both variants and paths): 24/24 bitwise. The Phase 3 prefetch test passes on the same build; X4 `absorbed` passes (with per-layer weight scaling added for E4) |
| E6 | Two reviews per commit | ✅ | Every expanded-K code and doc commit, Phase 1 (`c19695fd7b5`) and Phase 2 (`69f2c38e874`) through `3124e893b07`, reviewed by both; findings fixed or answered in `reviews/`. The final commit only adds the last review's log |

## Accuracy (E2)

PCC vs the CPU reference, from the harness's own check:

| Scenario | Absorbed | Expanded |
|---|---|---|
| K2.7 rot-multislab, scalar, `sp_only` | 0.99613 | 0.99616 |
| K2.7 rot_partial, metadata, `tp_sharded` | 0.99660 | 0.99661 |
| K3 rot-multislab, scalar, `sp_only` (k256) | 0.99584 | 0.99586 |
| K3 rot-multislab, scalar, `sp_only` (q320 / 2 bands) | 0.99584 | 0.99587 |

## Performance (E3)

Traced replays of a steady-state layer (layer 1 of 3), the parent plan's X5 method, `sp_only` layout. One run each.

| Point | Absorbed (unrolled-V) | Expanded K, Phase 1 | Expanded K, Phase 2 | Phase 2 vs absorbed | main `ring_mla` |
|---|---|---|---|---|---|
| K3 50k | 7.582 ms | 6.630 ms | 6.014 ms | −20.7% | 10.33 ms |
| K3 512k | 61.779 ms | 52.745 ms | 47.063 ms | −23.8% | 89.61 ms |
| K2.7 50k | 5.467 ms | 4.689 ms | 4.227 ms | −22.7% | 6.83 ms |
| K2.7 256k | 22.127 ms | 18.519 ms | 16.579 ms | −25.1% | 29.46 ms |

**Phase 2, `tp_sharded` with the prefetch** (overhead vs its own `sp_only`):

| Point | Time | Overhead |
|---|---|---|
| K3 50k | 6.054 ms | +0.66% |
| K3 512k | 46.851 ms | −0.45% |
| K2.7 50k | 4.210 ms | −0.39% |
| K2.7 256k | 16.404 ms | −1.06% |

Against main's `ring_mla`, the K3 512k forward is now 47% faster.

**`tp_sharded`** with the prefetch: K3 512k 52.822 ms, K2.7 256k 18.329 ms.

**Notes on these numbers:**
- **Not yet expanded once per device.** In Phase 1 each core expands its head's whole K chunk. With K-split bands,
  each band's 2 cores per head both do it, so expansion is still 2× redundant.
- **K3's schedule.** K3 at its absorbed schedule (q160, 4 cores per head, k256) was +11.5% slower (8.457 ms at 50k).
  The q320 / 2-band schedule is what made Phase 1 a gain for K3.

## Expansion throughput (stop criterion)

Realtime profiler, SDPA program, steady-state K2.7 256k layer (`sp_only`), all on the same q320 / k256 / 3-band
schedule:

| Variant | SDPA device time |
|---|---|
| Absorbed | 19.705 ms |
| Expanded | 16.543 ms |
| Expanded, 1 of every 16 expansion matmuls kept (a local kernel edit, not committed) | 12.744 ms |

- **The expansion takes ≈ 4.05 ms** (16/15 × 3.80 ms), about 24% of the expanded SDPA.
- **Throughput, not resolved:** it is 16.8M MACs per core per K chunk. Going from 576 to 192 wide saves 31.5M QK
  MACs, worth 19.71 − 12.49 = 7.2 ms. That puts the expansion at ≲ 95% of QK's MAC throughput, but only as an upper
  bound: the 7.2 ms also includes the narrower Q read and other non-MAC savings.
  - **Lower end:** a peak-based estimate (HiFi2, about 2 k MACs per cycle per core, 200–340 K chunks per core) puts
    the expansion at roughly 30–50% of FPU peak, against an assumed ≈ 70% for QK. That ratio is roughly 0.43–0.71.
    (The measurement below puts QK at about the nominal HiFi2 rate.)
  - **Superseded by the Phase 2 measurement below** (0.74–0.81 of QK's rate). The peak-based estimate was too
    low: it assumed about 2 k MACs per cycle at HiFi2, which is the LoFi rate.
  - **So the 60% floor was not established in Phase 1** (waived, below). A blocked (multi-tile) expansion matmul
    is the lever if throughput still limits E3 after Phase 2.
- **What's left:** the expansion is still 2× redundant (both Q-chunk cores of a head in a band expand it). Phase 2's
  exchange would save ≈ 2 ms more here, for ≈ −26% SDPA vs absorbed.

**Phase 1's throughput stop criterion: moot.**
- The author waived it (2026-10-18), pulling E3's measurement to the end of Phase 2.
- E3 is met there at all four points (−20.7% to −25.1%), so the question the 60% floor stood for (whether the
  expansion pays enough to clear E3) is answered directly.

## Expansion utilisation (Phase 2, measured)

Measured at the author's request, as input for optimisation, not as a gate.

**Method:**
- A probe patch, not committed (`expk_util_probe.patch` in the work dir), wraps the expansion's matmul + pack loop
  and the QK `blocked_matmul_and_pack` call in device-profiler sum zones (`DeviceZoneScopedSumN1/N2`).
- It counts the tile matmuls in each and emits the counts at kernel exit.
- The run is one eager `sp_only` forward with `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1`: one layer, 5k
  chunk, Phase 2 schedule (pairs).
- The figures are math-thread (TRISC 1) cycles per 32×32×32 tile matmul, over all 3072 SDPA cores of the mesh,
  with matmul + pack inside each zone.

| Point | Expansion | QK | Expansion rate vs QK | Expansion share of SDPA kernel | QK share | SDPA kernel |
|---|---|---|---|---|---|---|
| K3 512k | 39.5 cycles (p10–p90 39.1–40.1) | 32.1 | 0.81 | 18.2% | 27.7% | 42.48 ms |
| K2.7 256k | 43.8 cycles (42.5–45.2) | 32.3 | 0.74 | 19.7% | 27.4% | 14.46 ms |

**What it says:**
- **QK runs at about the nominal HiFi2 rate.** 32 cycles per tile matches two fidelity passes of 16 cycles, taking
  BH's nominal ≈ 2 k LoFi MACs per cycle. That is an inference from the nominal rate, not a measured peak. The QK
  zone wraps each subblock call, so profiler overhead is inside the 32.1.
- **The expansion runs at 74–81% of QK's rate.** It would have cleared the waived 60% floor.
- **Likely cause of the extra 7–12 cycles per tile (a hypothesis):** the single-tile structure.
  - One output tile per `tile_regs_acquire`.
  - Both operands unpacked again for every tile matmul.
  - One or two single-tile packs per 16 matmuls.
  - Not separated: the unpack thread reported no expansion sum, and both math and pack zones include
    synchronisation waits (`tile_regs_acquire` / `tile_regs_wait`). So pack backpressure is not ruled out. The pack
    thread's 37–41 cycles is per tile matmul in the same zone, mostly waiting.
- **Ceiling for a blocked expansion matmul**, at QK's rate: 18.2% × (1 − 32.1/39.5) = 3.4% of the SDPA kernel at K3
  512k (≈ 1.4 ms), 19.7% × (1 − 32.3/43.8) = 5.2% at K2.7 256k (≈ 0.75 ms).
- **Larger levers**, since the expansion's MAC count is now the cost (18–20% of the SDPA kernel today, ≈ 15% at
  QK's rate):
  - The expansion matmul at LoFi only. LoFi for all SDPA matmuls cut the program by 17–18%. It would need E1/E2 again.
  - Groups of 4 (each core expands 1 of 4 no-PE rows). This halves the per-core expansion again (≈ 7–10% of the
    kernel at best), on schedules with 4 Q-chunk cores per head. But the kernel supports pairs only, the wider
    exchange is not costed, and K3's 4-core schedule was 11.5% slower in Phase 1.
- **Not covered:** the k_pe and peer-row copies, and stalls outside the zones. The kernel times here come from one
  eager, profiled layer (42.48 / 14.46 ms); E3's 42.35 / 14.74 ms are realtime-profiler times from traced
  steady-state runs.
- **Reproduce:** `probes/` holds the probe patch, the harness (its header has the exact commands; one case per run)
  and the analysis script (`expk_util_analyze.py profile_log_device.csv`).

## Next steps

1. Measure the ring-vs-compute split (per-RISC wait time) for E3's profiler criterion.
2. E1: an fp32 reference for rotated starts and short → long → short sequences (today checked only against absorbed).
3. E4: trace the prefetch-off case too.
4. Phase 3: the chunk retune is done and `_EXPANDED_K_SDPA_CHUNKS` is unchanged. Steady-state traced `sp_only`
   forward, one run each (the first row of each is the Phase 2 column's configuration, a separate run):
   - K3 512k: q320 / k256 / 2 splits 47.09 ms; k128 / 2 splits 53.74; k256 / 1 split 82.73.
   - K2.7 256k: q320 / k256 / 3 splits 16.55 ms; k128 / 3 splits 18.72; k256 / 2 splits 22.61.

   Model-level runs with weights are done (E2): K3 golden trace, K3 chunked prefill L5 and L24, K3 traced
   transformer L24, K2.7 transformer. Still to do: head groups of 4 (other schedules); K3 `test_transformer_depth`
   L24 (CI).
5. Phase 4: default on for Kimi (E5 regression set), docs.
