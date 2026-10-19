# Unrolled-V MLA on the TP-sharded KV cache: status

Milestone: Phases 2 and 3 of [PLAN.md](PLAN.md) (TP latent gather; prefetched during the previous layer's SDPA). Branch
`pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-19. Everything was measured on one Blackhole Galaxy 8×4
(TorusXY, high-power host). Expanded K, built on top of this, has its own status:
[../expanded_k/STATUS.md](../expanded_k/STATUS.md); current MLA performance numbers are there.

## What it does

**Starting point**
- **Unrolled-V** expands V per head ahead of `ring_joint` SDPA. Measured as traced replays with the TP-replicated cache
  (`sp_only`), K3 MLA forward is 27–31% faster than main's `ring_mla` and K2.7 20–25% faster, with absorbed K (table
  below).
- **PR #59531** shards the dense KV cache over SP×TP: each chip keeps 1/32 of the cache instead of 1/8, so 4× less KV
  DRAM. As written, the two features clash: a chip can expand V only for rows it holds, and with the sharded cache
  3 of every 4 rows of its SP slab sit on TP neighbours.

**This branch makes them work together:**
1. After each layer's cache write, a TP `high_bw_all_gather` rebuilds this SP row's latent from the 4 TP stripes into
   a TP latent buffer, in the old TP-replicated layout.
2. The V expansion and `ring_joint` then run unchanged on that buffer, as a single-slot cache.
3. The gather reads the ND-sharded cache directly, with no reshape, using a new `outer_block_rows` option. Each
   160-row region is gathered as its own rank-interleaved block.
4. On the trace-safe path, the slot and the live extent come from the metadata tensors.
5. **Prefetch (PLAN.md §4.4):** layer L's SDPA runs inside a sub-device manager. Alongside it, layer L+1's gather of its
   earlier chunks runs on the 6 cores and the TP links the SDPA leaves idle, into the other parity's buffer.
   - Layer L+1 then writes only its own chunk into that buffer, locally.
   - Prefetching happens only inside `tt_ccl.tp_latent_prefetch_chain()`, which the transformers open around their
     layer loops.
   - Layer 0, any forward outside a chain, and any forward whose host-side key doesn't match the prefetch gather
     serially, as in step 1.
   - On by default. `ttMLA(tp_latent_prefetch=False)` turns it off. At K2.7 50k it misses Phase 3's ≤ +0.5% (+1.18% mean
     over 5 runs, against +1.71% serial); it stays on there by the author's decision, since it is not slower than
     serial on average and removes ≈0.8–1.4 pp (means) at 256k–512k.

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| X1 | `tp_shard_kv` runs unrolled-V for Kimi, no `ring_mla` | ✅ | `_chunked_attn` / `_tp_gather_latent`; other dense models keep `ring_mla` (`mla_config.MLA_ATTENTION_CONFIG`) |
| X2 | `tp_sharded` bitwise equal to `sp_only` | ✅ | `test_mla_chunked_prefill_tp_sharded_matches_sp_only`: 58 passed on functional + golden inputs, 44 on CPU-reference inputs, the TP latent buffer poisoned before every forward. No K2.7 golden trace exists (2 skips); deep/production scenarios run on functional inputs only (CPU MLA too slow). The PR's `test_mla_chunked_prefill` `tp_sharded` matrix: 140 passed |
| X3 | No regressions | ✅ | 53 Galaxy MLA tests pass on `sp_only`, PCCs identical to before the merge. K3 KDA carry (`kimi_k3/test_chunked_prefill.py`): 4/4 |
| X4 | Captured multi-layer, multi-user trace | ✅ | `test_mla_tp_sharded_trace_replay_multi_layer` (K2.7, K3): 3 layers, 2 users, one segmented capture (`SubDeviceTraceController`, prefetch on) replayed over 7 steps with user and prefix changed in place (long → short → long, rotated starts), buffers poisoned before every replay. Traced == `tp_sharded` eager == `sp_only` eager, bitwise, outputs and un-striped caches, layouts alternated; no compile while the trace is alive; passes with `TT_METAL_TRACE_ALLOC_TRACKING=1`. `test_mla_tp_latent_prefetch_matches_sp_only` (K2.7, K3 × scalar, metadata; 3 layers, 2 users, chunk-aligned / chip-boundary / mid-chip starts, partial chunks): prefetch == serial == `sp_only` bitwise, layers 1–2 consumed their prefetch on every step; no prefetch is consumed across chains or a `set_tp_shard_kv` layout change |
| X5 | Traced perf gate | ✅ | Steady-state layer with the prefetch, two runs, 4/4 points each (below). Gates: +1.2 / +1.1 / +2.4 / +1.3%; serial gather (`tp_latent_prefetch=False`): +2.5 / +2.2 / +3.0 / +3.0% |
| X6 | DRAM saving | ✅ | `test_mla_tp_sharded_dram_saving`, measured per chip at exact capacities, net of both TP latent buffers: 0.645 GB (K3 512k), 0.886 GB (K2.7 256k) per user slot |
| X7 | Op tests | ✅ | The new outer-block and validation tests pass. Other failures in `test_high_bw_all_gather.py` on this Galaxy (3 bandwidth floors, 4 fabric-init errors) happen identically on main |
| X8 | Defaults and CI | ◐ | `tp_sharded` is the Kimi default (PR #59531), so the TP latent gather and its prefetch are the production path; `PREFILL_TP_SHARD_KV=0` rolls back. `sp_only` K3 50k is unchanged on the realtime profiler (7.654 / 7.660 ms vs 7.656 ms before). A traced before/after was not measured. **CI legs not run** (next step 1) |

Every change was reviewed by Claude Code and Codex.

### Performance (X5)

These tables are from **absorbed K** (unrolled V only), before expanded K; they set the X5 gates, which compare
`tp_sharded` against `sp_only` on the same build. Current numbers with expanded K:
[../expanded_k/STATUS.md](../expanded_k/STATUS.md).

All numbers are **traced replays** of one 5k-chunk MLA forward on the same Galaxy: an untimed warm-up replay, then
the median of 5 replays 2 s apart (the middle three must agree within 2%). Main's `ring_mla` was measured with an
equivalent traced harness at `a7a3bc8f603` (the merge-base); a second run, with the replay checked against eager, was
within 0.3%.

Phase 2 (serial gather, single-layer forward):

| Point | main `ring_mla` | unrolled-V `sp_only` | unrolled-V `tp_sharded` | Speed-up kept with the sharded cache |
|---|---|---|---|---|
| K3 50k | 10.329 ms | 7.542 ms (−27.0%) | 7.605 ms (−26.4%) | 98% |
| K3 512k | 89.613 ms | 61.810 ms (−31.0%) | 62.462 ms (−30.3%) | 98% |
| K2.7 50k | 6.832 ms | 5.451 ms (−20.2%) | 5.548 ms (−18.8%) | 93% |
| K2.7 256k | 29.462 ms | 22.110 ms (−25.0%) | 22.466 ms (−23.7%) | 95% |

**Phase 3 (prefetch), the X5 gate.** The timed forward is a steady-state layer: layer 1 of 3, whose latent was
prefetched during layer 0's SDPA and which prefetches layer 2's during its own (`trace_time` with
`num_cache_layers=3`). Overhead vs `sp_only` on the same layer:

| Point | `sp_only` | Serial, run 1 / 2 | Prefetch, run 1 / 2 | Gate |
|---|---|---|---|---|
| K3 50k | 7.582 ms | +0.19% / +0.62% | +0.19% / +0.06% | +1.2% |
| K3 512k | 61.779 ms | +0.92% / +1.08% | +0.14% / +0.07% | +1.1% |
| K2.7 50k | 5.467 ms | +1.52% / +1.32% | +1.15% / +1.29% | +2.4% |
| K2.7 256k | 22.127 ms | +1.73% / +1.55% | +0.31% / +0.11% | +1.3% |

- **Gates** are the worst measured sample plus about 1 pp: run-to-run noise in the overhead is about ±0.5 pp at 50k and
  up to 0.4 pp at 256k–512k. The serial gather (`tp_latent_prefetch=False`) keeps Phase 2's gates:
  +2.5 / +2.2 / +3.0 / +3.0%.
- **Where the overhead comes from:**
  - **Serial:** the TP gather runs ahead of the expansion. It costs about 0.1 ms at 50k, mostly fixed latency, and
    0.3–0.6 ms at 256k–512k.
  - **Prefetch:** hides nearly all of the gather at 256k–512k.
  - **K2.7 50k:** over five runs, the means are +1.18% prefetch and +1.71% serial. There the gather is fixed latency,
    and running it concurrently doesn't hide it.
  - **What it isn't:** the sub-device load/clear. A region without the gather measures 5.420 ms against 5.460 ms
    `sp_only`. Moving the gather to the SDPA's idle row 9, or using one link, changes nothing (+1.45%).
- **Device profile, K2.7 50k layer 1 (prefetch vs `sp_only`):** the local chunk write adds 9 µs; the SDPA is 13 µs
  slower; the o_proj reduce-scatter after the region takes 348 vs 255 µs (chip skew).
- **Why this method:** back-to-back replays slow down under sustained load (K3 512k went 61.7 → 70.5 ms over 7
  replays), and the first replay after a capture runs about 2% fast at 50k.
- **Two kinds of numbers:** traced replays (host wall-clock, above) and the realtime profiler (the sum of per-program
  kernel times at the slowest chip, used by `test_mla_chunked_perf_check` and for per-op breakdowns) measure different
  things, and neither is a fixed offset from the other. K3 50k `sp_only` is 7.54 ms traced vs 7.65 ms profiler; K3
  512k `sp_only` 61.8 vs 57.0 ms; main K3 50k 10.33 vs 10.42 ms; main K3 512k 89.6 vs 79.7 ms. Compare only within one
  metric.

### DRAM (X6)

Per chip, per user slot, measured allocation deltas at whole-chunk capacities of 527360 rows (512k) and 266240 rows
(256k), which are runnable dense `tp_shard_kv` layouts. With the prefetch on (the default) there are two TP latent
buffers, one per layer parity; with it off, one.

| | Cache slot per layer, replicated → deduped | TP latent buffers (2), paid once | Net saving (measured) |
|---|---|---|---|
| K3 512k (24 MLA layers) | ≈41 → ≈10 MB | ≈82 MB | 0.645 GB |
| K2.7 256k (61 MLA layers) | ≈21 → ≈5 MB | ≈42 MB | 0.886 GB |

The saving repeats for every concurrent user slot. With pipeline ranks, each rank pays the buffers once.

## SP ring K+V gather: where the limit is

Measured 2026-10-18 with expanded K, KV 512k, TorusXY, `sp_only`.
- **Ethernet floor.** Each chip receives 7 of 8 shards at 100 GB/s ingress (4 links × 25 GB/s). Per token: bfp8 K
  latent (576) + per-head V (heads × 128), 1,088 B per 1,024-value tile: 3,876 B for K3, 2,788 B for K2.7. Floor:
  17.8 ms (K3), 12.8 ms (K2.7). Measured gather alone: 22.6 / 16.3 ms, 78–79% of the floor. With unrolled V the fused
  gather already splits the diametric shard between the two ring directions (`ring_joint_sdpa_program_factory.cpp`,
  split forwarding; off for latent V), so direction imbalance is not the gap.
- **Chunk size decides the bound, not KV length.** Both compute and the gather grow linearly with KV length.
  - At 5k chunks the SDPA is compute-bound: the gather alone is 53–57% of the SDPA program at HiFi2, 65–69% at LoFi.
  - At 2k it is gather-bound: 89–94%, and LoFi saves only 3–5%.
  - Break-even is about 2.7–3.5k tokens per chunk (HiFi2 to LoFi, K3 to K2.7).
- **Rejected:** ring CCL workers on logical row 0 (next to the ethernet row) and/or the gather's DRAM reads and fabric
  push swapped between NoCs. Today's layout (column 11, reads on NoC1, push on NoC0) was fastest at every point: row 0
  costs 11–12% at 2k and 3–4% at 5k; the NoC swap 23–27% at 2k, because the gather's DRAM reads compete with the SDPA
  reader on NoC0. Expanding V on arrival is rejected too ([../expanded_k/STATUS.md](../expanded_k/STATUS.md),
  decisions).
- **Levers left:**
  - The ~22% gap to the floor: fabric packet payload vs the 1,088 B pages (a 7,168 B packet holds 6 pages, 14,400 B
    holds 13; not measured), and NoC contention with the SDPA.
  - More links: only 2 of the 4 ethernet channels per direction have routing planes.

## How to reproduce (any Galaxy 8×4)

```bash
source /opt/venv/bin/activate && export TT_METAL_HOME=$PWD
./build_metal.sh --release --build-tests
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_TORUS_XY_CERTIFIED=1 OMP_NUM_THREADS=$(nproc)
T=models/demos/deepseek_v3_d_p/tests/test_mla.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py -k "outer_blocks"
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill_tp_sharded_matches_sp_only      # X2 (golden case needs KIMI_K3_HF_MODEL)
TT_METAL_TRACE_ALLOC_TRACKING=1 scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_trace_replay_multi_layer  # X4
scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_latent_prefetch_matches_sp_only            # prefetch, bitwise
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_tp_sharded_traced_perf                   # X5 (high-power host)
scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_dram_saving                           # X6
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill -k "torus-xy-8x4 and tp_sharded and (k3 or k2_7)"
```

- `KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e` enables the golden-trace case.
- The PR's transformer test needs `KIMI_K2_7_HF_MODEL`, `TT_KIMI_PREFILL_TTNN_CACHE` and `PREFILL_TRACE_DIR`, as in
  `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`.
- A fresh worktree needs `git submodule update --init --recursive` before the first build. Device tests always go
  through `scripts/run_safe_pytest.sh` (exit codes: 0 pass, 1 fail, 2 hang).

## Next steps

1. **CI (X8):** run the Blaze prefill legs on this branch, i.e. the PR's K2.7/K3 `tp_sharded` `maxedge` legs and the
   K2.7 transformer legs. The Kimi perf gates are re-baselined for TorusXY with expanded K on
   ([../expanded_k/STATUS.md](../expanded_k/STATUS.md), E5).
2. **Rebase onto main once PR #59531 merges,** dropping the merge commit.
3. **Upstream bits worth splitting out as separate PRs:**
   - `high_bw_all_gather` `outer_block_rows` (useful on its own);
   - the review findings on PR #59531 itself, which are not this branch's code: sparse `tp_shard_kv=False` must be
     rejected, and `zero_padded_kv_cache` with `tp_axis` has no op test.
