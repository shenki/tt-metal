# Unrolled-V MLA on the TP-sharded KV cache: status

Milestone: Phase 2 of [PLAN.md](PLAN.md) (serial TP latent gather, no overlap). Branch
`pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-17. Everything was measured on one Blackhole Galaxy 8×4
(TorusXY, high-power host).

## What it does

**Starting point**
- **Unrolled-V** (branch `pjosipovic/ring-joint-sdpa-unrolled-v-mcast`, commit `7d302b0c4e1`): expands V per head
  ahead of `ring_joint` SDPA. On this branch, measured as traced replays with the TP-replicated cache (`sp_only`), K3
  MLA forward is 27–31% faster than main's `ring_mla` and K2.7 20–25% faster (table below).
- **PR #59531** shards the dense KV cache over SP×TP: each chip keeps 1/32 of the cache instead of 1/8, so 4× less KV
  DRAM. As written, the two features clash: a chip can expand V only for rows it holds, and with the sharded cache
  3 of every 4 rows of its SP slab sit on TP neighbours.

**This branch makes them work together:**
1. After each layer's cache write, a TP `high_bw_all_gather` rebuilds this SP row's latent from the 4 TP stripes into
   one shared single-slot buffer, in the old TP-replicated layout.
2. The V expansion and `ring_joint` then run unchanged on that buffer.
3. The gather reads the ND-sharded cache directly, with no reshape, using a new `outer_block_rows` option. Each
   160-row region is gathered as its own rank-interleaved block.
4. On the trace-safe path, the slot and the live extent come from the metadata tensors.

## Commits (on top of `7d302b0c4e1`)

| Commit | What | Reviews |
|---|---|---|
| `f89b4fcd2ad` | Merge PR #59531 into unrolled-V | [log](reviews/f89b4fcd2ad.md) |
| `c7d47299a9e` | `high_bw_all_gather`: `outer_block_rows` / `gathered_outer_blocks` | [log](reviews/c7d47299a9e.md) |
| `321daf1d6df` | MLA: unrolled-V on the TP-deduped cache; bitwise and perf tests | [log](reviews/321daf1d6df.md) |
| `745a37d83fa` | Op review fixes: page-unit outer count (fixes a width-gather overrun), min-pages guard, tests | [log](reviews/745a37d83fa.md) |
| `00ef3010018` | MLA review fixes: slot allocated at init, poisoned bitwise test, DRAM test | [log](reviews/00ef3010018.md) |
| `25b664d628b` | Round-3 review fixes: X4 trace-replay test, replay poisoning, perf method and gates, op tests; docs | [log](reviews/25b664d628b.md) |
| `06690863f87` | Round-4 review fixes: X4 cache_layer_idx + sp_only equality + rotated starts, cache-hit proof, traced main baseline, docs | [log](reviews/06690863f87.md) |
| `5be86f27b90` | Round-5 review fixes: X4 cache comparison and layout alternation, gates, main-baseline patch, docs | [log](reviews/5be86f27b90.md) |
| `42fc9f477b3` | Round-6 review fixes: X4 compiles before capture, eager tp_sharded after sp_only, outputs freed (passes with trace allocation tracking), slot-extent check; K3 512k gate 2.2%; baseline patch checks replay | [log](reviews/42fc9f477b3.md) |
| `f99bc175844` | Round-7 review fixes: no-compile assertion in X4, extent-check wording, baseline replay output zeroed first | [log](reviews/f99bc175844.md) |
| `d4d22260987` | Round-8 review fixes: X4 program-cache snapshot taken before capture; baseline patch compiles its zeroing before capture (passes with allocation tracking) | [log](reviews/d4d22260987.md) |
| `239b41cb8e5` | Round-9 review fix: baseline patch hands the replay's output on after the timing branch | [log](reviews/239b41cb8e5.md) |
| *(last)* | Review log for `239b41cb8e5` (docs only) | — |

Each code commit (up to `239b41cb8e5`) was reviewed by Claude Code and Codex (gpt-6-astra). Every finding is fixed, or
answered in its log.

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| X1 | `tp_shard_kv` runs unrolled-V, no `ring_mla` | ✅ | `_chunked_attn` / `_tp_gather_latent` |
| X2 | `tp_sharded` bitwise equal to `sp_only` | ✅ | `test_mla_chunked_prefill_tp_sharded_matches_sp_only`: 58 passed on functional + golden inputs, plus 44 passed on CPU-reference inputs, with the shared slot poisoned before every forward. Exclusions: no K2.7 golden trace exists (2 skips), and the deep/production scenarios have no CPU-reference case (CPU MLA too slow); both run on functional inputs. The PR's `test_mla_chunked_prefill` `tp_sharded` matrix: 140 passed, 0 failed. |
| X3 | No regressions | ✅ | 53 Galaxy MLA tests pass on `sp_only`, with PCCs identical to before the merge. K3 KDA carry (`kimi_k3/test_chunked_prefill.py`): 4/4. |
| X4 | Captured multi-layer, multi-user trace | ✅ | `test_mla_tp_sharded_trace_replay_multi_layer` (K2.7, K3): 2 layers (each on its own cache slot), 2 users, one capture replayed over 7 steps with user and prefix changed in place (long → short → long, rotated starts on a chip boundary), slot poisoned before every replay. Traced replay == `tp_sharded` eager == `sp_only` eager, bitwise, for both the outputs and the un-striped caches, with every (user, layer) slot written. Layouts alternate: the capture stays alive through the `sp_only` pass, then eager `tp_sharded` forwards and a second replay of the same capture must still match. Every program is compiled before the capture. Rows past user 1's extent are written only in user 0's slots, which catches a transposed slot mapping. No program compiles while the trace is alive (asserted); with `TT_METAL_TRACE_ALLOC_TRACKING=1` no allocation made while it is alive is live at a replay (passes). Mid-chip rotated starts are covered by the single-layer X2 scenarios. The PR's 10-layer K2.7 transformer test passes `tp_sharded` traced and untraced. |
| X5 | Traced perf gate | ✅ | Two runs of the current method, 4/4 points each (below). Gates are the worst measured sample plus about 1 pp of measured noise, replacing the pre-measurement estimates. |
| X6 | DRAM saving | ✅ | `test_mla_tp_sharded_dram_saving`, measured per chip at exact capacities (see below) |
| X7 | Op tests | ✅ | 18 new outer-block, width-gather and validation tests pass. Other failures in `test_high_bw_all_gather.py` on this Galaxy (3 bandwidth floors, 4 fabric-init errors) happen identically on main. |
| X8 | Defaults and CI | ◐ | `sp_only` K3 50k is unchanged on the realtime profiler (7.654 / 7.660 ms vs 7.656 ms before). The criterion asks for a traced before/after, which was not measured: only the branch's traced `sp_only` (table below). **CI legs not run** (next step 1, deliberately out of this milestone). |
| X9 | Two reviews per commit | ✅ | Every code commit reviewed by both; the last round (`239b41cb8e5`) had no blocker or major. The final commit only adds that review's log |

### Performance (X5)

All numbers are **traced replays** of one 5k-chunk MLA forward on the same Galaxy: an untimed warm-up replay, then
the median of 5 replays 2 s apart (the middle three must agree within 2%). Main's `ring_mla` was timed the same way in
a main worktree (merge-base `a7a3bc8f603`).

Absolute times from run 1:

| Point | main `ring_mla` | unrolled-V `sp_only` | unrolled-V `tp_sharded` | Speed-up kept with the sharded cache |
|---|---|---|---|---|
| K3 50k | 10.329 ms | 7.542 ms (−27.0%) | 7.605 ms (−26.4%) | 98% |
| K3 512k | 89.613 ms | 61.810 ms (−31.0%) | 62.462 ms (−30.3%) | 98% |
| K2.7 50k | 6.832 ms | 5.451 ms (−20.2%) | 5.548 ms (−18.8%) | 93% |
| K2.7 256k | 29.462 ms | 22.110 ms (−25.0%) | 22.466 ms (−23.7%) | 95% |

The X5 gate on `tp_sharded` vs `sp_only`, two runs of the current method:

| Point | Run 1 | Run 2 | Gate |
|---|---|---|---|
| K3 50k | +0.83% | +1.29% | +2.5% |
| K3 512k | +1.06% | +1.16% | +2.2% |
| K2.7 50k | +1.78% | +0.81% | +3.0% |
| K2.7 256k | +1.61% | +2.01% | +3.0% |

- **Where the overhead comes from:** the TP gather runs serially ahead of the expansion. That's about 0.1 ms at 50k
  (mostly fixed latency) and 0.3–0.6 ms at 256k–512k. Phase 3 (overlap) is what removes it.
- **Gate history:** the gates were first estimated (+1.5/+1.2/+2.0/+1.5%). They are now the worst current-method
  sample plus about 1 pp, since run-to-run noise in the overhead is about ±0.5 pp at 50k and up to 0.4 pp at
  256k–512k. Earlier samples taken with older methods (first replay only, or back-to-back medians) are not used.
- **Why this method:** two effects made simpler methods misleading.
  1. Back-to-back replays slow down under sustained load: K3 512k went 61.7 → 70.5 ms over 7 replays.
  2. The first replay after a capture runs about 2% fast at 50k.
- **Two kinds of numbers:** traced replays (host wall-clock, above) and the realtime profiler (the sum of per-program
  kernel times at the slowest chip, used by `test_mla_chunked_perf_check` and for per-op breakdowns) measure different
  things, and neither is a fixed offset from the other. Measured on this Galaxy: K3 50k `sp_only` is 7.54 ms traced
  vs 7.65 ms profiler; K3 512k `sp_only` 61.8 ms traced vs 57.0 ms profiler; main K3 50k 10.33 ms traced vs 10.42 ms
  profiler; main K3 512k 89.6 ms traced vs 79.7 ms profiler. Compare only within one metric.
- **Main's numbers repeat:** a second run of main's baseline, with its replay checked against eager, gave 10.298 / 89.487 / 6.816 / 29.455 ms, within 0.3% of the table. Two more, with `TT_METAL_TRACE_ALLOC_TRACKING=1`, gave 10.341 / 89.584 / 6.854 / 29.452 ms and 10.358 / 89.507 / 6.838 / 29.465 ms.
- **Reproducing main's numbers:** apply [main_baseline_traced.patch](main_baseline_traced.patch) to a main worktree at
  `a7a3bc8f603`. It adds the same `trace_time` mode to main's `test_mla.py`, plus `test_mla_traced_baseline_scratch.py`
  for the four points. Then run that test the same way as above.

### DRAM (X6)

Per chip, per user slot, measured allocation deltas at whole-chunk capacities of 527360 rows (512k) and 266240 rows
(256k), which are runnable dense `tp_shard_kv` layouts:

| | Cache slot per layer, replicated → deduped | Shared TP latent slot | Net saving |
|---|---|---|---|
| K3 512k (24 MLA layers) | ≈41 → ≈10 MB | ≈41 MB, paid once | ≈0.69 GB |
| K2.7 256k (61 MLA layers) | ≈21 → ≈5 MB | ≈21 MB, paid once | ≈0.91 GB |

The saving repeats for every concurrent user slot. With pipeline ranks, each rank pays one slot.

## How to reproduce (any Galaxy 8×4)

```bash
source /opt/venv/bin/activate && export TT_METAL_HOME=$PWD
./build_metal.sh --release --build-tests
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_TORUS_XY_CERTIFIED=1 OMP_NUM_THREADS=$(nproc)
T=models/demos/deepseek_v3_d_p/tests/test_mla.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py -k "outer_blocks or selected_slot_width_gather"
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill_tp_sharded_matches_sp_only      # X2 (golden case needs KIMI_K3_HF_MODEL)
TT_METAL_TRACE_ALLOC_TRACKING=1 scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_trace_replay_multi_layer  # X4
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_tp_sharded_traced_perf                   # X5 (high-power host)
scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_dram_saving                           # X6
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill -k "torus-xy-8x4 and tp_sharded and (k3 or k2_7)"
```

- `KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e` enables the golden-trace case.
- The PR's transformer test needs `KIMI_K2_7_HF_MODEL`, `TT_KIMI_PREFILL_TTNN_CACHE` and `PREFILL_TRACE_DIR`, as in
  `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`.

## Next steps

1. **CI (X8):** run the Blaze prefill legs on this branch, i.e. the PR's K2.7/K3 `tp_sharded` `maxedge` legs and the
   K2.7 transformer legs. Update `K3_CHUNKED_RT_PERF_NS`: `test_mla_chunked_perf_check` still has main's 10.556 ms
   target, while this branch measures ~7.66 ms on `sp_only`.
2. **Rebase onto main once PR #59531 merges,** dropping the merge commit.
3. **Phase 3, overlap (optional, PLAN.md §4.4):** run the TP gather alongside CCL-free compute using sparse MLA's
   80/40 sub-device pattern (`tt_ccl.get_sparse_mla_overlap_resources`, `SubDeviceTraceController`). Target ≤ +0.5%
   overhead.
4. **Follow-up, expanded K on arrival (PLAN.md §6):** unabsorbed QK at 192 wide. Napkin estimate: −21% to −26%
   attention at HiFi2. This branch keeps the latent as K's wire format and the slot at 576 columns, so the step stays
   open.
5. **Upstream bits worth splitting out as separate PRs:**
   - `high_bw_all_gather` `outer_block_rows` (useful on its own);
   - the review findings on PR #59531 itself, which are not this branch's code: sparse `tp_shard_kv=False` must be
     rejected, and `zero_padded_kv_cache` with `tp_axis` has no op test.
