# Plan: unrolled-V MLA on the fully sharded KV cache (Galaxy 8×4)

Status: **rev 4, single design**, after two review rounds by Claude Code and Codex (gpt-6-astra) · Author: Pavle Josipovic (with Claude) · 2026-10-17
Branch: `pjosipovic/ring-joint-sdpa-unrolled-v-mcast` (HEAD `7d302b0c4e1`) · Builds on PR #59531 (local ref `pr-59531`)
Rejected alternatives and the explainer are in the artifact "Unrolled V vs Sharded KV". §8 lists what changed from
rev 3.

## 0. Exit criteria

**Done (ready to merge) when every item holds**, on the Galaxy 8×4 TorusXY, with the release build of the branch:

| # | Criterion | How it's checked |
|---|---|---|
| X1 | `tp_shard_kv=True` runs unrolled-V for K3 and K2.7. No dense `ring_mla` call, and the dense `tp_shard_kv` assert is removed. | Code; the X2 tests run this path |
| X2 | `tp_sharded` output is **bitwise equal** to `sp_only` on the same seed for every 8×4 `test_mla_chunked_prefill` scenario × {cpu, golden} × {scalar, metadata}, for K3 and K2.7 | New paired test, all green |
| X3 | No regressions: the 53 Galaxy MLA tests run in this session (K3/K2.7/Mistral4 chunked, `test_mistral_small4_mla`, llama4) pass on `sp_only`. The PR's KV-table tests and the K3 KDA-carry test pass on `tp_sharded`. | `run_safe_pytest.sh --run-all` |
| X4 | A captured multi-layer, multi-user device trace replays bitwise equal to eager, with user and prefix changed between replays, including long → short → long | New trace test |
| X5 | Traced-replay forward time, `tp_sharded` vs `sp_only`, serial gather: ≤ +2.5% (K3 50k), ≤ +2.2% (K3 512k), ≤ +3.0% (K2.7 50k), ≤ +3.0% (K2.7 256k). These replace the pre-measurement estimates (+1.5/+1.2/+2.0/+1.5%). They are the worst current-method sample plus about 1 pp of measured run-to-run noise; the 2026-10-17 runs measured +0.83/+1.29%, +1.06/+1.16%, +1.78/+0.81% and +1.61/+2.01%. Each layout is the median of 5 traced replays after an untimed warm-up, 2 s apart; the middle three must agree within 2%. The overlap phase (§4.4) is what removes the ~1–2%. | `test_mla_chunked_tp_sharded_traced_perf` (§5.4) |
| X6 | DRAM saving per user slot, measured as an allocation delta: ≥ 0.60 GB (K3 512k), ≥ 0.85 GB (K2.7 256k) | Memory report in the perf test |
| X7 | `high_bw_all_gather`: the new outer-block (E1) tests and all existing tests pass (failures that also occur on main, on this machine, excepted). Device times at the four sizes are recorded. | `test_high_bw_all_gather.py` |
| X8 | Default unchanged: `sp_only` traced perf within ±1% of before the change; `PREFILL_TP_SHARD_KV` off by default; the PR's `tp_sharded` CI legs green | CI run |

| X9 | Every commit on the branch has been reviewed by **both** Claude Code and Codex (gpt-6-astra). Every finding is fixed, or answered in the commit's review notes. | Review log (below) |

**Review rule, per commit:**
- Before a commit counts as done, two reviews run on it: a fresh Claude Code reviewer agent, and Codex
  `codex exec -m gpt-6-astra -c model_reasoning_effort="high"`.
- Both get the commit's diff (`git show <sha>`), this plan, and read access to the repo. Codex runs in a throwaway
  worktree at that commit, with stdin closed; its sandbox can't start on this machine. Both are told to stay read-only:
  no edits, builds or device runs.
- Blockers and majors are fixed in a follow-up commit, which is itself reviewed, before the next phase starts. Minors
  are fixed or answered.
- The review log is `reviews/<sha>.md` next to the plan: both reports, plus what was done about each finding.

**Phase exits:**
- **Phase 0:**
  - the gather's device time is measured at the four sizes, for both layouts (bank-owned, and the E1 interleaved
    view);
  - X5's gates are either confirmed by projection or revised with the numbers. If the projection misses them, Phase 3
    is scheduled right after Phase 2.
- **Phase 1:** the branch is rebased onto #59531. With `tp_shard_kv=False`, X3 passes, and output is bitwise equal to
  the pre-rebase branch.
- **Phase 2:** X1–X8 all hold. This is the merge point.
- **Phase 3 (optional):** traced-replay overhead ≤ +0.5% at all four points, with X2–X4 and X8 still holding.
  Otherwise the overlap isn't merged.

**Stop and re-plan if:**
- any of X2 or X4 fails for a reason other than a bug;
- or E1's interleaved gather falls below 50% of the bank-owned bandwidth and the bank-owned fallback doesn't recover it.

## 1. Goals

### Background
- **PR #59531** shards the dense MLA KVPE cache block-cyclically over SP×TP (`PREFILL_TP_SHARD_KV=1`).
  - Chip (s,t) keeps region t (160 rows) of each 640-row SP slab: 1/32 of the cache, 4× less KV DRAM per chip per
    layer.
  - It reads the cache with split-KV `ring_mla`.
- **This branch** replaces dense chunked `ring_mla` with unrolled-V:
  - `_expand_v` computes V = latent × W_UV for the chip's 24 (K3) / 16 (K2.7) heads with `minimal_matmul`;
  - `ring_joint_scaled_dot_product_attention` gathers K and V along the SP column, overlapped with the math.
- **Measured on the Galaxy** against main: K3 MLA forward −25% (50k) / −28% (512k); K2.7 −19% / −23% (50k / 256k).
- **Why they clash:** V is per head group. A chip can expand V only for rows it holds, using its own heads' W_UV. With
  the sharded cache, 3 of every 4 rows of its SP slab are on row neighbours that hold other heads' weights.

### Goals (all measured on the Galaxy 8×4 TorusXY)

**G1. Both features on together.** Dense chunked MLA runs unrolled-V with `tp_shard_kv=True`, for K3 and K2.7. No
fallback to `ring_mla`.

**G2. The DRAM saving survives.** Measured as allocation deltas against `sp_only`, at 1 user:
- The persistent cache is 1/(sp·tp) of the rows.
- The one added buffer is the shared temporary latent slot: 40.7 MB at K3 512k-capacity, 20.8 MB at K2.7 256k. It's
  allocated once for all layers and users.
- Expected net saving: K3 ≈ 0.69 GB per chip (24 MLA layers); K2.7 ≈ 0.93 GB (61 layers).

**G3. The speed-up survives.** The metric is the **host wall-clock of traced replays of the MLA forward**, adapted
from PR #60928 (Kimi MoE perf gate):
1. an untimed eager pass to compile;
2. capture;
3. one untimed warm-up replay;
4. five timed replays, 2 s apart, each ending in `synchronize_device`. The gate uses their median, and the middle three
   must agree within 2%.

Why not a single first replay, as the MoE gate does: here the first replay after a capture runs about 2% fast at 50k,
and back-to-back replays slow down under sustained load (+14% over 7 at K3 512k).

The metric includes overlap and gaps between programs, and excludes host dispatch. The realtime profiler's per-program
sum is kept only as the per-op breakdown. Compare `tp_sharded` against `sp_only` on the same build, and main's
`ring_mla` timed the same way.

| | K3 50k | K3 512k | K2.7 50k | K2.7 256k |
|---|---|---|---|---|
| main `ring_mla`, traced (HiFi2) | 10.33 ms | 89.61 ms | 6.83 ms | 29.46 ms |
| `sp_only`, traced (HiFi2) | 7.54 ms | 61.81 ms | 5.45 ms | 22.11 ms |
| Gate, serial gather (§2), set from measurement (X5) | ≤ +2.5% | ≤ +2.2% | ≤ +3.0% | ≤ +3.0% |
| Gate, if the fused step (§4.4) is needed | ≤ +0.5% | ≤ +0.5% | ≤ +0.5% | ≤ +0.5% |

The serial gates were first estimated from the gather's bandwidth (+1.2% to +2.0%), then set from traced runs of the
current method (X5). The serial gather costs about 0.1 ms at 50k (mostly fixed latency) and 0.3–0.6 ms at 256k–512k.
§4.4 is what removes it.

**G4. Identical results.** The temporary slot holds exactly the bytes the cache held before the PR, and everything
downstream is unchanged. So `tp_sharded` output must be **bit-identical** to `sp_only` on the same seed, across the
full `test_mla_chunked_prefill` 8×4 matrix (cpu and golden reference, scalar and metadata, rotated and overflow
scenarios, multi-user), plus the K3 KDA-state carry. Determinism repeats are bitwise equal.

**G5. Trace-safe and opt-in.**
- A captured multi-layer, multi-user device trace replays correctly when the user slot and prefix change. Each layer is
  its own captured invocation, with a fixed layer index.
- `tp_shard_kv=False` stays byte-for-byte on today's path.
- The feature stays behind `PREFILL_TP_SHARD_KV`, off by default.

### Non-goals
- The sparse (DSA) path and Mistral-Small-4 (tp = 1).
- LoFi SDPA (independent; −25% to −36% on top).
- Gather-routing changes inside `ring_joint`.
- Expanding K on arrival: a separate follow-up, kept open by §6.
- Overlapping the gather, until Phase 2 lands (§4.4).

## 2. The design

Per layer and chunk, on chip (s,t), all on one command queue, in order:

```
q/kv projections ── update_padded_kv_cache (PR #59531, writes this chip's region) ── TP latent gather ── expand V(t) ── ring_joint SDPA
                                                                                    high_bw_all_gather    unchanged      unchanged
                                                                                    into the temp slot,   reads temp     K = temp slot
                                                                                    SP-local order        slot
```

1. **Temporary latent slot.** Allocated once in `tt_ccl`, **zero-filled**, shared by every layer and user.
   - Shape `[1, 1, cap_sp, 576]` bfp8, where `cap_sp = tp × the deduplicated cache's local capacity` (the pre-PR local
     capacity).
   - Viewed as `[1, nblk, 160, 576]`, with `nblk = cap_sp/160`.
   - It holds this SP row's latent in **SP-local order**: global row `c·5120 + s·640 + t′·160 + off` sits at local row
     `c·640 + t′·160 + off`. That's exactly the pre-PR cache layout.
   - All 576 columns are kept (§6).
2. **TP latent gather**, after the cache write: `high_bw_all_gather(cluster_axis=TP, outer_block_rows=160)`.
   - Source: the deduplicated cache itself, `[slots, 1, rows, 576]` and ND-sharded. There is no reshape: a reshape
     of the ND-sharded cache would be a copy.
   - The op gathers each 160-row region as its own block, interleaving the 4 ranks per chunk into SP-local order.
   - The slot comes from `input_batch_index_tensor` (user id) + `batch_slot_num_layers` / `batch_slot_layer_idx`
     (host runtime arguments, fixed per captured layer invocation).
   - The extent is every live block. On the metadata path: `gathered_prefix_tensor = metadata[1]`,
     `gathered_slab_global = 5120`. On the scalar path: `gathered_outer_blocks = min(ceil((P + 5120)/5120), nblk)`.
   - It covers rotated chunks (2 blocks) and clamps to capacity. It's never empty: chunk 0 gathers 1 block.
3. **Expand V.** `_expand_v` reads the temporary slot:
   - `MinimalMatmulDynamicRows` without a slot tensor, `num_layers=1`, `layer_idx=0`;
   - the same row formula as today: `min(round_up(ceil(P/5120)·640 + 640, k_chunk), cap_sp)`.
4. **SDPA.** `ring_joint` takes K from the temporary slot.
   - Metadata path: a persistent one-element uint32 row-major DRAM **zero** tensor as `slot_id`, the original
     `kv_actual_isl_tensor`, `kv_cache_num_layers=1`, `kv_cache_layer_idx=0`, with no scalar slot or ISL arguments.
     Validation accepts this (`ring_joint_sdpa_device_operation.cpp:224-292`).
   - Scalar path: `kv_cache_batch_idx=0`.
   - The SP-ring gather, masks and configs are unchanged.

**Three extents, kept apart:**
- **Valid tokens:** masked by the SDPA through `logical_n` / `kv_actual`.
- **Gathered rows:** step 2's extent.
- **Rows read:** `_expand_v` rounds up to the SDPA K chunk (512 for K3, 256 for K2.7), and the SDPA reads whole K
  chunks. So the expansion and SDPA can read up to one K chunk past the gathered extent.

Those rows only ever hold zeros (from allocation) or latent gathered for an earlier layer or user. Both are finite, and
the SDPA masks them. That's the same contract as today's cache (`_expand_v` docstring, `mla.py:1167-1172`).

**Ordering.** Everything is on one queue in program order. So the next layer's gather into the temporary slot starts
only after this layer's `ring_joint`, the last reader of the slot, has finished. There's no sub-device and no event.

**Why this design** (measured on this Galaxy):
- **The attention is compute-bound.** The fused K+V ring gather alone takes 43–48% of the HiFi2 SDPA time and 67–74%
  of the LoFi time. So only work in front of the SDPA shows up.
- **The TP latent gather is that work, and it's small.** Each chip receives 3/4 of its SP row's latent: 3.2 MB at 50k,
  30.6 MB at K3 512k. `high_bw_all_gather` on the TP ring measured 57 GB/s at 512k and 67 GB/s at 1M, bank-owned
  layout. The interleaved view has to be re-measured (Phase 0).
- **Every alternative is slower on this hardware** (explainer): shipping V over TP +7–11%, expanding V on arrival
  +13–21%, routing K over the full mesh no gain.

## 3. Op work in `high_bw_all_gather` (small)

`ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/` already has `cluster_axis=1` as a TorusXY ring,
persistent output, `input_batch_index_tensor` slot selection, and the `gathered_prefix_tensor` /
`gathered_slab_global` extent. Two changes:

**E1. `outer_block_rows`: gather in rank-interleaved blocks, with no reshape.** (Implemented.)
- The KV cache is **ND-sharded DRAM** (32-row shards round-robined over the banks), so `ttnn.reshape` to
  `[slots, nblk, 160, 576]` is a copy, not a view. A copy is caught by an alias guard during the first run.
- Instead the op reinterprets the page order. With `outer_block_rows = B`, each rank's gather dim is split into
  `rows/B` blocks, and each block is gathered as its own rank-interleaved group:
  rank r's local row `b·B + i` → output row `b·(N·B) + r·B + i`.
  - The op already handles dims between batch and the gather dim as outer groups. Splitting the gather dim just
    feeds that path a virtual outer count (`rows/B`) and a stripe of B rows.
  - It's host-only, in `derive_page_geometry`, validation and hashing. The kernels and the output-shape rule are
    unchanged. Reads go through the TensorAccessor, so an ND-sharded source works.
- The slot path no longer requires singleton outer dims. Only a **row** extent does.
- Scalar `gathered_outer_blocks` gives the live block count; it's patched on program-cache hits.
- These layouts fall back to the scatter schedule (bank-owned needs one stripe group). Phase 0 measures the
  bandwidth. The fallback, if needed, is block-aware batching in the bank-owned schedule.

**E2. Not needed (dropped).**
- With `outer_block_rows`, one `gathered_slab_global` is one block. Pass the global chunk (5120) as the slab, and the
  op's existing on-device `min(round_up(P + 5120, 5120), nblk·5120) / 5120` is the live block count. That covers
  rotated starts (2 blocks), chunk 0 (1 block) and capacity clamping, with no start divisor.
- The scalar path passes `gathered_outer_blocks = min(ceil((P + 5120)/5120), nblk)`.
- Guards: the scalar form gets the same one-block minimum-pages build check as the metadata form. The slab no
  longer needs to divide by the ring size in block mode.

**Op tests** (`tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py`):
- the interleaved `[slots, nblk, 160, 576]` gather on TP (R = 4) and SP (R = 8), against a torch reference;
- extents for aligned, rotated, chunk-0 and capacity-clamped starts, scalar and tensor forms;
- slot selection over several users and layers, with distinct data per slot;
- bytes outside the extent left untouched (poisoned output check);
- a captured trace with the start and user changed between replays, hitting the same program;
- packet batches that cross block boundaries;
- a perf case at the four G3 sizes, recording device time.

## 4. Phases

### 4.1 Phase 0: measure (1–2 days)
1. **Gather device time.** `high_bw_all_gather` on TP, timed as traced replays with the G3 method (warm-up, then the
   median of spaced replays) plus a realtime-profiler breakdown, at the four G3 sizes:
   - today's bank-owned layout;
   - the E1 interleaved view, once E1 builds.

   This replaces the wall-clock numbers and sets the G3 serial gates.
2. **Per-layer timeline** at 50k and 512k / 256k, to locate the gather's position and measure `_expand_v`. If §4.4 is
   needed, this gives the expansion time available to hide the gather in.
3. **Exit:** the projected serial cost is within G3's serial gates. Otherwise plan §4.4 from the start.

### 4.2 Phase 1: integrate PR #59531 (2–3 days)
- **Rebase and resolve:**
  - `mla.py` `_chunked_attn`: keep the PR's cache-write `tp_axis`, the cache-topology assert (`pr-59531:mla.py:1037`)
    and `cache_rows_global`; drop its `ring_mla` call.
  - `mla.py` `__init__`: `set_tp_shard_kv`.
  - `tt_prefill_transformer.py`.
  - `tests/perf/test_mla_perf.py`: take the PR's `and sp_only` and `--wrapper-invocation`.
- **Carry everything else in the PR** (not just the conflicts): TP-aware cache allocation, preload and readback,
  `zero_padded_kv_cache`, the KV chunk table, and the test helpers.
- Remove the dense `assert not self.tp_shard_kv` (`mla.py:997`).

### 4.3 Phase 2: serial gather (4–5 days) → G1, G2, G4, G5, G3 serial gates
- E1 (`outer_block_rows`, `gathered_outer_blocks`) with the op tests (§3); E2 dropped.
- `tt_ccl`: the zero-filled temporary slot, plus a persistent zero `slot_id` tensor.
- `mla.py`:
  - `_tp_gather_latent(kvpe_cache, metadata | scalars, cache_layer_idx)` after the cache write;
  - `_expand_v` and `_unrolled_v_attn` on the temporary slot (§2, steps 3–4);
  - only when `tp_shard_kv`.
- Tests and perf (§4.5).

### 4.4 Phase 3 (after Phase 2 lands): overlap, following sparse MLA's pattern
- **Reuse the existing pieces:**
  - the sparse MLA overlap resources: `tt_ccl.get_sparse_mla_overlap_resources`, profile `loudbox_80_40` on the 12×10
    Galaxy grid, with 80 compute cores (x 0–7), 40 gather cores (x 8–11) and caller-owned ready/data_valid
    semaphores. Dense Kimi layers have no indexer, so the same per-mesh resources can be shared;
  - its closed-region structure (`_select_and_gather_overlapped`): load the manager through the trace controller, run
    the two branches on their sub-devices, and let the clear be the device-side join. On error, reset the semaphores.
  - `SubDeviceTraceController`, which splits the trace at each load/clear. PR #60928's traced-replay timing goes
    through it.
- **Region contents:**
  - the TP latent gather on the 40-core gather sub-device (`subdevice_id`, `sub_core_grids`, the two semaphores). The
    sparse KV gather already runs `high_bw_all_gather` exactly this way (`_gather_kvpe_prefix`);
  - CCL-free compute on the 80-core sub-device. Candidates, picked from Phase 0's per-layer timeline:
    - the q-side projection matmuls and norms that run after the cache write;
    - the expansion of the chip's own stripe straight from the cache. That needs a block-interleaved row map on
      `minimal_matmul`'s output.
- **Budget:** the hidden time is bounded by the CCL-free compute available, about 0.4 ms of projection matmuls per
  layer, prefix-independent. The cost is 2 extra trace segments per layer. Measure both with the traced replay, and
  keep the overlap only where it wins (≤ +0.5%, G3).
- **Fallback** if the available compute is too short: fuse the gather into the expansion matmul as one program.

### 4.5 Tests, perf, CI (in Phase 2, re-run after Phase 3)
**Model**
- `test_mla_chunked_prefill`: `tp_sharded` against `sp_only` **bitwise**, in a paired test with the same seed. Cover
  every 8×4 scenario (rot-*, maxedge-1u/2u, plain-5k, production, padoverflow, deep-50k+5k, deep-2u) × cpu / golden ×
  scalar / metadata, for K3 and K2.7.
- The K3 golden trace at 55k, the KDA carry, and the PR's KV-table tests.

**Captured device trace** (the `reference="trace"` axis is a golden comparison, not this)
- multi-layer, multi-user;
- user and prefix changed between replays;
- long → short → long, to catch stale rows in the shared slot;
- `sp_only` and `tp_sharded` alternated;
- bitwise equal to eager.

**Robustness**
- Poison the temporary slot **beyond the K-chunk-rounded read extent** only. Rows between the gathered and read extents
  are allowed reads; they must stay finite.

**Perf and CI**
- The G3 table from the traced-replay perf test (§5.4), with the realtime-profiler per-op split (gather, expansion,
  SDPA) and the fused ring gather's share of SDPA time (§6.5).
- `K3_CHUNKED_RT_PERF_NS` updated for the branch.
- The PR's `tp_sharded` CI legs running on unrolled-V.

## 5. Build and run

### 5.1 Environment (once per machine)
```bash
cd ~/tt-metal
source /opt/venv/bin/activate          # the Python env; activate in every new shell
export TT_METAL_HOME=$PWD
# If ttnn does not import (namespace package / missing deps), install the dev requirements and tt-metal itself into the venv:
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu -r tt_metal/python_env/requirements-dev.txt
uv pip install --index-strategy unsafe-best-match -r tools/triage/requirements.txt
uv pip install -e .
python -c "import ttnn; assert hasattr(ttnn, 'open_mesh_device')"
```

### 5.2 Build
```bash
./build_metal.sh --release                 # needed after the high_bw_all_gather (C++) changes
./build_metal.sh --release --build-tests   # also builds the C++ gtests
```
A fresh worktree needs `git submodule update --init --recursive` before the first build.

### 5.3 Galaxy TorusXY environment (every test shell)
```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_TORUS_XY_CERTIFIED=1
export OMP_NUM_THREADS=$(nproc)
# Golden-trace runs:
export KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e
```

### 5.4 Running tests
Device tests always go through the wrapper. It serializes device access with a lock and resets the device after a
hang. Exit codes: 0 pass, 1 fail, 2 hang. `--run-all` runs every case; `--dev` turns on the watcher and triage.
```bash
scripts/run_safe_pytest.sh [--run-all] [--dev] <test> [pytest args]
```

**Op (E1 / E2):**
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py
```

**MLA accuracy (each phase):**
```bash
T=models/demos/deepseek_v3_d_p/tests/test_mla.py
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill -k "torus-xy-8x4 and tp_sharded and (k3 or k2_7)"
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill \
  -k "k3 and torus-xy-8x4 and scalar and ((maxedge-1u and cpu and no_determinism) or (plain-5k and func and with_determinism))"
scripts/run_safe_pytest.sh $T::test_mla_chunked_prefill \
  -k "k3 and production-50k+5k and trace and torus-xy-8x4 and scalar and no_determinism"
scripts/run_safe_pytest.sh --run-all $T::test_mistral_small4_mla
scripts/run_safe_pytest.sh --run-all $T -k llama4
scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v3_d_p/tests/test_kv_cache_table.py
scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v3_d_p/tests/kimi_k3/test_chunked_prefill.py
```

**MLA perf (G3).**
- `test_mla_chunked_perf_check -k "k3 and torus-xy-8x4"` is the CI 50k + 5k check.
- G3 itself comes from a **traced-replay perf test**, following
  `models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py` from PR #60928:
  - parametrized over `variant` (k3 / k2_7), `prefill_len` (50k and 512k / 256k) and `tp_shard_kv`;
  - preload the prefix, then run an eager forward of one 5k chunk to compile;
  - capture the forward (MLA does not free its input, so no input-refresh trace is needed);
  - one untimed warm-up replay, then 5 replays 2 s apart, each timed with `time.perf_counter_ns()` around
    `execute_trace` + `synchronize_device`; gate on the median, and require the middle three to agree within 2%.
  - Main's baseline uses the same method: `main_baseline_traced.patch` (next to this plan) adds `trace_time` and a
    four-point test to a main worktree.
- Keep the realtime-profiler run (`_run_chunked_prefill(..., profile=True)`) for the per-op breakdown only.
- Both need Blackhole and a high-power host.

**Baseline on the same machine** (main or `sp_only` from another commit):
```bash
git worktree add /data/$USER/tt-metal-baseline <commit> && cd /data/$USER/tt-metal-baseline
git submodule update --init --recursive && TT_METAL_HOME=$PWD ./build_metal.sh --release
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD/tt_metal/tt-llk/tools/python:$PWD
python -c "import ttnn; print(ttnn.__file__)"   # must point into the baseline worktree
```

## 6. Keeping the next step open: expanded K on arrival

**The next step after this plan:** expand K per head inside the SDPA from the latent it already receives, and run QK
unabsorbed. That's 192 wide (128 no-PE + 64 PE) instead of 576. Napkin estimate on the Galaxy: attention −21% to −26%
at HiFi2 and −4% to −20% at LoFi. Only the latent travels, so it works on the sharded cache. Constraints this plan
respects so that step stays a kernel change:

1. **The latent stays the K wire format.** The temporary slot, `ring_joint`'s K input and its SP-ring K gather all
   carry the 576-wide latent. Nothing in this work introduces a per-head K (or V-only) transport.
2. **The temporary slot keeps all 576 columns,** including the 64 PE columns. V expansion reads only the first 512
   (`in0_k_prefix`); K expansion needs the PE part too. It's named and documented as a latent slot.
3. **SP-local row order, the three extents (§2) and the single-slot read path are part of the interface.** K expansion
   reuses the same rows, masks and row counts.
4. **E1 and E2 stay generic:** any `cluster_axis`, ring size and block size. With K expanded, every chip needs every
   latent row, and a full-mesh latent gather (two Hamiltonian cycles) could deliver this row's stripes and the remote
   rows together.
5. **CCL headroom is tracked.** The §4.5 perf table records the fused ring gather's share of SDPA time. With K
   expanded, it rises to an estimated 73–86% of the LoFi compute. The ring-gather levers in §7 then become part of the
   K step.
6. **The fused gather-into-expansion from §4.4, if built, keeps a latent output** (the temporary slot) next to V, so
   the SDPA can still read K from it.

## 7. Risks and when to revisit

**Risks**
1. **Scatter-schedule bandwidth** for the interleaved view (E1). Phase 0 measures it. The fallback is block-aware
   batching in the bank-owned schedule (§3, E1).
2. **The serial gather at short prefixes:** fixed per-call device latency is unmeasured. Phase 0 measures it, and §4.4
   is the remedy.
3. **Block count on cache hits:** `gathered_outer_blocks` is patched like the existing extent arguments and re-validated on every hit. The op tests change the count on one program and pass a bad count on a hit.
4. **The single shared slot with several users in one forward** (multi-user chunked prefill): users are processed one
   after another, so the slot is reused in queue order. The multi-user tests cover it.

**Revisit the design if:**
- SDPA compute drops about 30%, for example LoFi plus the 14 idle cores, or K expanded (§6). The SP-ring gather then
  becomes the bottleneck. Levers, in order:
  1. batched writes in the fused gather;
  2. the latent on two edge-disjoint Hamiltonian cycles (Staletovic's `fabric_all_gather`, 184.6 GB/s per chip);
  3. 2D routing for V.
- DRAM, not speed, becomes the limit. Expanding V on arrival removes the 1.74 GB gathered-V buffer per chip at K3 512k,
  at +13–21% attention time.

## 8. Changes from rev 3 (review round 2)

| Finding (reviewer) | Change |
|---|---|
| E1 isn't writer-only: the reader relays through the output layout, and interleaving breaks the bank-owned schedule (both) | E1 is now an outer-dimension view the iterator already interleaves; bandwidth re-measured in Phase 0; bank-owned batching is the fallback |
| E2 bounds were in the wrong units; rotated chunks span two blocks; staircase write offsets (both) | E2 is a start divisor on the existing extent formula; serial gather of all live blocks after the cache write covers rotation and overflow |
| An empty prefix window hangs (Claude) | No prefix/tail split; the serial gather is never empty |
| No dense CCL sub-device, trace segmentation, grid conflicts, MoE's manager, TP collectives contention (both) | Overlap with projections dropped; the only optimization is the single-program fused gather-into-expansion (§4.4) |
| The slot is read by `ring_joint` too, so ordering keyed on `_expand_v` was wrong (both) | Single queue, program order; the next gather runs after `ring_joint` |
| Gate math inconsistent (0.25 ms was wall time); metric can't see overlap (both) | G3 = wall-clock of one traced replay (PR #60928 method), which includes overlap and excludes host dispatch; serial gates set from device-time estimates; Phase 0 measures them |
| `_expand_v` and SDPA read past the gathered rows by design (both) | Three extents defined; slot zero-filled; poison test only beyond the read extent |
| Single-slot wiring: drop the slot tensor in `_expand_v`, zero `slot_id` + `kv_cache_layer_idx=0` for SDPA (both) | §2 steps 3–4 |
| G4 can be bit-exact; determinism should be bitwise; trace tests must capture device traces (both) | G4, G5, §4.5 |
| Integration needs the whole PR, not just the conflict hunks (Codex) | §4.2 |
| Wrong op test path (both) | §3, §5.4 |
| Layer index is a captured host argument (Codex) | G5 wording: per-layer captured invocations; user and prefix change |
