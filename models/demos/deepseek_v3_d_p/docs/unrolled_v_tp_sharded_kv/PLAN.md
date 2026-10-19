# Plan: unrolled-V MLA on the fully sharded KV cache (Galaxy 8×4)

Author: Pavle Josipovic (with Claude) · 2026-10-17, updated 2026-10-19 to the final design. Branch
`pjosipovic/unrolled-v-tp-sharded-kv`, which carries unrolled-V and merges PR #59531. Status, measurements and how to
run the tests: [STATUS.md](STATUS.md). Expanded K, built on top: [../expanded_k/PLAN.md](../expanded_k/PLAN.md).

## 0. Exit criteria

**Done (ready to merge) when every item holds**, on the Galaxy 8×4 TorusXY, with the release build of the branch:

| # | Criterion | How it's checked |
|---|---|---|
| X1 | `tp_shard_kv=True` runs unrolled-V for K3 and K2.7, with no `ring_mla` call and no dense `tp_shard_kv` assert. Other dense models keep `ring_mla` (`mla_config.MLA_ATTENTION_CONFIG`). | Code; the X2 tests run this path |
| X2 | `tp_sharded` output is **bitwise equal** to `sp_only` on the same seed for every 8×4 `test_mla_chunked_prefill` scenario × {cpu, golden} × {scalar, metadata}, for K3 and K2.7 | New paired test, all green |
| X3 | No regressions: the 53 Galaxy MLA tests (K3/K2.7/Mistral4 chunked, `test_mistral_small4_mla`, llama4) pass on `sp_only`. The PR's KV-table tests and the K3 KDA-carry test pass on `tp_sharded`. | `run_safe_pytest.sh --run-all` |
| X4 | A captured multi-layer, multi-user device trace replays bitwise equal to eager, with user and prefix changed between replays, including long → short → long | New trace test |
| X5 | Traced-replay forward time, `tp_sharded` vs `sp_only`, within the gates (worst measured sample plus about 1 pp of run-to-run noise; values in STATUS.md, X5). Method: G3 | `test_mla_chunked_tp_sharded_traced_perf` |
| X6 | DRAM saving per user slot, measured as an allocation delta: ≥ 0.60 GB (K3 512k), ≥ 0.85 GB (K2.7 256k) | `test_mla_tp_sharded_dram_saving` |
| X7 | `high_bw_all_gather`: the new outer-block (E1) tests and all existing tests pass (failures that also occur on main, on this machine, excepted). | `test_high_bw_all_gather.py` |
| X8 | `sp_only` traced perf within ±1% of before the change; the `tp_sharded` CI legs green (PR #59531 makes `tp_sharded` the Kimi default; `PREFILL_TP_SHARD_KV=0` rolls back) | CI run |

Every change was reviewed by Claude Code and Codex.

**Phase 3 (prefetch) exit:** traced-replay overhead ≤ +0.5% at all four points, with X2–X4 and X8 still holding. Met at
K3 50k, K3 512k and K2.7 256k; K2.7 50k misses it, and the prefetch stays on there by the author's decision
(STATUS.md).

## 1. Goals

### Background
- **PR #59531** shards the dense MLA KVPE cache block-cyclically over SP×TP. It is the Kimi default
  (`PREFILL_TP_SHARD_KV=0` rolls back to the TP-replicated cache).
  - Chip (s,t) keeps region t (160 rows) of each 640-row SP slab: 1/32 of the cache, 4× less KV DRAM per chip per
    layer.
  - It reads the cache with split-KV `ring_mla`.
- **Unrolled-V** replaces dense chunked `ring_mla` for Kimi K2.7 / K3 (`mla_config.MLA_ATTENTION_CONFIG`):
  - `_expand_v` computes V = latent × W_UV for the chip's 24 (K3) / 16 (K2.7) heads with `minimal_matmul`;
  - `ring_joint_scaled_dot_product_attention` gathers K and V along the SP column, overlapped with the math.
- **Why they clash:** V is per head group. A chip can expand V only for rows it holds, using its own heads' W_UV. With
  the sharded cache, 3 of every 4 rows of its SP slab are on row neighbours that hold other heads' weights.

### Goals (all measured on the Galaxy 8×4 TorusXY)

**G1. Both features on together.** Dense chunked MLA runs unrolled-V with `tp_shard_kv=True`, for K3 and K2.7. No
fallback to `ring_mla`.

**G2. The DRAM saving survives.** Measured as allocation deltas against `sp_only`, at 1 user. The persistent cache is
1/(sp·tp) of the rows. The added buffers are the TP latent buffers (two with the prefetch, §2.3), allocated once for all
layers and users.

**G3. The speed-up survives.** The metric is the **host wall-clock of traced replays of the MLA forward**, adapted
from PR #60928 (Kimi MoE perf gate):
1. an untimed eager pass to compile;
2. capture;
3. one untimed warm-up replay;
4. five timed replays, 2 s apart, each ending in `synchronize_device`. The gate uses their median, and the middle three
   must agree within 2%.

The first replay after a capture runs about 2% fast at 50k, and back-to-back replays slow down under sustained load,
so neither a single replay nor back-to-back medians are used. The metric includes overlap and gaps between programs,
and excludes host dispatch. The realtime profiler's per-program sum is kept only as the per-op breakdown. Compare
`tp_sharded` against `sp_only` on the same build, and main's `ring_mla` timed the same way.

**G4. Identical results.** The TP latent buffer holds exactly the bytes the cache held before the PR, and everything
downstream is unchanged. So `tp_sharded` output must be **bit-identical** to `sp_only` on the same seed, across the
full `test_mla_chunked_prefill` 8×4 matrix (cpu and golden reference, scalar and metadata, rotated and overflow
scenarios, multi-user), plus the K3 KDA-state carry. Determinism repeats are bitwise equal.

**G5. Trace-safe, with a rollback.**
- A captured multi-layer, multi-user device trace replays correctly when the user slot and prefix change. Each layer is
  its own captured invocation, with a fixed layer index.
- `tp_shard_kv=False` stays byte-for-byte on the previous path, with one deliberate exception: the V expansion's fp32
  accumulation ([../expanded_k/STATUS.md](../expanded_k/STATUS.md), decisions).
- Sharding follows PR #59531's default (on for Kimi); `PREFILL_TP_SHARD_KV=0` runs the replicated path.

### Non-goals
- The sparse (DSA) path, and dense models without an `MLA_ATTENTION_CONFIG` entry (Mistral-Small-4, DeepSeek-V3),
  which keep `ring_mla`.
- Gather-routing changes inside `ring_joint`.

## 2. The design

Per layer and chunk, on chip (s,t):

```
q/kv projections ── update_padded_kv_cache (PR #59531, writes this chip's region) ── TP latent gather ── expand V(t) ── ring_joint SDPA
                                                                                    high_bw_all_gather    unchanged      unchanged
                                                                                    into a TP latent      reads the      K = the
                                                                                    buffer, SP-local      buffer         buffer
                                                                                    order
```

### 2.1 TP latent buffers

Allocated once in `tt_ccl.get_mla_tp_latent_slot`, **zero-filled**, shared by every user.
- Shape `[1, 1, cap_sp, 576]` bfp8, SP-sharded, where `cap_sp = tp × the deduplicated cache's local capacity` (the
  pre-PR local capacity).
- It holds this SP row's latent in **SP-local order**: global row `c·5120 + s·640 + t′·160 + off` sits at local row
  `c·640 + t′·160 + off`. That's exactly the pre-PR cache layout.
- All 576 columns are kept, so K can be expanded from it ([../expanded_k/PLAN.md](../expanded_k/PLAN.md)).
- **Two buffers, by layer parity** (`index`): with the prefetch on, layer L reads buffer L % 2 while the next layer's
  latent is prefetched into the other. With the prefetch off, every layer uses buffer 0. `set_tp_shard_kv` allocates
  them at init, never inside a forward or a capture.

### 2.2 Serial path: gather, expand, attend

1. **TP latent gather** (`_tp_gather_latent`), after the cache write:
   `high_bw_all_gather(cluster_axis=TP, outer_block_rows=160)`.
   - Source: the deduplicated cache itself, `[slots, 1, rows, 576]` and ND-sharded. There is no reshape: a reshape
     of the ND-sharded cache would be a copy.
   - The op gathers each 160-row region as its own block, interleaving the 4 ranks per chunk into SP-local order.
   - The slot comes from `input_batch_index_tensor` (user id) + `batch_slot_num_layers` / `batch_slot_layer_idx`
     (host runtime arguments, fixed per captured layer invocation); on the scalar path from `input_batch_index`.
   - The extent is every live block. On the metadata path: `gathered_prefix_tensor = metadata[1]`,
     `gathered_slab_global = 5120`. On the scalar path: `gathered_outer_blocks = min(ceil((P + 5120)/5120), nblk)`.
   - It covers rotated chunks (2 blocks) and clamps to capacity. It's never empty: chunk 0 gathers 1 block.
2. **Expand V.** `_expand_v` reads the buffer as a one-slot cache (`MinimalMatmulDynamicRows` without a slot tensor,
   `num_layers=1`, `layer_idx=0`), with the same row formula as before: `min(round_up(ceil(P/5120)·640 + 640, k_chunk),
   cap_sp)`.
3. **SDPA.** `ring_joint` takes K from the buffer.
   - Metadata path: a persistent one-element uint32 zero tensor (`tt_ccl.get_zero_slot_tensor`) as `slot_id`, the
     original `kv_actual_isl_tensor`, `kv_cache_num_layers=1`, `kv_cache_layer_idx=0`. Validation accepts this
     (`validate_metadata_tensors`, `ring_joint_sdpa_device_operation.cpp`).
   - Scalar path: `kv_cache_batch_idx=0`.
   - The SP-ring gather, masks and configs are unchanged.

### 2.3 Prefetch: the next layer's gather during this layer's SDPA

Most of a layer's gather does not depend on that layer's computation: it moves the layer's earlier chunks, which are
already in its cache. Only the current chunk is new, and every TP chip already holds that chunk's whole SP-local latent
(`tt_kvpe`). So:
1. **Sub-devices** (`tt_ccl.get_tp_latent_prefetch_resources`, created at init): the SDPA sub-device is the SDPA
   compute grid (x 0–10) plus its ring all-gather workers (x 11, y 0–3); the gather sub-device is the 6 idle cores at
   x 11, y 4–9, with its own semaphore pair.
2. **Prefetch** (`_sdpa_with_tp_latent_prefetch`): layer L loads the manager, issues layer L+1's TP gather on the gather
   sub-device and its own `ring_joint` on the SDPA sub-device, then clears the manager. The gather uses the TP links,
   which the SDPA (SP ring) does not use. It reads layer L+1's cache, which layer L does not touch, and writes buffer
   (L+1) % 2, last read by layer L−1. The manager clear is the device-side join. On an exception the default manager
   is restored and the semaphores are reset after draining.
3. **Consume** (`_write_tp_latent_chunk`): layer L+1 writes only its current chunk into the prefetched buffer, locally:
   `update_padded_kv_cache` without `tp_axis`, i.e. the pre-#59531 TP-replicated write, clamped to `actual_end`. The
   prefetch read this layer's cache before its write, so this chunk's rows in it are stale until then; rows past
   `actual_end` are written by neither and hold the same bytes either way.
4. **Identity.** The host cannot see metadata values, so the prefetch is scoped. It runs only inside
   `tt_ccl.tp_latent_prefetch_chain()`, which the owner of the layer loop opens (`TtPrefillTransformer` and the K3
   transformer). The caller promises that the forwards inside are consecutive layers of one chunk, with unchanged
   metadata; `stop_layer` excludes a kv_only last layer.
   - A host-side key (`_tp_latent_key`, stored in `tt_ccl.tp_latent_prefetched`) records what was prefetched: the
     chain, cache buffer and local chunk length, plus the layer and metadata tensors (metadata path) or the flat slot
     and prefix (scalar path). The next forward consumes the prefetch only if its key matches.
   - The key is dropped on entering or leaving a chain, by `set_tp_shard_kv`, and by every forward that reads it.
   - Anything else, including layer 0 and any forward outside a chain, gathers serially (§2.2).
5. **Traces.** `SubDeviceTraceController` splits the capture at each load/clear, giving 2 extra segments per layer.
   `TtPrefillBlock` and the K3 block forward the controller to MLA.
6. `ttMLA(tp_latent_prefetch=False)` turns the prefetch off (one buffer, serial gathers).

**Cost.** One extra local write per layer (≈9 µs), and the second latent buffer: ≈40 MB per chip at K3 512k, against
a DRAM saving of over 0.6 GB.

### 2.4 Three extents, kept apart

- **Valid tokens:** masked by the SDPA through `logical_n` / `kv_actual`.
- **Gathered rows:** the gather's extent (§2.2, step 1).
- **Rows read:** `_expand_v` rounds up to the SDPA K chunk, and the SDPA reads whole K chunks. So the expansion and
  SDPA can read up to one K chunk past the gathered extent.

Those rows only ever hold zeros (from allocation) or latent gathered for an earlier layer or user. Both are finite, and
the SDPA masks them. That's the same contract as the cache's (`_expand_v` docstring).

### 2.5 Ordering

- **Serial:** everything is on one queue, in program order. A layer's gather into a buffer starts only after the
  previous reader of that buffer (an earlier layer's `ring_joint`) has finished.
- **Prefetch:** layer L+1's gather writes buffer (L+1) % 2 inside layer L's load/clear region. Its last reader, layer
  L−1's `ring_joint`, finished before that region was loaded, and its next reader, layer L+1, starts after the clear.
- Several users in one forward are processed one after another, so the buffers are reused in queue order.

### 2.6 Why this design (measured on this Galaxy)

- **The attention is compute-bound** at 5k chunks with absorbed K: the fused K+V ring gather alone takes 43–48% of the
  HiFi2 SDPA time. So only work in front of the SDPA shows up.
- **The TP latent gather is that work, and it's small.** Each chip receives 3/4 of its SP row's latent: 3.2 MB at 50k,
  30.6 MB at K3 512k. Serially it costs about 0.1 ms at 50k (mostly fixed latency) and 0.3–0.6 ms at 256k–512k; the
  prefetch hides nearly all of it at 256k–512k.
- **Every alternative is slower on this hardware:** shipping V over TP +7–11%, expanding V on arrival +13–21%, routing
  K over the full mesh no gain.

## 3. Op work in `high_bw_all_gather`

`ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/` already had `cluster_axis=1` as a TorusXY ring,
persistent output, `input_batch_index_tensor` slot selection, and the `gathered_prefix_tensor` /
`gathered_slab_global` extent.

**E1. `outer_block_rows`: gather in rank-interleaved blocks, with no reshape.**
- The KV cache is **ND-sharded DRAM** (32-row shards round-robined over the banks), so `ttnn.reshape` to
  `[slots, nblk, 160, 576]` is a copy, not a view.
- Instead the op reinterprets the page order. With `outer_block_rows = B`, each rank's gather dim is split into
  `rows/B` blocks, and each block is gathered as its own rank-interleaved group:
  rank r's local row `b·B + i` → output row `b·(N·B) + r·B + i`.
  - Splitting the gather dim gives the page schedule an outer count of `rows/B` and a stripe of B rows. Selected-slot
    and partial gathers still require singleton dims between batch and the gather dim.
  - It's host-only, in `derive_page_geometry`, validation and hashing. The kernels and the output-shape rule are
    unchanged. Reads go through the TensorAccessor, so an ND-sharded source works.
- Scalar `gathered_outer_blocks` gives the live block count; it's patched on program-cache hits.
- These layouts use the scatter schedule (bank-owned needs one stripe group).

**E2. Not needed.** With `outer_block_rows`, one `gathered_slab_global` is one block. Passing the global chunk (5120)
as the slab makes the op's existing on-device `min(round_up(P + 5120, 5120), nblk·5120) / 5120` the live block count.
That covers rotated starts (2 blocks), chunk 0 (1 block) and capacity clamping. The scalar form gets the same
one-block minimum-pages build check as the metadata form, and the slab no longer needs to divide by the ring size in
block mode.

**Op tests** (`tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py`):
- the rank-interleaved block gather of a `[slots, 1, rows, 576]` cache (logically `[slots, nblk, 160, 576]`; no such
  4-D tensor exists) on TP (R = 4) and SP (R = 8), against a torch reference;
- extents for aligned, rotated, chunk-0 and capacity-clamped starts, scalar and tensor forms;
- slot selection over several users and layers, with distinct data per slot;
- bytes outside the extent left untouched (poisoned output check);
- a captured trace with the start and user changed between replays, hitting the same program;
- packet batches that cross block boundaries.

## 4. Phases

| Phase | Delivered |
|---|---|
| 0 | Gather device time measured at the four G3 sizes; X5's serial gates set from traced runs |
| 1 | PR #59531 merged into unrolled-V, carrying the whole PR (TP-aware cache allocation, preload and readback, `zero_padded_kv_cache`, the KV chunk table, test helpers); the dense `tp_shard_kv` assert removed |
| 2 | Serial TP latent gather (§2.2), E1, the paired bitwise, trace, DRAM and perf tests. X1–X8 |
| 3 | Prefetch (§2.3, §4.4) |

### 4.4 Phase 3: prefetch

Design in §2.3; results in STATUS.md (X5). The first approach, hiding the gather behind the q-side projections on an
80/40 core split, was ruled out by measurement:
- **Too little to hide behind.** Per 5k chunk (realtime profiler) the TP latent gather is 177 µs at K3 50k, 781 µs at
  K3 512k and 459 µs at K2.7 256k; the CCL-free q-side compute is about 170 µs (K3) and 110 µs (K2.7).
- **Not confinable.** Almost every q-side op (`q_b_proj`, `nlp_create_q_heads_split`, `rms_norm`, the RoPE ops) puts
  work on cores in both halves of an 80/40 split, which is a fatal error under a loaded manager, and several of these
  ops take no core-grid argument.

### 4.5 Tests

**Model**
- `test_mla_chunked_prefill`: `tp_sharded` against `sp_only` **bitwise**, in a paired test with the same seed, over
  every 8×4 scenario (rot-*, maxedge-1u/2u, plain-5k, production, padoverflow, deep-50k+5k, deep-2u) × cpu / golden ×
  scalar / metadata, for K3 and K2.7.
- The K3 golden trace at 55k, the KDA carry, and the PR's KV-table tests.
- Prefetch == serial == `sp_only` bitwise over 3 layers and 2 users, with chain and layout-change negative cases.

**Captured device trace** (the `reference="trace"` axis is a golden comparison, not this): multi-layer, multi-user;
user and prefix changed between replays; long → short → long, to catch stale rows; `sp_only` and `tp_sharded`
alternated; segmented by the prefetch; bitwise equal to eager.

**Robustness:** poison the TP latent buffers **beyond the K-chunk-rounded read extent** only. Rows between the
gathered and read extents are allowed reads; they must stay finite.

## 5. Build and run

See [STATUS.md](STATUS.md), "How to reproduce".

## 6. Expanded K on arrival

Built on top of this plan: [../expanded_k/PLAN.md](../expanded_k/PLAN.md) and
[../expanded_k/STATUS.md](../expanded_k/STATUS.md). This plan keeps it a kernel change: the latent stays K's wire
format, the TP latent buffers keep all 576 columns, and SP-local row order, the three extents (§2.4) and the
single-slot read path are part of the interface.

## 7. Split-KV `ring_mla` (PR #58613) stays

PR #58613 added a split-KV mode to `ring_mla` (`cluster_axis=None`): the latent cache striped over SP × TP and attended
as packed sources inside the SDPA. Kimi no longer reaches it (the TP latent gather plus unrolled-V `ring_joint` replace
it there), but the dense models without unrolled V (Mistral-Small-4, DeepSeek-V3) still read a TP-sharded cache through
it, so it is not removed. Expanded K and unrolled V do not depend on it; `k_expand` rejects it in the device op.
