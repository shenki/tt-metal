# AUTODEBUG — intermittent nondeterminism of traced prefill logits (LFM2.5-24B-A2B, 1x4 Blackhole, TP4)

This was an inspection-only investigation; nothing was run on hardware. Every claim below cites the source it rests on. Confirming the runtime behaviour still needs the experiments in §5.

Paths are relative to the repo root (`/home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal`). `AP/` stands for `models/autoports/liquidai_lfm2_5_24b_a2b/`.

---

## 1. Headline

### H1. The multicast all-gather can declare completion before all data has landed

**Where it happens.** Every S=64..512 prefill all-reduce ends in a `ttnn.all_gather`, and on this 1x4 ring that all-gather runs the multicast factory. Its completion barrier does not cover the packets that the chip two hops away sends over the *alternate* (backward) route.

**What goes wrong.**
- A receiver can count DONE = 3 and leave the op while up to half of that sender's packets are still in flight.
- The next op in the trace is the residual `ttnn.add(x, r)`. It can read the all-gather output before those tiles land, so it reads stale contents.
- Under trace replay every address is fixed, and the prompt is the same every time. So the stale contents are deterministic, e.g. whatever an earlier op of the same replay left at that DRAM address.

That gives the shape of the symptom: rare, a handful of discrete wrong results that recur bit-exactly, and independent of the slot.

**How sure this is.** The contract violation is proven in code (§3). That the S=512 prefill actually selects this op variant is proven by the developers' own tracy capture. That this race *is* what produces the 9/149 is the best fit to the evidence but has not been confirmed on hardware; §5 lists the cheapest decisive checks.

It explains every observation except one: the higher rate with decode traffic between prefills has only an unproven explanation (§4).

### Fixes implied (smallest first)
1. **Fix the op (tt-metal).** Make the completion credit for each sender→receiver pair travel every route that sender's data used to reach that receiver. For example, send DONE on both the primary and the alternate route, have the reader wait for its writer's last send, and raise `barrier_wait_value` to match.
   - The quickest A/B test of the diagnosis is to disable route alternation: `axis_load_balance = false` at `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp:123`.
2. **Avoid the op from the model.** `MeshReduce.prefill` (`AP/tt/multichip_decoder.py:785-794`) can avoid the multicast all-gather for S=64..512. This cannot be done by passing `num_links`: `ttnn.all_gather` ignores that deprecated argument (`ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp:143-160`), so factory selection is purely size-driven.
   - One candidate is the semaphore-carrying `ttnn.experimental.all_reduce_async(...)` overload. It routes through `reduce_scatter_minimal_async` + `prim::all_gather_async` (`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:402-457`), with handles from `models/common/modules/tt_ccl.py`. I have **not audited** that path.

---

## 2. Observations vs interpretations

**Direct observations** (from `AP/doc/vllm_integration/logs/diag_*.log` and the stress script):

| Run | Mismatches vs run 0 |
|---|---|
| Decode steps (3 live requests, bucket-4 model + sampling traces) between prefills | 9/149 |
| Same, with `ttnn.synchronize_device` before every prefill | 10/149 |
| No decode | 1/149 |
| Layers 0-19 | 4/149 |
| Layers 20-39 | 2/149 |
| Layers (0,2,3) | 0/149 |
| Layer 2 alone | 0/149 |

- Mismatches have max |Δlogit| of 1.16 to 3.97, with 123.8k to 127.8k of 128k vocab entries differing, and argmax sometimes flips.
- Identical (maxdiff, ndiff, argmax) tuples recur across iterations, slots and separate processes:
  - 2.385 / 126374 / argmax 324: four times, in two runs.
  - 1.164 / 123838: three times.
  - 1.367 / 124607: twice, in two runs.
- Layer-0-only and layer-3-only runs crashed: an attention-free model breaks `set_kv_cache` (`IndexError`, `AP/tt/model.py:313`).
- `diag_stress_eager.log` (`LFM2_EAGER_PREFILL=1`) stops during weight loading. **There is no trace-vs-eager result.**
- Each prefill writes to fresh KV blocks, which `prepare()` zeroed; blocks are never recycled within a run. Prompt: 300 tokens → bucket 512, first-chunk program, `host_logits` path (no sampler).

**Interpretations** (mine):
- "A few discrete corrupt states": several distinct wrong states recur bit-exactly. That means deterministic wrong data reached through a timing-dependent choice. It is not accumulation-order noise.
- The large magnitude is plausibly amplification:
  - A wrong tile in the residual changes MoE top-k selections (full fp32 compare, `AP/tt/optimized_decoder.py:1135-1149`).
  - Expert-parallel dispatch assumes routing is **bitwise identical on all 4 chips** (`AP/tt/multichip_decoder.py:370-380`). A chip-local stale tile breaks that assumption, so a token can lose or double an expert contribution.
- The 3-layer and 1-layer nulls are statistically weak. At the full model's per-all-reduce rate, 6 all-reduces would give about 0.7 expected events in 149. They do not show the bug is layer-specific.

---

## 3. H1 in detail: lowered chain and the violated contract

### 3.1 What the S=512 prefill all-reduce lowers to

1. **Model call.** `MeshReduce.prefill`, for S > 32, calls `ttnn.all_reduce(partial[1,1,512,2048] bf16 DRAM, num_links=2, topology=Ring, memory_config=DRAM)` (`AP/tt/multichip_decoder.py:785-794`). It is called twice per layer, so 80 times per trace (`:975`, `:981`). The result goes straight into `ttnn.add` (`:976`, `:982`).
2. **all_reduce.** `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp:45-57` calls `experimental::all_reduce_async` without semaphores.
   - The scatter dimension is 3 (tile shape [1,1,16,64]).
   - Neither the composite all-gather nor the composite reduce-scatter applies.
   - So it runs `ttnn::reduce_scatter` (`all_reduce_async.cpp:415`) followed by `ttnn::all_gather(scattered, dim, cluster_axis, ...)` (`:459`). No `num_links` is passed to the all-gather.
3. **Reduce-scatter.** The 2 MiB input is above the 512 KiB direct-path gate (`ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.cpp:54,91`), so it takes the ring `prim::reduce_scatter`. Its protocol is balanced and its reduction order is fixed (§6).
4. **All-gather factory selection.** On Blackhole, `per_link_bytes = 512·512·2·4 / 4 links = 524,288`, which is below the 1,000,000 multicast ceiling for 2 KB pages, so it uses `AllGatherMulticastFactory` (`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp:255-276`).
   - **Confirmed by the developers' profile.** `AP/doc/datatype_sweep/perf/tracy_probe_selected/ops.csv.xz` shows `AllGatherDeviceOperation` with input [512,512] on 4 cores. Its attributes are `axis_num_links {0;4}`, `axis_topology {Linear; Ring}`, `fabric_config FABRIC_1D_RING`, `packet_size 8192`.
   - **Which bucket sizes use which factory.** [64,512], [128,512] and [256,512] also run on the 4-core multicast factory. [1024,512] and [2048,512] run on 24 cores, i.e. the unicast factory.

### 3.2 The multicast protocol on a 4-ring

- **Route lengths.** On a 4-ring, forward is 2 hops and backward is 1 hop (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1833-1834`).
- **Route alternation is on.** `axis_load_balance = ring && axis_size even` is true (`all_gather_multicast_factory.cpp:123`). It is passed to both kernels as a compile-time argument (`:168`, `:375`, `:391`).
- **Reader routes** (forward connection). Primary is `line_hops = e_hops = 2`; alternate is `line_hops_alt = w_hops = 1` (`:465-468`).
- **Writer routes** (backward connection). Primary is `w_hops = 1`; alternate is `e_hops = 2` (`:509-513`).
- **Per-packet alternation.** `FabricWriter` flips between the primary and alternate route on every packet (`kernels/multicast_common.hpp:122-124,140-141`). Both reader and writer send every chunk, starting on the primary route.
  - Pages are 2 KB tiles, the packet size is 8192 B, and `NOC_SCATTER_WRITE_MAX_CHUNKS = 4` (`tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp:194`). So one packet carries 4 tiles.
  - Each chip's [512,512] slice is 256 tiles, 64 per link, which is **16 packets per link**.
- **Who delivers what, for sender X.**
  - **Even packets:** reader → X+1 and X+2 (forward); writer → X+3.
  - **Odd packets:** reader → X+1 only; writer → X+3, then X+2 (backward, 2 hops).
- **Consequence:** chip X+2 receives X's **odd packets only via the backward path X→X+3→X+2**.
- **Completion credits.** Both reader and writer send DONE on `sem_route_id`, which was set up with the **primary** `ranges` only:
  - reader: `kernels/multicast_reader.cpp:130-139,309-314`;
  - writer: `kernels/multicast_writer.cpp:122-131,235-240`.
  - Each chip waits for `DONE ≥ barrier_wait_value = num_devices − 1 = 3` (`multicast_reader.cpp:316-317`; factory `:447`).
  - Receiver Y's three credits come from: reader(Y−1) forward, **reader(Y−2) forward (Y−2→Y−1→Y)**, and writer(Y+1) backward.
- **The gap.**
  - The reader comment (`multicast_reader.cpp:300-308`) says "the sem is sent after all data sends on a particular link, so it's correctly ordered at the receiver". That holds only for data on the *same* path.
  - Y−2's odd packets reach Y on the other path (Y−2→Y+1→Y), which ends at a different router on Y.
  - The fabric's flush-before-increment orders only the receive channel the increment arrived on (`flush_write_to_noc_pipeline(rx_channel_id)`, `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:96,188-193`).
  - Worse, reader(Y−2) sends its DONE right after its own last send (`multicast_reader.cpp:270-314`) without waiting for its writer. The writer works through the same CB pages after the reader pushes them, and can lag by up to one CB page: 4 tiles per packet × multiplier 4 = 16 tiles, i.e. 4 packets (factory `:286-306`). So the last odd packets toward Y may not even have left Y−2 yet.
  - Y therefore exits the all-gather with no guarantee that Y−2's odd packets have arrived. The dispatcher starts the next program as soon as Y's workers finish. Incoming fabric writes are carried out by Y's ethernet routers and are not part of that completion count.
- **Which data can be late.** Hidden columns `512·((Y+2) mod 4) + [128,256) ∪ [384,512)` of the gathered residual on chip Y.
  - These are the odd packets, i.e. tile columns 4-7 and 12-15 of that slice. CB pages are whole packets, so the alternation parity carries across pages.
  - Link l covers tile rows 4l..4l+3. The most exposed data is each link's last CB page, tile row 4l+3 = rows 128l+96..128l+127.
  - Links 0-2 cover rows 0..383, so valid rows can be hit, e.g. rows 96-127 and 224-255. A stale valid row reaches row 299 through the attention layers that follow.
- **Not affected:**
  - **Single-packet all-gathers.** The decode sampler's [32,32] all-gather sends packet 0 on the same route as DONE.
  - **The unicast factory** (S ≥ 1024). It counts per-chunk `data_valid` credits from one upstream neighbour (`kernels/unicast_reader.cpp:110-111,141-143`), so completion means every chunk was counted.
  - **The decode `all_reduce_async`.** Its fused write+increment travels on the same path.

### 3.3 Fit to the observations

| Observation | H1 |
|---|---|
| Intermittent, about 6% per prefill | Explained: a timing race on each of 80 all-gathers per replay, × 4 chips. |
| Same wrong values recur bit-exactly across slots and processes | Explained: stale contents at fixed trace addresses are deterministic; a few late-packet patterns dominate. |
| Large diffs with argmax flips | Explained: whole stale 32×32 tiles in the residual, amplified by MoE top-k and by cross-chip routing disagreement (§2). |
| `synchronize_device` before prefill doesn't help | Explained: the race is inside a single all-gather, after its start barrier. |
| Rate scales with layer count; small targets 0/149 | Explained: exposure is proportional to the number of all-reduces. |
| Decode greedy and earlier determinism tests pass | Explained: the decode path uses `all_reduce_async` and the safe single-packet all-gather. Eager tests have host-dispatch gaps that hide the race (prediction). |
| Traced prefill specifically | Consistent: trace replay has the smallest op-to-op gap. Eager prefill should show a lower rate, possibly not zero. |
| 1/149 without decode vs 9-10/149 with decode, even after sync | **Not explained by a proven code path.** No code path carries decode state into the S=512 trace. Candidate mechanisms (timing, or decode overwriting trace-scratch bytes so silent races become visible) are in §4. |

---

## 4. Unresolved: why decode traffic raises the rate

The S=512 prefill trace shares nothing that decode writes, so decode can only change *timing or physical state*:
- **No semaphores, scratch or programs are shared.** Decode `all_reduce_async` uses tt_ccl handles and persistent L1 scratch; the prefill uses program-owned CCL semaphores.
- **No post-capture buffer is read** (§6).
- **Stale contents don't change the outcome.** H1's stale reads land on prefill-written addresses, so decode content cannot alter *which* wrong value appears. That matches the bit-exact recurrence even with decode traffic.

Plausible but unproven explanations:
- skew between chips at trace start;
- fabric router or DRAM state after the decode CCL;
- clock or thermal state;
- **visibility rather than rate.**
  - A late packet is only visible if the stale bytes at its target differ from the correct ones.
  - With back-to-back identical prefills, a stale tile may still hold the *previous replay's* identical value, so the race is silent.
  - Decode traces reuse the same trace-scratch DRAM range; the generator documents this cross-trace address reuse for `logits[b]` (`AP/tt/generator.py:396-398`). Decode can therefore overwrite those bytes with different data and expose races that were silent.
  - This fits the mix in the decode runs: of 9 and 10 mismatches, 7 and 8 values respectively are distinct (decode-dependent stale data), while a few recur (stale data written earlier in the same replay).
  - Unproven: it needs the actual DRAM address map of the S=512 and bucket-4 traces.

A second, decode-coupled mechanism is not ruled out. Experiment 5.2 would show whether all divergences start at an all-gather.

---

## 5. Cheapest discriminating experiments

1. **Bucket contrast, no code change.** Rerun `diag_determinism_stress.py` with prompt lengths in different buckets, using at least 300 iterations each.
   - **Prediction:** ≈0 mismatches at 900 tokens (bucket 1024: unicast all-gather, same MoE dispatch, SDPA and ShortConv paths) and at ≤32 tokens (decode-path all-reduce). A non-zero rate at 100, 200 and 300 tokens (buckets 128/256/512, multicast).
   - Bucket 1024 is the cleanest control. The reduce-scatter stays the same ring factory, MoE stays on token dispatch and SDPA stays first-chunk paged; the only CCL choice that changes is the all-gather factory.
2. **Signature probe.** Inside the captured trace, `ttnn.copy` each prefill all-reduce output (or each layer output) into preallocated per-layer DRAM buffers allocated before capture. After each replay, read every device separately.
   - **Prediction:** in mismatching runs the 4 chips' copies stop being bitwise identical *right after an all-gather*. The divergence sits on one chip Y, confined to hidden columns `512·((Y+2) mod 4) + [128,256) ∪ [384,512)`.
   - This detects the race even when the logits don't change, and it answers §4.
3. **One-line kernel A/B.** `axis_load_balance = false` (`all_gather_multicast_factory.cpp:123`).
   - **Prediction:** 0/149 at S=512, at a small all-gather perf cost.
4. **Eager contrast.** Complete the `LFM2_EAGER_PREFILL=1` run.
   - **Prediction:** a lower rate, not necessarily zero. This is supporting evidence only, not decisive.
5. **Notes on the requested switches.**
   - `LFM2_STABLE_TOPK` only changes the TTSampling local top-k (`AP/tt/generator.py:131`), and the `host_logits` stress path never runs the sampler. Not informative.
   - `LFM2_POLICY_OVERRIDE` / `LFM2_PLAN_OVERRIDE` **do not exist** in this tree. Overrides go through `LFM2_PRECISION_CONFIG=<json>` with `policy_overrides` / `plan_overrides` (`AP/tt/precision.py:110-119`).

---

## 6. Areas audited and cleared (no mechanism for valid-row nondeterminism)

- **Reduce-scatter (ring, S=512).**
  - Every increment matches its wait, and readers reset to 0 only after consuming all credits.
  - The pairwise start barrier plus causality makes back-to-back reuse of the program-owned semaphores safe.
  - Chunk parity is computed from the tile index and the add order is fixed, so reduction is deterministic.
  - Semaphores are created on a program-cache miss during `prepare()` and owned by the cached program, so they are never freed or reallocated. Captures forbid cache misses (`AP/tt/generator.py:563-573`).
- **Decode CCL and tt_ccl** (`models/common/modules/tt_ccl.py:66-95`, `all_reduce_async` prim).
  - The parity of `_scratch_idx` and `barrier_semaphore_idx[1]` holds at every capture and replay boundary: `prepare()` makes 640 calls, each trace with all-reduces has 80, and traces with none don't matter.
  - Two consecutive all-reduces never share a handle or scratch.
  - The prefill S=512 trace shares no CCL state with decode.
- **Chunked SDPA and paged cache for valid rows.**
  - The runtime chunk start is 0, so there is one k-chunk of 512 keys, read only through page-table entries 0..7.
  - The in-kernel mask is a true −inf (`0xFF80`), and the accurate exp maps −inf to exactly 0.
  - Work assignment is static, with no cross-core reduction.
  - Every key visible to rows 0..299 comes from blocks this prefill just wrote, one writer per tile.
- **MoE dispatch** (`MCMoE._dispatch_forward`, `AP/tt/multichip_decoder.py:600-690`).
  - Each op is deterministic for identical inputs: topk is single-core per tile row and compares full fp32; all integer arithmetic stays exact and ≤ 3040.
  - `unified_routed_expert_moe` is strictly row-independent with respect to counts and region layout. A token's row moves only by whole tiles, and K-blocking, fidelity and accumulation order are compile-time.
  - So padding rows cannot change row 299's MoE output.
- **ShortConv, dense MLP, norms, embeddings, the `'last'` head, `pf_logits` accumulation.**
  - These are causal or row-local, or exact one-hot selects.
  - The first-chunk carry is `0 · state`, which is exact while the state is finite.
- **Input refresh vs replay.**
  - `copy_host_to_device_tensor` writes wait on `expected_num_workers_completed` (`tt_metal/impl/buffers/dispatch.cpp:729-746`), so they are ordered after earlier non-blocking replays.
  - All 11 input keys are refreshed, and their addresses are stable.
- **Trace / L1 allocation.**
  - CB-vs-L1 validation runs on every enqueue, including capture (`tt_metal/impl/program/program.cpp:2439-2551,3320-3384`).
  - No persistent L1 is allocated after capture: `_CONCAT_BUFS` are created during `prepare()`.
  - The only buffer created inside a capture is the acknowledged decode `logits[b]`, which the prefill never reads.
- **Prefill matmul configs at S=512.** All DRAM-sharded weights use full-width output blocks, so the known partial-block bug (§7.2) is not triggered.

---

## 7. Other potential issues (real, but not the cause of this symptom)

1. **`paged_fill_cache` write-write race on scratch block 0.** This is a proven race, but it cannot reach row 299.
   - **The race.** Bucket-512 padding maps virtual blocks 5..7 (positions 320..511) to physical block 0 through the chunk page table (`AP/tt/model.py:462-466`). Three cores then write different K/V into the same block-0 tiles at once, and the last write wins (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/writer_fill_cache_interleaved.cpp:21-38`).
   - **Why it stays contained.** It only perturbs attention for *padding* query rows. Valid rows mask those keys (§6), and nothing downstream couples padding rows into valid rows (§6, MoE).
   - **The model's guarantee rests on this.** The docstring claim "padded rows ... never feed valid rows" (`AP/tt/model.py:570-574`) depends on that row-independence and on block-0 data staying finite.
   - **Cheap fix.** Put `SKIP_PAGE_TABLE_ENTRY = -1` in the padded `cpt` entries; the writer supports it at `:15,31-33`.
2. **DRAM-sharded 2D-mcast matmul with `out_block_w < per_core_N` is wrong and racy.** This is the root cause of the developers' R2-8 workaround.
   - **The bug.** In `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:394-466`, the DRAM read offset is reset on every width block (the comment there says "bw is always 1 for DRAM-sharded senders"). Each K-row also reads `per_core_N` tiles into an `out_block_w`-tile row pitch.
     - So block `bw ≥ 1` re-reads block 0's columns.
     - Rows overlap in L1, and the last row overruns into the next CB slot, which races with compute.
   - Nothing validates the assumption.
   - **Latent trap.** The fallback `(legal_bw[-1], 1, 1)` in `AP/tt/optimized_decoder.py:358` (and in v1 at `:319`) would silently choose `out_block_w=1` for a DRAM-sharded weight if no full-width block fits. No current shape hits it.
   - **Suggested fix.** Add `TT_FATAL(out_block_w == per_core_N)` for DRAM-sharded in1, or fix the column offset.
3. **`unified_routed_expert_moe` ignores the requested math fidelity.** It hard-codes `MathFidelity::LoFi` (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp:1161`). The policy's `dispatch_fidelity="hifi2"` therefore has no effect. This is deterministic, but it affects precision and the conclusions of the datatype sweep.
4. **`unified_routed_expert_moe` has a NoC-1 worker multicast marked "FABRIC HAZARD, KNOWINGLY ACCEPTED"** (`:663-675`). The documented failure is a hang when it collides with concurrent fabric CCL traffic. It only overlaps fabric traffic here if late all-gather packets (H1) are still landing. Cheap A/B: `DS_NO_WRITER_MCAST=1`.
5. **Ring reduce-scatter semaphores sit in regular L1, not L1_SMALL.** `use_l1_small_for_semaphores` defaults to false (`ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_program_factory.cpp:35-47`), which contradicts the `mesh_device_params` comment (`AP/tt/multichip_decoder.py:1013-1015`). Harmless for correctness.
6. **The decode `all_reduce_async` alternation is correct but fragile.**
   - `reduction_receiver` sets the semaphore to 0 rather than decrementing it. A parity slip, such as an odd number of `MeshReduce.decode` calls before a capture, would silently corrupt data instead of hanging.
   - Suggested hardening: assert `_scratch_idx[0] == 0 and ccl.barrier_semaphore_idx[1] == 0` at each capture start.
7. **The ShortConv first-chunk carry depends on finite state.** It is `carry_sel(0) @ state` over all 32 slot rows (`AP/tt/fused_decoder.py:353-354`), so it stays exactly 0 only while every slot's state is finite.
8. **Attention-free targets crash.** `Lfm2Model.set_kv_cache` crashes when there are no attention layers (`AP/tt/model.py:313`), so 1-layer conv targets cannot be bisected through the adapter.

---

## 8. Candidate ledger

| Candidate | Code contradiction | Reached by the repro | Explains recurrence / sync / layers | Explains decode effect | Verdict |
|---|---|---|---|---|---|
| **H1: multicast all-gather DONE misses alternate-route packets** | Yes (§3.2) | Yes (tracy: 4-core multicast at [512,512]) | Yes / yes / yes | Unproven (§4) | **Headline** |
| `paged_fill_cache` block-0 race | Yes | Yes | Only for padding rows; MoE and attention isolate row 299 | No | Other |
| DRAM-sharded partial-block matmul | Yes | No (all S=512 configs are full-width) | — | — | Other |
| Decode CCL parity / tt_ccl leakage | No | — | — | — | Cleared |
| CB overlap / post-capture aliasing | No | — | — | — | Cleared |
| MoE dispatch nondeterminism or padding coupling | No | — | — | — | Cleared |
| SDPA stale/NaN keys for valid rows | No | — | — | — | Cleared |
| Input refresh vs replay ordering | No | — | — | — | Cleared |
