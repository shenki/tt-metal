# LFM2.5-24B-A2B — optimized decoder (stage 3)

`tt/optimized_decoder.py`: `OptimizedDecoder` = stage-2 `FusedDecoder` graph + an explicit per-group
`Policy` (weight dtypes, fidelities, KV dtype, program configs, data placement, MoE execution path).
`DEFAULT_POLICY` is the measured selection; `BASELINE_POLICY` reproduces the fused decoder exactly
and is the "before" in every table. One Blackhole p150b (1x1 mesh). Same prefill/decode contract as
stages 1–2, plus two optional static decode arguments:

```python
dec, rope = make_optimized(sd, cfg, args, layer_idx, mesh_device, max_seq_len)   # DEFAULT_POLICY + DEFAULT_FUSIONS
out = dec.decode_forward(x, rope=..., rope_cos=..., rope_sin=..., cur_pos=..., page_table=..., slot_active=...,
                         active_slots=b,        # optional static bucket: only slots 0..b-1 active (one trace per bucket)
                         slot_mask=mask_bh)     # optional slot_active broadcast to [1, 1, 32, H], built once per step
# with a bucket and dec.bucketed_rope: cos/sin for the b active users, rope.decode_cos_sin(rope_pos, batch=b)
# (tests/harness.py::decode_step); bucket choice / check: decode_bucket(...), validate_decode_bucket(cur_pos, b)
```

Rounds 2-3 (after stage reviews 1-2) are documented in `work_log.md` § "Round 2" / "Round 3".
Raw Tracy `ops.csv` files and logs > 1 MB are stored xz-compressed (`*.xz`; read with `xzcat` / `xzgrep`).

## Results (Tracy device time, `before_after.txt`, `tracy_final_r3/<kind>/b<B>/<tag>/`)

| kind | decode b1, bucket 1 μs | b1, no bucket | b4, bucket 4 | b32 | prefill 2048 μs |
|---|---|---|---|---|---|
| attn_moe | 6925 → 1113 (r1) → 419 (r2) → **348** | 1052 | 743 | 6141 → 2050 → 2033 → **1992** | 337 358 → 7 601 → **6 631** |
| conv_moe | 6057 → 548 → 325 → **300** | 497 | 664 | 6057 → 1619 → 1582 → **1567** | 355 500 → 7 081 → **6 192** |
| conv_dense | 685 → 333 → 311 → **311** | 311 | – | 685 → 332 → 311 → **310** | 5 102 → 4 391 → **3 688** |

Per-token decoder-stack estimate on this single chip (2 conv_dense + 28 conv_moe + 10 attn_moe), batch 1 with
bucket 1: 2·311 + 28·300 + 10·348 ≈ 12.5 ms (round 1: 27.1 ms; stage 2: ~240 ms); batch 32 ≈ 64.4 ms per step for
32 users. Prefill 2048 ≈ 2·3.69 + 28·6.19 + 10·6.63 ≈ 247 ms.

### Performance accounting (`repro/r3_roofline.py`, `sweeps/r3_roofline.txt`)
bytes/step = weights actually read with the final dtypes (taken from `DEFAULT_POLICY`: all BFP8, experts gate_up/down BFP8, router bf16) (decode experts: b·K compact slots for b ≤ 4, the expected
union over b users above) + KV read by SDPA at position ~2080; roofline at 512 GB/s; device = Tracy (`tracy_final_r3`);
host e2e = traced replay wall time of the same configuration (`sweeps/r3_final_wall.jsonl`, 50 replays):

| kind | b | bucket | MB/step | roofline μs | device μs | device/roofline | host e2e μs | host − device |
|---|---|---|---|---|---|---|---|---|
| attn_moe | 1 | 1 | 54.6 | 107 | 348.3 | 3.3 | 370.6 | 22.3 |
| attn_moe | 4 | 4 | 181.7 | 355 | 742.9 | 2.1 | 761.2 | 18.3 |
| attn_moe | 32 | – | 645.0 | 1260 | 1991.9 | 1.6 | 2021.0 | 29.1 |
| conv_moe | 1 | 1 | 59.0 | 115 | 299.7 | 2.6 | 327.6 | 27.9 |
| conv_moe | 4 | 4 | 179.3 | 350 | 664.0 | 1.9 | 692.1 | 28.1 |
| conv_moe | 32 | – | 579.2 | 1131 | 1567.1 | 1.4 | 1584.0 | 16.9 |
| conv_dense | 1 | – | 98.0 | 192 | 310.5 | 1.6 | 326.4 | 15.9 |
| conv_dense | 32 | – | 98.0 | 192 | 310.4 | 1.6 | 326.5 | 16.1 |

Host − device (13–29 μs) is the per-replay launch cost of a single-layer trace (the wall-clock bench issues 50
non-blocking `execute_trace` replays, then one synchronize, and divides); a full model replays one trace for all 40
layers, so it is paid once per token (full-model stage). Device − roofline at batch 1 is
op count, not bandwidth: ~90 ops per MoE layer at a ~1–3 μs floor each (routing 19 ops ≈ 67 μs, attention plumbing
≈ 32 μs, conv elementwise), single-core fp32 TopK 15.8 μs, SDPA 48 μs at 2K context, and sparse expert matmuls at
~60–68 % of DRAM bandwidth (`work_log.md` R2-2, R3-1, R3-2 for every op-level candidate tried).

## Correctness (real weights unless noted)

Final code (md5s in `logs/r3_final_code_md5.txt`), `TT_METAL_TRACE_ALLOC_TRACKING=1` with program-cache allocations
included, `logs/r3_final_*.log`: **96/96** in `test_optimized_decoder.py` — single user 1/31/32/64/65/96/97/300@128/300/2049;
batch 32; slot reuse; increasing lengths with ONE decode trace and prefill between replays; full 131072 context with
and without bucket; bucketed decode b1/b2/b4 (idle slot inside the bucket)/b8/b16; **several bucket traces at once**
(`test_optimized_multi_bucket_traces`); router near-tie control; deep layers on real hidden states; perf / determinism /
no-host-fallback at b32, b1, b1 bucket 1, b4 bucket 4. Synthetic weights 72/72 (`logs/r3_final_synthetic.log`). Stage-1/2
suites after the harness changes: functional 36, fused + perf-determinism 42 (`logs/r3_final_*_regression.log`).
Watcher (`TT_METAL_WATCHER=10`, nothing disabled; unified expert FFN, see limitation 3): 39 tests, 0 errors (`logs/watcher_r3/`).

| layer (kind) | min output PCC prefill / decode | min update (out − x) PCC prefill / decode |
|---|---|---|
| 0 (conv_dense) | 0.99962 / 0.99924 | 0.99912 / 0.99982 |
| 3 (conv_moe) | 0.99981 / 0.99831 | 0.99208 / 0.99972 |
| 2 (attn_moe) | 0.99969 / 0.99930 | 0.99665 / 0.99235 |
| 1, 6, 7, 15, 23, 31, 38, 39 (real hidden states) | ≥ 0.99971 / ≥ 0.99858 | ≥ 0.99770 / ≥ 0.99724 |

(Minima over every user of every real-weight test in `logs/r3_final_*.log`.) Bars: output 0.995, update 0.99. Full context
(131069-token prompt): attn_moe prefill update PCC 0.9976 (round 1: 0.9906; cause: the prefill SDPA k chunk, R2-9).

Test-harness facts (`work_log.md` R2-10/R2-11):
- **Router near-ties**: when the HF fp32 biased scores of the 4th/5th expert differ by < 1e-3 at a decode token, bf16
  arithmetic may pick the other expert (the stage-1 functional decoder does the same). The update bar then also accepts HF
  with exactly that one swap; every alternative tried is logged as a WARNING with the gap and both PCCs; the output bar is
  always against plain HF; explicit control `test_router_near_tie_control`.
- **Deep layers need real inputs**: embedding-row inputs are out of distribution beyond the first layers (the functional
  decoder fails layer 39 on them); `tests/gen_real_hidden.py` builds real hidden states (generated on demand, sha256 logged).

## Cumulative contract (final default)

| contract | value | evidence |
|---|---|---|
| projection topology | packed wqkv → `nlp_create_qkv_heads_decode` straight from the DRAM-sharded qkv output (b-user split); conv in_proj packed B/C/x; dense w1, w3 separate | stage 2, R3-2 |
| precision | all projections and experts BFP8; HiFi2 (router HiFi4 bf16 → fp32, SDPA decode HiFi4, prefill SDPA HiFi2, dispatch LoFi). Real-input LoFi/BFP4 candidates deferred to the datatype sweep (R3-3) | `dtype_policy_rows.txt`, R2-6, R3-3 |
| attention decode | DRAM-sharded qkv 8/8/2 and wo 8/8/2; b-user head split, norm, RoPE; fused K/V paged update; SDPA on page-table rows < b (k_chunk 512, exact exp); pad-free concat into a zeroed buffer; sharded residual add | R2-1, R3-2 |
| conv weights | DRAM-sharded (in 32/2/3, out 8/8/2); state update with a full-width slot mask | R2-5 |
| dense MLP | decode: DRAM-sharded w1/w3 (32/2/3, N 11776 → 12288 zero-padded) + K-padded DRAM-sharded w2 (48/8/2); prefill: interleaved w1/w3 + DRAM-sharded w2 (46/8/2) | R2-6 |
| router / routing | DRAM-sharded router with fused sigmoid on the ffn_norm's sharded output; `moe_route_v3` (bitwise-equal, broadcast-free selection from topk indices, exact 0/1 matmuls) | R2-3, R3-1 |
| experts (decode) | buckets ≤ 4: compact `sparse_matmul(indices=)` over the 4b selected experts; else 64-wide union-of-active-users sparsity; L1 intermediates | R2-2 |
| experts (prefill) | token dispatch + `hybrid_routed_expert_moe` (LoFi, threshold 128) for chunks ≥ 96 rows (multi-core rank scans, 32-row split scatter, direct untilize); tuned sparse below | R2-7 |
| prefill matmuls / SDPA | `prefill_mm_v2`; DRAM-sharded weights only with full-width output blocks (op bug, R2-8); chunked SDPA q 64 / k 512 HiFi2 | R2-7–R2-9 |
| KV cache | BFP8 paged, block 64 | full-context tests |
| decode buckets (serving) | `active_slots` = `decode_bucket(active slots)` from (1, 2, 4, 8, 16, 32): smallest bucket covering the highest active slot; new requests take the lowest free slot (no state moves; holes allowed); `validate_decode_bucket` rejects violations; one trace per bucket, each trace output acknowledged as corruptible by the other traces (reviewed invariant) | R2-1, R3-4 |
| logical batch vs padding | 32 slots = 32 tile rows; idle slots cur_pos −1 / slot_active 0, excluded from routing | |

## Operation-topology audit (decode, batch 1 bucket 1; `tracy_final_r3/*/b1/final_bucket1`, μs per layer)
- attn_moe 348.3: experts 112.5 (compact gate_up + down), SDPA 47.8, data movement 46.9 (incl. per-step cos/sin 14.5 that a
  full model computes once for all 10 attention layers), elementwise 41.2, matmuls 55.8 (qkv 20.8, wo 14.4, router 10.3,
  routing 0/1 matmuls 10.3), norms 20.1, top-k 15.8, fused KV update 8.3.
- conv_moe 299.7: experts 112.3, elementwise 63.9 (ShortConv taps/state, routing), matmuls 58.8 (in 33.3, out 14.7,
  router 10.5, routing 10.3), data movement 27.0, top-k 15.8, norms 12.0.
- conv_dense 310.5: matmuls 240.3 (w1+w3 133.3, w2 59.1, conv in 33.5, out 14.4), elementwise 35.3, data movement 22.8,
  norms 12.1.
tt-perf-report still flags these matmul rows SLOW (final reports `tracy_final_r3/*/*/*/{decode,prefill}_perf_report.txt`);
dispositions:
- decode router 32×2048×256 (10 μs, 19.8 % DRAM BW): N padded 64 → 256 so the DRAM-sharded op gets one tile per bank,
  bf16 weights for exact top-k, HiFi4 fp32 accumulate; net −28 μs per MoE layer vs the interleaved router (R2-3). The
  low % is the 4× N padding plus one-tile-per-bank latency.
- decode routing 0/1 matmuls (32×32×256, 32×256×64, 32×64×256, 32×256×32; 2–3 μs each): latency-floor ops of the
  routing glue; each alternative tried in R3-1 was slower.
- prefill qkv / wo / conv_in / conv_out (54–57 % of HiFi2 peak) and dense w1/w3 (2048×2048×11776, ~50 %, 1×1 output
  subblock), advice "place input 0 in L1": measured in `repro/r3_prefill_mm_l1.py` → `sweeps/r3_prefill_mm_l1.log`
  (traced replay, live configs). Input 0 in L1 saves ≤ 2.6 μs (≤ 2.2 %) on every row, while an 8 MB L1 activation
  would collide with the hybrid MoE dispatch, which claims all of L1 (limitation 3), so inputs stay in DRAM. w1/w3:
  the best ≥ 2×1-subblock geometry (grid 11×8, 4×17 blocks) gives 646 vs 667 μs (−3 %). Padding N to 12288 (the
  decode DS layout) gives 627 μs (−6 %), but the prefill w2 would also need K padded. That is ~0.16 ms of a ~247 ms
  2048-token prefill. Deferred: the full-model stage can drop the separate prefill w1/w3/w2 copies by running prefill
  on the padded decode weights.
- prefill ShortConv carry/tail selection matmuls (four 32×32×2048 per conv layer, 3 μs each, ~0.2 % of a chunk):
  latency-floor one-hot selection matmuls of the stage-2 conv design (`tt/fused_decoder.py` conv prefill).
- prefill routing matmuls (2048×2048×64 router, 256×256×64 / b=256 32×32×64 rank scans, 32×64×64): fp32-accumulating
  HiFi4 routing math sized by the 64-expert dimension, 0.1–0.8 % of prefill each (R2-7).
The "op-to-op gap" advice item is the start of the profiled window.
Round-by-round op tables and every rejected alternative: `work_log.md` (R2-1…R2-12, R3-1…R3-5).

## Named limitations (precise)
1. Bucketed decode needs the serving layer to hold one decode trace per bucket and to place new requests in the lowest free
   slot (`decode_bucket` / `validate_decode_bucket`); without a bucket (b = 32) single-user decode is 1052 / 497 μs
   (attn / conv MoE). Multi-trace coexistence evidence: `test_optimized_multi_bucket_traces` on the final code
   (`logs/r3_final_misc.log`, watcher `logs/watcher_r3/`). The full-model / vLLM stages must call
   `validate_decode_bucket` themselves (today only the test harness does), and must warm every layer before trace
   capture: `_concat_out_buf` and `_ROUTE_V3_CONSTS` are lazy process-global caches allocated on first use.
2. Expert weights held twice (decode stacked + prefill per-expert); dense layers hold decode-padded and prefill copies
   (~0.15 GB). Fits per estimate (`context_contract.json`), re-verify with all 40 layers in stages 4/6.
3. `hybrid_routed_expert_moe` exceeds the TENSIX kernel config buffer when watcher-instrumented (firmware constant), so the
   watcher runs use the unified expert FFN for MoE prefill; it also claims every L1 byte, so no persistent L1 constants can
   coexist with it (routing constants and the zero buffer live in DRAM).
4. Op issues (upstream candidates; repros in `repro/`): 2D-mcast matmul with a DRAM-width-sharded weight and
   `out_block_w < per_core_N` is nondeterministic (`r2_mm_dram_sharded_partial_block.py`); `sparse_matmul` zero-fills its
   expanded output (FILL 21.9 μs at b32) unless compact, which needs a static nnz; fp32 TopK on [32,64] runs on one core
   (15.8 μs); `rms_norm` rejects height-sharded input; SDPA decode cannot write sharded output for GQA; `fast_reduce_nc`
   loses precision on fp32; mixed BFP4/BFP8 expert weights rejected by the hybrid dispatch op; uint16 reshape unsupported.
5. Precision: no BFP4 group passes the stage tests; decode expert gate_up BFP4 + LoFi and attention LoFi pass on real
   hidden states but fail the embedding-input stage tests (layers 3 / 2), conv LoFi fails layer 0 — deferred to the
   datatype sweep by decision (R3-3), with the Policy fields in place.
6. Traced prefill / TTFT host overhead (eager prefill wall 8.3–8.4 ms attn_moe / 9.9–10.1 ms conv_moe vs 6.6 / 6.2 ms device per 2048 chunk, a ~575 μs host gap
   before the expert FFN) and the windowed conv tail compiling one program per tail window are full-model/serving items.

## Checklist (optimize skill)
- [x] Decoder path fully traced, no host fallbacks (b32, b1, bucket 1/4, several bucket traces at once)
- [x] Decode activations width-sharded in L1 across norm → projections (DRAM-sharded matmuls), attention residual sharded;
      MoE intermediates in L1; remaining interleaved hops measured (L1 residual no gain, R2-4)
- [x] Prefill DRAM interleaved with 2D configs (`prefill_mm_v2`), large-op advice addressed (R2-7)
- [x] Operation-topology audit (above; per-round op tables in work_log)
- [n/a] Multi-device topology families / CCL / fused matmul-CCL / persistent CCL buffers (single chip; stage 4)
- [x] Best-candidate comparison and final default reproduced (`before_after.txt` = final default runs)
- [x] Dtype/fidelity verified in measured rows (`dtype_policy_rows.txt`)
- [x] SDPA decode/prefill and composite ops (create_qkv_heads_decode, paged_fused_update_cache, rotary_embedding_hf, sparse_matmul, hybrid_routed_expert_moe)
- [x] Packed same-input projections kept where they win (wqkv, conv in_proj, gate_up packed vs split re-measured R2-2; dense w1/w3 separate, stage 2)
- [x] Explicit memory / program / compute-kernel configs on important ops (Policy; DramShardedLinear; sparse configs; SDPA configs)
- [x] Dominant matmul roles swept (grid, in0_block_w incl. large legal values, subblocks, readers) per dtype/fidelity (R2-2, R2-6, round-1 §4/§7/§9)
- [x] Decode fidelity swept per dominant group (HiFi2 vs LoFi: experts, attention, conv, dense; R2-6, R3-3) — LoFi/BFP4 candidates that pass on real inputs but fail the stage tests deferred to the datatype sweep by decision
- [x] Attention dtype/fidelity swept separately (BFP4 qkv / out / both, BFP8 LoFi; R2-6, R3-3)
- [x] Expert and dense BFP4/LoFi (gate_up and down separately; R2-2, R2-6, R3-3)
- [x] Shard specs / grids dividing dimensions (padding only where measured: dense N 12288, router N 256; out_mem matches the op's grid, R3-5)
- [x] DRAM-sharded decode matmuls with 1/2/3 readers per role (explicit per-role counts in DEFAULT_POLICY)
- [x] MoE: active-expert sparse_matmul, nnz handling measured, gate_up/down tuned separately, compact indexed path, no dense all-expert path, L1 intermediates
- [n/a] LM head / sampling (decoder stage; full-model stage)
- [x] Reduced-precision experiments on real weights and real input activations (R3-3); final frontier → datatype sweep
- [x] Performance accounting: roofline vs device vs host e2e from the same configurations (above)
- [x] Batch capability: batch 1 primary, 2/4/8/16 buckets and 32 tested
- [x] Watcher clean, trace allocation tracking, determinism, stress (slot reuse, increasing lengths, multi-bucket schedule)

## Commands
```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -v --timeout=4000 models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_optimized_decoder.py
python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.gen_real_hidden          # inputs for the deep-layer tests
LFM2_POLICY='{"expert_gate_up":"bfp4"}' pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_policy.py -k conv_moe   # A/B
LFM2_BENCH_BUCKET=1 LFM2_BENCH_MODE=perf_only LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_BATCH=1 \
  python -m tracy -r -p -v -m pytest -q models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_policy.py -k attn_moe
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=<dir> \
  LFM2_POLICY_OVERRIDE='{"dispatch_ffn":"unified","dispatch_fidelity":"hifi2"}' pytest ... -k "<selection>"
```
