# Op issue drafts (stage 4) -- NOT filed upstream (filing is the user's decision)

## 1. deepseek_prefill.hybrid_routed_expert_moe: L1 arena assumes an empty L1
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp`
  `arena_bytes_for()` returns `l1_size_per_core - allocator base`; the op then allocates that arena across 88 banks.
- Any other L1 allocation makes it OOM: with `l1_small_size=24576` the bank is 24576 B short; with 0 the CCL global
  semaphores (2688 B / bank) suffice: "Not enough space to allocate 128578560 B L1 buffer across 88 banks ... allocated:
  2688 B" (doc/multichip_decoder/logs/dev_single_user.log, dev_l1small0.log).
- Expected: size the arena from the allocator's largest free L1 block (and pass that size to the program factory,
  which currently re-derives it after the arena is allocated).
- Impact here: multichip uses the unified op (stage 3 single chip: hybrid 3101 vs unified 3494 us per 2048-row MoE prefill).

## 2. sparse_matmul indexed mode: in0_block_w = K tiles nondeterministic inside a traced decoder
- x [1,1,32,2048] @ w [1,64,2048,768] BFP8, `indices` with 16 slots, 1D mcast (8,3,in0_block_w=64,1), per_core_N 1.
- Inside the traced multichip conv_moe bucket-4 decode: "traced decode replay is not deterministic" 2/2; in0_block_w
  32 / 8 pass 2/2 (logs/det_bisect.log).  The isolated op is bitwise stable (repro/sparse_mm_idx_bw64_determinism.py).
- Suspect: in0 CB (2 x 64 tiles = 256 KB / core) placement next to live L1 tensors inside the trace.

## 3. experimental.all_reduce_async: Blackhole rejects DRAM and interleaved inputs
- `all_reduce_async_device_operation.cpp:24-56`: TT_FATAL for Blackhole DRAM ("does not use an accessor") and for
  non-WIDTH_SHARDED input / buffer / output.  Forces decode-only use; prefill uses the composite all_reduce (RS + AG).
  (sweeps/ccl_topology_probe.log)
