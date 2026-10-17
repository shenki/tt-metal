# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for LiquidAI/LFM2.5-24B-A2B (stage 3: single-device optimization).

``OptimizedDecoder`` = ``FusedDecoder`` (stage 2 graph) + an explicit, per-tensor-group
``Policy``: weight dtypes, math fidelity, KV-cache dtype, decode routing of idle slots and
program configs.  The prefill/decode contract is the functional one (see
``functional_decoder``); decode additionally passes ``slot_active`` to every layer kind.
``DEFAULT_POLICY`` is the measured selection (``doc/optimized_decoder/README.md``).
"""

import math
from dataclasses import dataclass, replace

import ttnn

from .functional_decoder import DRAM, L1, TILE, _hifi4
from .fused_decoder import (
    DEFAULT_FUSIONS,
    FusedAttention,
    FusedDecoder,
    FusedDenseMLP,
    FusedMoE,
    FusedShortConv,
    _height_shard,
    _row_major_cores,
    decode_head_mem,
    make_rope,
)
from .model_config import Lfm2MoeArgs
from .weights import stacked_expert_weights

DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def _ckc(device, fidelity, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity={
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )


@dataclass(frozen=True)
class Policy:
    """Per-tensor-group precision / execution policy (strings so it serializes into reports)."""

    expert_gate_up: str = "bf16"
    expert_down: str = "bf16"
    expert_fidelity: str = "hifi2"
    # stacked (decode / sparse_matmul) expert copy only; "" = same as expert_gate_up / expert_down.
    # The per-expert prefill dispatch copy keeps expert_gate_up / expert_down (hybrid dispatch needs
    # one dtype for gate_up and down); expert_fidelity applies to the stacked copy only.
    expert_gate_up_decode: str = ""
    expert_down_decode: str = ""
    attn_qkv: str = "bf16"
    attn_out: str = "bf16"
    attn_fidelity: str = "hifi2"
    conv_in: str = "bf16"
    conv_out: str = "bf16"
    conv_fidelity: str = "hifi2"
    dense_w13: str = "bf16"
    dense_w2: str = "bf16"
    dense_fidelity: str = "hifi2"
    kv_cache: str = "bf16"
    sdpa_decode_k_chunk: int = 0  # 0: op default
    sdpa_decode_exp_approx: bool = False
    sdpa_decode_fidelity: str = "hifi4"
    sdpa_prefill_chunk: int = (
        256  # cap of the chunked-prefill SDPA q/k chunk (actual = largest power of 2 dividing start and S_pad)
    )
    mask_idle_routing: bool = True  # decode: idle slots (cur_pos < 0) select no experts
    # sparse_matmul 1D-mcast configs per role: (grid_x, grid_y, in0_block_w, out_subblock_w); None = stage-2 helper
    moe_gate_up_cfg: tuple = None
    moe_down_cfg: tuple = None
    moe_gate_up_prefill_cfg: tuple = None
    moe_down_prefill_cfg: tuple = None
    moe_decode_l1: bool = False  # decode MoE intermediates (expert outputs, swiglu, combine) in L1 instead of DRAM
    # DRAM-sharded decode matmuls per role: (input/output width-shard cores, in0_block_w, readers per DRAM bank);
    # None = interleaved weight (stage-2 path).  Prefill uses the same sharded weight with a 2D mcast config.
    ds_wqkv: tuple = None
    ds_wo: tuple = None
    ds_conv_in: tuple = None
    ds_conv_out: tuple = None
    ds_dense_w2: tuple = None
    # DRAM-sharded dense w1/w3 (cores, in0_block_w, readers); N zero-padded to a multiple of
    # banks*readers tiles, kept padded through silu*mul (inert zeros: silu(0)*0 = 0) and consumed by
    # the DRAM-sharded w2 whose K is zero-padded to match (requires ds_dense_w2)
    ds_dense_w13: tuple = None
    ds_dense_w2_decode: tuple = None  # decode w2 config with ds_dense_w13 (K padded); prefill uses ds_dense_w2
    # decode RMSNorms width-sharded in L1 (operator_norm output already in the next DRAM-sharded
    # projection's input layout; ffn_norm sharded on this many cores then back to interleaved)
    sharded_decode_norm: bool = False
    dense_prefill_2d: bool = False
    # router matmul on bf16 operands (the checkpoint router weight is bf16 and x is bf16, so every
    # product is exact in fp32) with fp32 accumulation/output, instead of fp32 x fp32
    router_bf16: bool = False
    # decode router as router_ksplit batched K-slices (more cores than the 2 output tiles allow) + sum
    router_ksplit: int = 0
    conv_prefill_windowed_tail: bool = False  # ShortConv prefill tail rows selected from a <= 64-row window
    dispatch_ffn: str = "unified"  # "unified" | "hybrid" (hybrid_routed_expert_moe: fused kernel for low-count experts)
    dispatch_hybrid_threshold: int = 320
    dispatch_fidelity: str = "hifi2"  # explicit 2D mcast program configs for the dense MLP w1/w3 prefill matmuls
    # prefill MoE via token dispatch (expert-sorted buffer + deepseek_prefill.unified_routed_expert_moe)
    # for chunks of at least this many rows; 0 = off (sparse_matmul over 32-row tiles)
    moe_dispatch_min_rows: int = 0
    moe_dispatch_max_rows: int = 2048  # static constants are sized for chunks up to this many rows
    ffn_norm_cores: int = 8
    # ---- prefill round 2 (review P2 #4 / P5) ----
    sdpa_prefill_k_chunk: int = 0  # cap of the chunked-prefill SDPA k chunk (power of 2 dividing start); 0 = q chunk
    sdpa_prefill_fidelity: str = "hifi4"  # chunked-prefill SDPA only (qk-norm / rope keep HiFi4)
    moe_scan_multicore: bool = False  # rank-scan batched [32x32]@[32xE] matmuls with a batch-parallel program config
    moe_scatter_rows: int = 1  # dispatch source-map scatter split over this many rows (cores), then summed
    moe_out_direct_untilize: bool = False  # BFP8 FFN output -> BF16 row-major in one untilize (no typecast)
    prefill_mm_v2: bool = False  # prefill_2d_config: dtype-aware L1 budget, full output block first, subblock w 3
    # decode MoE for a static small active-user bucket (active_slots <= this; 0 = off): the
    # bucket*K selected experts are computed compactly with sparse_matmul indexed (gather) mode
    # instead of the 64-expert-wide path; requires at most active_slots active slots
    moe_compact_max_users: int = 0
    moe_gate_up_compact_cfg: tuple = None  # (grid_x, grid_y, in0_block_w, out_subblock_w); None = moe_gate_up_cfg
    moe_down_compact_cfg: tuple = None  # None = moe_down_cfg
    # decode router as a DRAM-sharded matmul (N padded 64 -> 256) reading the ffn_norm's width-sharded
    # output directly (no reshape/permute K-split); value = cores (must equal ffn_norm_cores)
    router_ds_cores: int = 0
    router_kth_slice: bool = False  # k-th largest biased score = last column of the sorted topk values
    decode_residual_l1: bool = False  # decode residual adds (layer outputs) in L1 instead of DRAM
    # decode routing glue without column-broadcast / single-core FillPad ops (T = 32 rows): selection
    # mask from the topk indices as exact 0/1 matmuls + full-width eq (same expert set as the k-th
    # value threshold), full-width bias / eps / idle-slot mask, fused bf16 division; compact path
    # (buckets): ids straight from the topk indices of rows 0..b-1 (the bucket contract: only slots
    # < b are active) and weights from exact 0/1 matmuls.  Weights bitwise equal to the current path.
    moe_route_v3: bool = False
    # moe_route_v3 constants in L1 (index constants shared by all layers).  Must stay off with
    # moe_dispatch_min_rows: the hybrid prefill expert op allocates every L1 bank in full (OOM)
    moe_route_v3_l1: bool = False
    # DS decode router: sigmoid fused into the DRAM-sharded matmul and the E logit columns sliced
    # straight from its width-sharded output (no sharded->interleaved copy, no separate sigmoid)
    router_ds_fused: bool = False
    # ---- attention decode plumbing (stage 3 round 3) ----
    # nlp_create_qkv_heads_decode reads the DRAM-sharded qkv output (width-sharded L1) directly
    # (no sharded->interleaved conversion; the interleaved-input kernel is 15 us vs ~3); needs ds_wqkv
    attn_heads_sharded: bool = False
    # paged_fused_update_cache (K and V in one op, V resharded to a disjoint grid) for decode buckets
    # b <= this (0 = two paged_update_cache; 32 = always)
    attn_kv_fused_max_bucket: int = 0
    # bucketed decode: concat the b-user SDPA output straight into a zeroed [32, NH*HD] L1 buffer
    # (copied from a DRAM zeros tensor; the op writes rows < b only, rows >= b stay 0) instead of
    # ttnn.pad back to 32 users; the concat output is resharded directly (sharded -> sharded) into
    # the out-projection input layout
    attn_pad_free: bool = False
    # bucketed decode: split heads for the b active users only (nlp_create_qkv_heads_decode
    # slice_size=b), so q/k-norm, RoPE, the KV update and SDPA run on b-user tensors; needs b-user
    # cos/sin from the caller (``rope.decode_cos_sin(rope_pos, batch=b)`` when
    # ``decoder.bucketed_rope``), else the 32-user split is used
    attn_bucket_heads: bool = False
    # decode attention residual on the out-projection's width-sharded output: x is resharded once
    # (operator-norm input layout = out-proj output layout = ffn-norm input layout), h = x + attn is
    # a sharded add consumed by the sharded ffn norm (no S2I + DRAM add + I2S); MoE ffn only
    attn_residual_sharded: bool = False
    # attn_pad_free concat output kept as a persistent L1 buffer per (bucket rows, layout) instead of a per-call
    # copy of DRAM zeros: rows >= b of a bucket's buffer are never written, so they stay 0 (stage 5)
    attn_concat_persistent: bool = False
    # (max_bucket, (grid_x, grid_y, in0_block_w, out_subblock_w)): decode down-projection config of the 64-wide expert
    # path for static buckets <= max_bucket (fewer active experts than the b32 tuning point); None = moe_down_cfg
    moe_down_small_bucket_cfg: tuple = None
    # compact decode experts: apply the per-slot routing weight to the [n, 32, I] activation before the down projection
    # (linear, so only the rounding order changes) instead of to the [n, 32, H] output (H / I x fewer elements)
    moe_compact_weight_before_down: bool = False
    # compact decode experts: the expert-sum reduction writes straight into this width-sharded layout (multichip:
    # the all-reduce / residual layout) instead of L1 interleaved; set by the decoder that owns the layout
    moe_compact_out_sharded: bool = False

    def summary(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


BASELINE_POLICY = Policy(mask_idle_routing=False)  # == fused decoder
DEFAULT_POLICY = Policy(
    kv_cache="bfp8",
    sdpa_decode_k_chunk=512,
    expert_gate_up="bfp8",
    expert_down="bfp8",
    # Precision candidates deferred to the datatype-sweep stage (decision 2026-10-17; work_log R3-3):
    # expert_gate_up_decode="bfp4" + expert_fidelity="lofi" and attn_fidelity="lofi" pass on real hidden
    # states (sweeps/r3_bfp4_real_*.jsonl) but fail the stage tests' embedding-row inputs (L3 / L2);
    # conv_fidelity="lofi" fails layer 0 on its real (embedding) input (sweeps/r3_precision_bisect.jsonl).
    moe_gate_up_cfg=(8, 8, 8, 2),
    moe_down_cfg=(8, 6, 16, 2),
    moe_decode_l1=True,
    attn_qkv="bfp8",
    attn_out="bfp8",
    conv_in="bfp8",
    conv_out="bfp8",
    dense_w13="bfp8",
    dense_w2="bfp8",
    ds_wqkv=(8, 8, 2),
    ds_wo=(8, 8, 2),
    ds_conv_in=(32, 2, 3),
    ds_conv_out=(8, 8, 2),
    ds_dense_w2=(46, 8, 2),  # prefill (and the unpadded decode path)
    ds_dense_w13=(32, 2, 3),  # decode w1/w3: N 11776 -> 12288 inert padding, 3 readers (sweeps/r2_bfp4_cand.jsonl)
    ds_dense_w2_decode=(48, 8, 2),  # decode w2, K padded to 12288 to consume the padded w1/w3 output
    sharded_decode_norm=True,
    moe_gate_up_prefill_cfg=(8, 8, 8, 1),
    moe_down_prefill_cfg=(8, 6, 8, 2),
    moe_dispatch_min_rows=96,
    sdpa_prefill_chunk=64,
    sdpa_prefill_k_chunk=512,
    sdpa_prefill_fidelity="hifi2",
    dense_prefill_2d=True,
    router_bf16=True,
    router_ds_cores=8,  # replaces the router_ksplit=8 decode router (-28 us/MoE layer, same selection: sweeps/r2_router.jsonl)
    dispatch_ffn="hybrid",
    dispatch_fidelity="lofi",
    dispatch_hybrid_threshold=128,
    conv_prefill_windowed_tail=True,
    moe_scan_multicore=True,
    moe_scatter_rows=32,
    moe_out_direct_untilize=True,
    prefill_mm_v2=True,
    moe_compact_max_users=4,  # buckets 1/2/4 (b=8 loses: sweeps/r2_moe_bench.jsonl)
    moe_gate_up_compact_cfg=(8, 6, 8, 1),
    moe_down_compact_cfg=(8, 6, 48, 2),
    attn_heads_sharded=True,  # round 3 attention plumbing (sweeps/r3_attn_tracy): attn_moe b1 420 -> 375 us
    attn_kv_fused_max_bucket=32,
    attn_pad_free=True,
    attn_bucket_heads=True,
    attn_residual_sharded=True,
    moe_route_v3=True,  # bitwise-equal routing; conv_moe decode 325.4->299.4 (b1 bucket) / 706.1->664.0 (b4) / 1579.6->1568.2 us (b32) (sweeps/r3_route_tracy)
    router_ds_fused=True,  # bitwise-equal scores (fused sigmoid + slice from the sharded output)
)


def width_shard_layout(mesh_device, cores, width):
    """Width-sharded [32, width] L1 layout: a rectangular 8-wide grid when ``cores % 8 == 0``
    (so sharded norms can use the same grid), else row-major over the compute grid."""
    grid = mesh_device.compute_with_storage_grid_size()
    if cores % 8 == 0:
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, cores // 8 - 1))})
    else:
        crs = ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return ttnn.create_sharded_memory_config(
        (TILE, width),
        crs,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class ShardedNorm:
    """Decode RMSNorm on a width-sharded [32, H] L1 tensor (``cores`` cores, multiple of 8)."""

    def __init__(self, mesh_device, hidden, cores, eps, ckc):
        assert cores % 8 == 0 and (hidden // TILE) % cores == 0, (hidden, cores)
        self.mem = width_shard_layout(mesh_device, cores, hidden // cores)
        block_w = hidden // TILE // cores
        self.cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, cores // 8),
            subblock_w=max(w for w in (1, 2, 4) if block_w % w == 0),
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        self.eps, self.ckc = eps, ckc

    def __call__(self, x, weight):
        already = x.memory_config() == self.mem  # caller keeps ownership of an already-sharded input
        xs = x if already else ttnn.to_memory_config(x, self.mem)
        y = ttnn.rms_norm(
            xs,
            epsilon=self.eps,
            weight=weight,
            program_config=self.cfg,
            memory_config=self.mem,
            compute_kernel_config=self.ckc,
        )
        if not already:
            ttnn.deallocate(xs)
        return y


def prefill_2d_config(mesh_device, m, k, n, v2=False, w_tile_bytes=2048, dram_sharded_w=False, l1_budget=1_300_000):
    """2D-mcast matmul config for large-M prefill over the full compute grid (works with
    interleaved and DRAM-width-sharded weights)."""
    grid = mesh_device.compute_with_storage_grid_size()
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    gy, gx = min(grid.y, mt), min(grid.x, nt)
    per_core_m, per_core_n = math.ceil(mt / gy), math.ceil(nt / gx)
    if v2:
        return _prefill_2d_config_v2(
            gx, gy, kt, per_core_m, per_core_n, w_tile_bytes, full_w=dram_sharded_w, budget=l1_budget
        )
    # circular buffers must fit L1: double-buffered bf16 in0 / (<= bf16) in1 blocks, and the bf16
    # output block with its fp32 partials; shrink the output block (divisors of the per-core tile
    # counts) first, then in0_block_w
    budget = l1_budget
    divs = lambda v: [d for d in range(v, 0, -1) if v % d == 0]
    legal_bw = [b for b in (8, 4, 2, 1) if kt % b == 0]
    choice = None
    for bw in legal_bw:
        # DRAM-sharded weight: full-width output blocks only (see _prefill_2d_config_v2)
        for ob_w in [per_core_n] if dram_sharded_w else divs(per_core_n):
            for ob_h in divs(per_core_m):
                used = 2 * per_core_m * bw * 2048 + 2 * bw * per_core_n * 2048 + ob_h * ob_w * (2048 + 4096)
                if used <= budget:
                    choice = (bw, ob_h, ob_w)
                    break
            if choice:
                break
        if choice:
            break
    in0_bw, out_block_h, out_block_w = choice or (legal_bw[-1], 1, 1)
    sub_w = max(w for w in (1, 2, 4) if out_block_w % w == 0)
    sub_h = max(h for h in (1, 2, 4) if out_block_h % h == 0 and h * sub_w <= 4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _prefill_2d_config_v2(gx, gy, kt, per_core_m, per_core_n, w_tile_bytes, full_w=False, budget=1_300_000):
    """Largest output block first (fewer partial-output spills), then the largest in0_block_w that
    fits; in1 blocks sized by the weight dtype; subblock widths up to 3/4 (<= 4 tiles with fp32 acc).
    Measured (repro/r2_prefill_ops_bench.py mm, 2048 rows): qkv 230 -> 166 us, conv_in 553 -> 330 us."""
    divs = lambda v: [d for d in range(v, 0, -1) if v % d == 0]
    legal_bw = [b for b in (8, 4, 2, 1) if kt % b == 0]
    blocks = sorted(((h, w) for h in divs(per_core_m) for w in divs(per_core_n)), key=lambda hw: -hw[0] * hw[1])
    # DRAM-width-sharded weights: only full-width output blocks.  out_block_w < per_core_N with a
    # DRAM-sharded in1 gives run-to-run different results (interleaved in1 is fine):
    # doc/optimized_decoder/repro/r2_mm_dram_sharded_partial_block.py
    if full_w:
        blocks = [hw for hw in blocks if hw[1] == per_core_n]
    choice = None
    for ob_h, ob_w in blocks:
        for bw in legal_bw:
            used = 2 * per_core_m * bw * 2048 + 2 * bw * per_core_n * w_tile_bytes + ob_h * ob_w * (2048 + 4096)
            if used <= budget:
                choice = (bw, ob_h, ob_w)
                break
        if choice:
            break
    in0_bw, out_block_h, out_block_w = choice or (legal_bw[-1], 1, 1)
    sub_h, sub_w = max(
        (
            (h, w)
            for h in (1, 2, 4)
            for w in (1, 2, 3, 4)
            if out_block_h % h == 0 and out_block_w % w == 0 and h * w <= 4
        ),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


_TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576}


class DramShardedLinear:
    """``x @ W`` with ``W`` width-sharded over the DRAM banks.

    Decode (M = 32): activation resharded to a width-sharded L1 grid of ``cores`` cores,
    ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` (``in0_block_w``, readers per
    bank), output back to interleaved.  Prefill: the same weight with a 2D mcast config."""

    def __init__(self, mesh_device, w, dtype, cfg, prefill_v2=False, pad_n=False, k_pad=None):
        """``pad_n``: zero-pad N to a multiple of banks * readers tiles (output keeps the padding);
        ``k_pad``: zero-pad K (weight rows) to this size, e.g. to consume a padded producer."""
        self.mesh_device = mesh_device
        self.prefill_v2 = prefill_v2
        self.w_tile_bytes = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dtype]
        banks = mesh_device.dram_grid_size().x
        k0, n0 = w.shape
        unit = TILE * banks * cfg[2]
        k, n = (k_pad or k0), (math.ceil(n0 / unit) * unit if pad_n else n0)
        if (k, n) != (k0, n0):
            import torch

            wp = torch.zeros(k, n, dtype=w.dtype)
            wp[:k0, :n0] = w
            w = wp
        assert n % (TILE * banks) == 0, (n, banks)
        self.k, self.n = k, n
        dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        wmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weight = ttnn.from_torch(
            w.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=wmem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        cores, in0_bw, readers = cfg
        self.cores = cores
        width = lambda w_: width_shard_layout(mesh_device, cores, w_)
        assert (k // TILE) % cores == 0 and (k // TILE // cores) % in0_bw == 0, (k, cores, in0_bw)
        per_core_n = math.ceil(n / TILE / cores)
        self.in_mem = width(k // cores)
        # the op lays the output over ceil(N_tiles / per_core_N) cores row-major on the full grid
        # (whatever the input grid); describe exactly that layout (no "computed MemoryConfig" mismatch)
        grid = mesh_device.compute_with_storage_grid_size()
        out_cores = math.ceil(n / TILE / per_core_n)
        self.out_mem = ttnn.create_sharded_memory_config(
            (TILE, per_core_n * TILE),
            ttnn.num_cores_to_corerangeset(out_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_bw,
            per_core_M=1,
            per_core_N=per_core_n,
            fused_activation=None,
            num_workers_per_dram_bank=readers,
        )
        self._prefill_cfgs = {}

    def decode(self, x, ckc, mem=L1):
        already = x.memory_config() == self.in_mem  # e.g. produced by a ShardedNorm on the same grid
        xs = x if already else ttnn.to_memory_config(x, self.in_mem)
        o = ttnn.linear(
            xs, self.weight, program_config=self.decode_cfg, compute_kernel_config=ckc, memory_config=self.out_mem
        )
        if not already:
            ttnn.deallocate(xs)
        oi = ttnn.to_memory_config(o, mem)
        ttnn.deallocate(o)
        return oi

    def decode_sharded(self, x, ckc):
        """Decode matmul, output left in its width-sharded L1 layout (``out_mem``)."""
        already = x.memory_config() == self.in_mem
        xs = x if already else ttnn.to_memory_config(x, self.in_mem)
        o = ttnn.linear(
            xs, self.weight, program_config=self.decode_cfg, compute_kernel_config=ckc, memory_config=self.out_mem
        )
        if not already:
            ttnn.deallocate(xs)
        return o

    def prefill(self, x, ckc, mem=DRAM):
        m = x.shape[-2]
        if m not in self._prefill_cfgs:
            self._prefill_cfgs[m] = prefill_2d_config(
                self.mesh_device,
                m,
                self.k,
                self.n,
                self.prefill_v2,
                self.w_tile_bytes,
                dram_sharded_w=True,
                l1_budget=getattr(self, "prefill_l1_budget", 1_300_000),
            )
        return ttnn.linear(
            x, self.weight, program_config=self._prefill_cfgs[m], compute_kernel_config=ckc, memory_config=mem
        )

    def __call__(self, x, ckc, mem):
        return self.decode(x, ckc, mem) if x.shape[-2] <= TILE else self.prefill(x, ckc, mem)


def _add_residual(out, residual, policy=None):
    if residual is None:
        return out
    decode_l1 = policy is not None and policy.decode_residual_l1 and out.shape[-2] <= TILE
    h = ttnn.add(residual, out, memory_config=L1 if decode_l1 else DRAM)
    ttnn.deallocate(out)
    return h


_ZERO_ROWS = {}  # (device id, rows, NH*HD) -> (device, DRAM zeros [1, 1, rows, NH*HD]) shared by all layers
_CONCAT_BUFS = {}  # (device id, B, NH*HD, rows, layout) -> (device, persistent zeroed L1 concat output)


class OptimizedAttention(FusedAttention):
    def __init__(self, mesh_device, args, sd, max_batch_size, policy: Policy, fusions, sdpa_chunk=256):
        super().__init__(mesh_device, args, sd, max_batch_size, DTYPES[policy.attn_qkv], fusions, sdpa_chunk=sdpa_chunk)
        self.policy = policy
        self.r3_decode = bool(
            policy.attn_heads_sharded
            or policy.attn_kv_fused_max_bucket
            or policy.attn_pad_free
            or policy.attn_bucket_heads
        )
        if self.r3_decode:
            assert {"qkv_heads_decode", "rope_hf", "concat_heads_decode"} <= set(fusions)
            assert not policy.attn_heads_sharded or policy.ds_wqkv, "attn_heads_sharded needs ds_wqkv"
            assert not policy.attn_bucket_heads or policy.attn_heads_sharded, "slice_size needs the sharded qkv input"
        self._bmems = {}
        if policy.attn_bucket_heads:
            import torch

            self.batch_offset0 = ttnn.from_torch(  # nlp_create_qkv_heads_decode slice start (slot 0)
                torch.zeros(1, dtype=torch.int32),
                dtype=ttnn.uint32,
                device=mesh_device,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        if policy.attn_kv_fused_max_bucket:
            grid = mesh_device.compute_with_storage_grid_size()
            # paged_fused_update_cache needs K and V shards on disjoint core grids
            self.v_fused_mem = _height_shard(_row_major_cores(self.B, self.B, grid.x), (TILE, self.HD))
        if policy.attn_out != policy.attn_qkv:
            from .functional_decoder import _as_tt

            self.wo = _as_tt(sd["self_attn.out_proj.weight"].T, mesh_device, dtype=DTYPES[policy.attn_out])
        self.ckc = _ckc(mesh_device, policy.attn_fidelity)
        self.ckc_sdpa_decode = _ckc(mesh_device, policy.sdpa_decode_fidelity)
        self.ckc_sdpa_prefill = _ckc(mesh_device, policy.sdpa_prefill_fidelity)
        self.ds_qkv = self.ds_wo = None
        if policy.ds_wqkv:
            import torch

            wqkv = torch.cat(
                [sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"], sd["self_attn.v_proj.weight"]], dim=0
            ).T
            ttnn.deallocate(self.wqkv)
            self.ds_qkv = DramShardedLinear(
                mesh_device, wqkv, DTYPES[policy.attn_qkv], policy.ds_wqkv, policy.prefill_mm_v2
            )
        if policy.ds_wo:
            ttnn.deallocate(self.wo)
            self.ds_wo = DramShardedLinear(
                mesh_device,
                sd["self_attn.out_proj.weight"].T,
                DTYPES[policy.attn_out],
                policy.ds_wo,
                policy.prefill_mm_v2,
            )

    def prefill_forward(self, x, *, rope, chunk_start, page_table, chunk_page_table, chunk_start_tensor, residual=None):
        p = self.policy
        if not p.sdpa_prefill_k_chunk and p.sdpa_prefill_fidelity == "hifi4":
            return super().prefill_forward(
                x,
                rope=rope,
                chunk_start=chunk_start,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_tensor=chunk_start_tensor,
                residual=residual,
            )
        S = x.shape[-2]
        assert S % TILE == 0 and chunk_start % self.k_cache.shape[2] == 0, (S, chunk_start)
        qkv = self._qkv_linear(x, DRAM)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.NH, num_kv_heads=self.NKV, transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        q = self._qk_norm_f(q, "q", DRAM)
        k = self._qk_norm_f(k, "k", DRAM)
        cos, sin = rope.prefill_cos_sin(chunk_start, S)
        q = self._rope(rope, q, cos, sin, False)
        k = self._rope(rope, k, cos, sin, False)
        if k.dtype != self.k_cache.dtype:
            k = ttnn.typecast(k, self.k_cache.dtype)
            v = ttnn.typecast(v, self.v_cache.dtype)
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx=0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q_chunk = self.prefill_sdpa_chunk(chunk_start, S)
        # k chunk capped independently of q: few, large k chunks = fewer online-softmax rescales
        # (k128 costs update PCC 0.9906 at 131k context); power of two dividing chunk_start, no
        # larger than the attended length (short prompts would otherwise pay for a padded k chunk)
        k_chunk = p.sdpa_prefill_k_chunk or q_chunk
        while k_chunk > TILE and (chunk_start % k_chunk or k_chunk > chunk_start + S):
            k_chunk //= 2
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            page_table,
            chunk_start_idx_tensor=chunk_start_tensor,
            scale=self.scale,
            program_config=cfg,
            compute_kernel_config=self.ckc_sdpa_prefill,
        )
        ttnn.deallocate(q)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=DRAM)
        out = self._out_project(attn, residual)
        ttnn.deallocate(attn)
        return out

    def _qkv_linear(self, x, mem):
        if self.ds_qkv is None:
            return super()._qkv_linear(x, mem)
        return self.ds_qkv(x, self.ckc, mem)

    def _out_project(self, attn, residual):
        if self.ds_wo is None:
            return super()._out_project(attn, residual)
        return _add_residual(self.ds_wo(attn, self.ckc, DRAM), residual, self.policy)

    def _concat_out_buf(self, mem):
        """Zeroed L1 concat-heads output.  Not persistent in L1: the prefill hybrid routed-expert op
        claims the whole L1 bank as its arena, so a DRAM zeros tensor is copied in per call."""
        key = (id(self.mesh_device), self.B, self.NH * self.HD)  # shape too: decoders with other head counts
        hit = _ZERO_ROWS.get(key)
        if hit is None or hit[0] is not self.mesh_device:
            import torch

            t = ttnn.from_torch(
                torch.zeros(1, 1, self.B, self.NH * self.HD),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            hit = _ZERO_ROWS[key] = (self.mesh_device, t)
        return ttnn.to_memory_config(hit[1], mem)

    def _persistent_concat_buf(self, mem, rows):
        """Persistent zero-initialized L1 concat output for ``rows`` written rows (Policy.attn_concat_persistent);
        shared by the layers of a device, one per (rows, layout); allocated on first use (warm every bucket
        before trace capture, as for the other lazy per-device caches)."""
        key = (id(self.mesh_device), self.B, self.NH * self.HD, rows, repr(mem))
        hit = _CONCAT_BUFS.get(key)
        if hit is None or hit[0] is not self.mesh_device:
            hit = _CONCAT_BUFS[key] = (self.mesh_device, self._concat_out_buf(mem))
        return hit[1]

    def _bucket_mems(self, b):
        """Per-bucket b-user layouts: q/k/v heads (SDPA grid width), V for the fused KV update."""
        mems = self._bmems.get(b)
        if mems is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            mems = self._bmems[b] = (
                decode_head_mem(self.mesh_device, b, self.HD),
                _height_shard(_row_major_cores(b, b, grid.x), (TILE, self.HD)),
            )
        return mems

    def _decode_r3(self, x, rope_cos, rope_sin, rope, cur_pos, page_table, residual, b):
        p = self.policy
        B, NH, HD = x.shape[-2], self.NH, self.HD
        # b-user head split: only when the caller built b-user cos/sin (rope.decode_cos_sin(.., batch=b))
        nb = b if (p.attn_bucket_heads and b < B and rope_cos.shape[1] == b) else B
        head_mem, v_fused_mem = (
            (self.head_mem, getattr(self, "v_fused_mem", None)) if nb == B else self._bucket_mems(nb)
        )
        if p.attn_heads_sharded:
            qkv = self.ds_qkv.decode_sharded(x, self.ckc)  # width-sharded L1, whole heads per shard
        else:
            qkv = self._qkv_linear(x, L1)
        kw = {} if nb == B else dict(batch_offset=self.batch_offset0, slice_size=nb)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NH, num_kv_heads=self.NKV, memory_config=head_mem, **kw
        )  # [1, nb, 32(pad), HD] height-sharded, one user per core
        ttnn.deallocate(qkv)
        # per-head RMSNorm rejects height-sharded inputs: L1 interleaved round trip (on nb users)
        q = ttnn.to_memory_config(self._qk_norm_f(ttnn.to_memory_config(q, L1), "q", L1), head_mem)
        k = ttnn.to_memory_config(self._qk_norm_f(ttnn.to_memory_config(k, L1), "k", L1), head_mem)
        q = rope.apply_hf(q, rope_cos, rope_sin, True, self.ckc_sdpa)
        k = rope.apply_hf(k, rope_cos, rope_sin, True, self.ckc_sdpa)
        full_pos, full_pt = cur_pos, page_table
        # a caller may pass b-row page_table / cur_pos (sliced once per step for all layers)
        sliced = b < B and page_table.shape[0] > b
        assert page_table.shape[0] in (b, B) and (nb < B or page_table.shape[0] == B)
        if sliced:  # SDPA (and the b-user KV update) see rows < b only
            page_table = ttnn.slice(page_table, (0, 0), (b, page_table.shape[-1]))
            cur_pos = ttnn.slice(cur_pos, (0,), (b,))
        # 32-user k/v: the update takes all 32 indices (idle slots -1 are skipped)
        upd_pos, upd_pt = (cur_pos, page_table) if nb < B else (full_pos, full_pt)
        if b <= p.attn_kv_fused_max_bucket:
            v_d = ttnn.to_memory_config(v, v_fused_mem)
            ttnn.deallocate(v)
            v = v_d
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache, k, self.v_cache, v, update_idxs_tensor=upd_pos, page_table=upd_pt
            )
        else:
            ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=upd_pos, page_table=upd_pt)
            ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=upd_pos, page_table=upd_pt)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # q rows == page-table rows: no slicing / padding inside (q with 32 users reads rows < b)
        attn = self._sdpa_decode(q, cur_pos, page_table, pad=not p.attn_pad_free)  # [1, b, NH(32), HD] L1
        ttnn.deallocate(q)
        if sliced:
            ttnn.deallocate(page_table)
            ttnn.deallocate(cur_pos)
        if not p.attn_pad_free:
            assert residual is None or not residual.is_sharded(), "attn_residual_sharded needs attn_pad_free"
            return self._out_proj_decode(attn, B, residual)
        rows = attn.shape[1]
        # nlp_concat_heads_decode needs its input on a rectangular rows-core grid at (0, 0); it writes
        # output rows < rows only, so a zeroed output keeps the idle rows exactly 0
        cin = _height_shard(ttnn.CoreGrid(y=max(1, rows // 8), x=min(rows, 8)), (TILE, HD))
        attn_sh = ttnn.to_memory_config(attn, cin)
        ttnn.deallocate(attn)
        on_wo_grid = self.ds_wo is not None and self.ds_wo.in_mem == self._concat_mem()
        cmem = self.ds_wo.in_mem if on_wo_grid else self._concat_mem()
        if p.attn_concat_persistent:
            buf = self._persistent_concat_buf(cmem, rows)
        elif rows < B:
            buf = self._concat_out_buf(cmem)
        else:  # every row is written
            buf = ttnn.empty((1, 1, B, NH * HD), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, cmem)
        ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH, output_tensor=buf)
        ttnn.deallocate(attn_sh)
        if self.ds_wo is None:
            o = ttnn.to_memory_config(buf, L1)
            out = ttnn.linear(o, self.wo, compute_kernel_config=self.ckc, memory_config=DRAM)
            ttnn.deallocate(o)
        elif residual is not None and residual.is_sharded():
            out = self.ds_wo.decode_sharded(buf, self.ckc)  # sharded -> sharded reshard in
            if not p.attn_concat_persistent:
                ttnn.deallocate(buf)
            return self._sharded_residual_add(out, residual)
        else:
            out = self.ds_wo.decode(buf, self.ckc, DRAM)  # sharded -> sharded reshard (none on the wo grid)
        if not p.attn_concat_persistent:
            ttnn.deallocate(buf)
        return _add_residual(out, residual, p)

    def _sharded_residual_add(self, out, residual):
        assert out.memory_config() == residual.memory_config(), (out.memory_config(), residual.memory_config())
        h = ttnn.add(residual, out, memory_config=residual.memory_config())
        ttnn.deallocate(out)
        return h

    def _concat_mem(self):
        return width_shard_layout(self.mesh_device, self.NH, self.HD)  # 8 x 4 rectangle, one head per core

    def decode_forward(self, x, *, active_slots=None, **kw):
        if self.r3_decode:
            return self._decode_r3(
                x,
                kw["rope_cos"],
                kw["rope_sin"],
                kw["rope"],
                kw["cur_pos"],
                kw["page_table"],
                kw.get("residual"),
                active_slots or x.shape[-2],
            )
        # static per-trace bucket, consumed by _sdpa_decode (the stage-2 forward is reused unchanged)
        self._active_slots = active_slots
        try:
            return super().decode_forward(x, **kw)
        finally:
            self._active_slots = None

    def _sdpa_decode(self, q, cur_pos, page_table, pad=True):
        p = self.policy
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=0,
            k_chunk_size=p.sdpa_decode_k_chunk,
            exp_approx_mode=p.sdpa_decode_exp_approx,
        )
        B = q.shape[1]
        b = getattr(self, "_active_slots", None) or B
        # Bucketed decode: SDPA splits its cores over the page-table rows, so with only slots
        # 0..b-1 active it runs on b rows (Q stays the 32-user height-sharded tensor; the op reads
        # rows < b) and the idle rows are zero-padded back.  doc/optimized_decoder/sweeps/r2_sdpa_bucket*.log
        if b < B:
            page_table = ttnn.slice(page_table, (0, 0), (b, page_table.shape[-1]))
            cur_pos = ttnn.slice(cur_pos, (0,), (b,))
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=cfg,
            compute_kernel_config=self.ckc_sdpa_decode,
            memory_config=L1,
        )
        if b < B:
            ttnn.deallocate(page_table)
            ttnn.deallocate(cur_pos)
        if b < B and pad:
            padded = ttnn.pad(out, [(0, 0), (0, B - b), (0, 0), (0, 0)], 0.0)
            ttnn.deallocate(out)
            out = padded
        return out

    def allocate_kv_cache(self, num_blocks, block_size, dtype=None):
        super().allocate_kv_cache(num_blocks, block_size, dtype=dtype or DTYPES[self.policy.kv_cache])


class OptimizedShortConv(FusedShortConv):
    def __init__(self, mesh_device, args, sd, max_batch_size, policy: Policy, fusions):
        super().__init__(mesh_device, args, sd, max_batch_size, DTYPES[policy.conv_in], fusions)
        self.policy = policy
        if policy.conv_out != policy.conv_in:
            from .functional_decoder import _as_tt

            self.out_proj = _as_tt(sd["conv.out_proj.weight"].T, mesh_device, dtype=DTYPES[policy.conv_out])
        self.ckc = _ckc(mesh_device, policy.conv_fidelity)
        self.ds_in = self.ds_out = None
        if policy.ds_conv_in:
            ttnn.deallocate(self.in_proj)
            self.ds_in = DramShardedLinear(
                mesh_device,
                sd["conv.in_proj.weight"].T,
                DTYPES[policy.conv_in],
                policy.ds_conv_in,
                policy.prefill_mm_v2,
            )
        if policy.ds_conv_out:
            ttnn.deallocate(self.out_proj)
            self.ds_out = DramShardedLinear(
                mesh_device,
                sd["conv.out_proj.weight"].T,
                DTYPES[policy.conv_out],
                policy.ds_conv_out,
                policy.prefill_mm_v2,
            )

    def _gated_input(self, x, mem):
        if self.ds_in is None:
            return super()._gated_input(x, mem)
        bcx = self.ds_in(x, self.ckc, mem)
        rows, H = x.shape[-2], self.H
        b = ttnn.slice(bcx, (0, 0, 0, 0), (1, 1, rows, H), memory_config=mem)
        c = ttnn.slice(bcx, (0, 0, 0, H), (1, 1, rows, 2 * H), memory_config=mem)
        xx = ttnn.slice(bcx, (0, 0, 0, 2 * H), (1, 1, rows, 3 * H), memory_config=mem)
        ttnn.deallocate(bcx)
        bx = ttnn.multiply(b, xx, memory_config=mem)
        ttnn.deallocate(b)
        ttnn.deallocate(xx)
        return bx, c

    def _out_project(self, y, residual):
        if self.ds_out is None:
            return super()._out_project(y, residual)
        return _add_residual(self.ds_out(y, self.ckc, DRAM), residual, self.policy)


class OptimizedDenseMLP(FusedDenseMLP):
    def __init__(self, mesh_device, sd, policy: Policy, fusions):
        super().__init__(mesh_device, sd, DTYPES[policy.dense_w13], fusions)
        self.policy = policy
        if policy.dense_w2 != policy.dense_w13:
            from .functional_decoder import _as_tt

            self.w2 = _as_tt(sd["feed_forward.w2.weight"].T, mesh_device, dtype=DTYPES[policy.dense_w2])
        self.ckc = _ckc(mesh_device, policy.dense_fidelity)
        self.policy, self.mesh_device, self._w13_cfgs = policy, mesh_device, {}
        self.ds_w2 = self.ds_w1 = self.ds_w3 = self.ds_w2_dec = None
        w2t = sd["feed_forward.w2.weight"].T
        if policy.ds_dense_w2:
            ttnn.deallocate(self.w2)
            self.ds_w2 = DramShardedLinear(
                mesh_device, w2t, DTYPES[policy.dense_w2], policy.ds_dense_w2, policy.prefill_mm_v2
            )
        if policy.ds_dense_w13:
            # decode-only copies: DRAM-sharded w1/w3 (I zero-padded to banks * readers tiles) and a
            # K-padded DRAM-sharded w2.  Prefill keeps the interleaved w1/w3 + unpadded w2: a DRAM-sharded
            # w1/w3 in prefill needs full-width output blocks (repro/r2_mm_dram_sharded_partial_block.py),
            # 901 vs 651 us per 2048-row matmul (tracy_r2/conv_dense/b1/{fullw_bucket1,w13_interleaved_v2})
            assert self.ds_w2 is not None, "ds_dense_w13 needs ds_dense_w2 (prefill w2)"
            dt, cfg = DTYPES[policy.dense_w13], policy.ds_dense_w13
            self.ds_w1 = DramShardedLinear(mesh_device, sd["feed_forward.w1.weight"].T, dt, cfg, pad_n=True)
            self.ds_w3 = DramShardedLinear(mesh_device, sd["feed_forward.w3.weight"].T, dt, cfg, pad_n=True)
            self.ds_w2_dec = DramShardedLinear(
                mesh_device, w2t, DTYPES[policy.dense_w2], policy.ds_dense_w2_decode, k_pad=self.ds_w1.n
            )

    def forward(self, x, residual=None):
        T = x.shape[-2]
        if self.ds_w1 is not None and T <= TILE:  # decode: padded I end to end
            g = self.ds_w1.decode(x, self.ckc, L1)
            u = self.ds_w3.decode(x, self.ckc, L1)
            h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            out = _add_residual(self.ds_w2_dec.decode(h, self.ckc, DRAM), residual, self.policy)
            ttnn.deallocate(h)
            return out
        if (
            T <= TILE
            or not self.policy.dense_prefill_2d
            or "dense_minimal_swiglu" in self.f
            or "dense_merged_w13" in self.f
        ):
            return super().forward(x, residual)
        cfg = self._w13_cfgs.get(T)
        if cfg is None:
            p = self.policy
            cfg = self._w13_cfgs[T] = prefill_2d_config(
                self.mesh_device, T, x.shape[-1], self.I, p.prefill_mm_v2, _TILE_BYTES[p.dense_w13]
            )
        g = ttnn.linear(x, self.w1, program_config=cfg, compute_kernel_config=self.ckc, memory_config=DRAM)
        u = ttnn.linear(x, self.w3, program_config=cfg, compute_kernel_config=self.ckc, memory_config=DRAM)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = self._down_project(h, residual)
        ttnn.deallocate(h)
        return out

    def _down_project(self, h, residual):
        if self.ds_w2 is None:
            return super()._down_project(h, residual)
        return _add_residual(self.ds_w2(h, self.ckc, DRAM), residual, self.policy)


def _mcast_1d_config(cfg, rows, n):
    """sparse_matmul 1D in0-mcast config from a policy tuple (grid_x, grid_y, in0_block_w, out_subblock_w)."""
    gx, gy, in0_bw, sub_w = cfg
    per_core_n = -(-(n // TILE) // (gx * gy))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=rows // TILE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


_ROUTE_V3_CONSTS = {}  # moe_route_v3 index constants, shared by every MoE layer on a device


class OptimizedMoE(FusedMoE):
    def __init__(self, mesh_device, args, sd, policy: Policy, fusions):
        gu_dec = policy.expert_gate_up_decode or policy.expert_gate_up
        down_dec = policy.expert_down_decode or policy.expert_down
        super().__init__(mesh_device, args, sd, DTYPES[gu_dec], fusions)
        self.policy = policy
        if down_dec != gu_dec:
            from .functional_decoder import _as_tt

            _, _, down = stacked_expert_weights(sd, self.E)
            ttnn.deallocate(self.w_down)
            self.w_down = _as_tt(down.transpose(1, 2).unsqueeze(0), mesh_device, dtype=DTYPES[down_dec])
        self.ckc = _ckc(mesh_device, policy.expert_fidelity)
        if policy.moe_dispatch_min_rows:
            self._init_dispatch(sd)
        if policy.router_bf16:
            from .functional_decoder import _as_tt

            ttnn.deallocate(self.router)
            self.router = _as_tt(sd["feed_forward.gate.weight"].T, mesh_device, dtype=ttnn.bfloat16)
        if policy.router_ksplit:
            from .functional_decoder import _as_tt

            assert policy.router_bf16, "router_ksplit is implemented for the bf16 router"
            ks = policy.router_ksplit
            wr = sd["feed_forward.gate.weight"].T  # [H, E]
            self.router_split = _as_tt(
                wr.reshape(ks, self.H // ks, self.E).unsqueeze(0), mesh_device, dtype=ttnn.bfloat16
            )
        self.router_ds = None
        self._x_sharded = None
        if policy.router_ds_cores:
            import torch

            assert policy.router_bf16 and policy.sharded_decode_norm and policy.router_ds_cores == policy.ffn_norm_cores
            banks = mesh_device.dram_grid_size().x
            n_pad = -(-self.E // (TILE * banks)) * TILE * banks
            wr = torch.nn.functional.pad(
                sd["feed_forward.gate.weight"].T, (0, n_pad - self.E)
            )  # zero columns, sliced off
            cores = policy.router_ds_cores
            self.router_ds = DramShardedLinear(mesh_device, wr, ttnn.bfloat16, (cores, self.H // TILE // cores, 1))
            c = self.router_ds.decode_cfg
            self.router_ds_sig_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=c.in0_block_w,
                per_core_M=c.per_core_M,
                per_core_N=c.per_core_N,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            )
        if policy.moe_compact_max_users:
            self._init_compact()
        if policy.moe_route_v3:
            self._init_route_v3(sd)

    def _router_scores(self, x):
        if not self.policy.router_bf16:
            return super()._router_scores(x)
        T = x.shape[-2]
        if self.router_ds is not None and self._x_sharded is not None and T <= TILE and self.policy.router_ds_fused:
            ds = self.router_ds
            o = ttnn.linear(
                self._x_sharded,
                ds.weight,
                program_config=self.router_ds_sig_cfg,
                compute_kernel_config=self.ckc_router,
                memory_config=ds.out_mem,
                dtype=ttnn.float32,
            )
            scores = ttnn.slice(o, (0, 0, 0, 0), (1, 1, T, self.E), memory_config=L1)
            ttnn.deallocate(o)
            return scores
        if self.router_ds is not None and self._x_sharded is not None and T <= TILE:
            ds = self.router_ds
            o = ttnn.linear(
                self._x_sharded,
                ds.weight,
                program_config=ds.decode_cfg,
                compute_kernel_config=self.ckc_router,
                memory_config=ds.out_mem,
                dtype=ttnn.float32,
            )
            oi = ttnn.to_memory_config(o, L1)
            ttnn.deallocate(o)
            logits = ttnn.slice(oi, (0, 0, 0, 0), (1, 1, T, self.E), memory_config=L1)
            ttnn.deallocate(oi)
            scores = ttnn.sigmoid(logits, memory_config=L1)
            ttnn.deallocate(logits)
            return scores
        ks = self.policy.router_ksplit
        if ks and T <= TILE:
            # x [1,1,T,H] -> [1,ks,T,H/ks]; partial logits per K-slice on ks x more cores, summed in fp32
            xs = ttnn.permute(ttnn.reshape(x, (T, ks, self.H // ks)), (1, 0, 2))
            part = ttnn.matmul(
                ttnn.reshape(xs, (1, ks, T, self.H // ks)),
                self.router_split,
                compute_kernel_config=self.ckc_router,
                dtype=ttnn.float32,
            )
            ttnn.deallocate(xs)
            # ttnn.sum keeps fp32 exactly; fast_reduce_nc on fp32 partials lost precision (max |err|
            # 0.02-0.05 on the logits, flipping near-tie selections: decode update PCC 0.935 at 16 slices)
            logits = ttnn.sum(part, dim=1, keepdim=True)
            ttnn.deallocate(part)
            return ttnn.sigmoid(ttnn.reshape(logits, (1, 1, T, self.E)))
        return ttnn.linear(
            x, self.router, compute_kernel_config=self.ckc_router, dtype=ttnn.float32, activation="sigmoid"
        )

    # ---------------- prefill token dispatch ----------------
    def _init_dispatch(self, sd):
        """Per-expert 2D weights for unified_routed_expert_moe (a second copy of the expert weights:
        decode keeps the stacked sparse_matmul layout) and every constant the dispatch needs, so
        the prefill pass has no host work."""
        import torch

        from .functional_decoder import _as_tt

        md, E, K, H = self.mesh_device, self.E, self.K, self.H
        gate, up, down = stacked_expert_weights(sd, E)  # [E,I,H], [E,I,H], [E,H,I]
        dt_gu, dt_d = DTYPES[self.policy.expert_gate_up], DTYPES[self.policy.expert_down]
        self.d_gates = [_as_tt(gate[e].T, md, dtype=dt_gu) for e in range(E)]  # (H, I)
        self.d_ups = [_as_tt(up[e].T, md, dtype=dt_gu) for e in range(E)]
        self.d_downs = [_as_tt(down[e].T, md, dtype=dt_d) for e in range(E)]  # (I, H)
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.d_gidx = _as_tt(torch.arange(E, dtype=torch.int32), md, dtype=ttnn.uint32, layout=rm)
        self.d_arange_e = _as_tt(torch.arange(E, dtype=torch.float32).reshape(1, E), md, dtype=ttnn.float32)
        pmax = self.policy.moe_dispatch_max_rows * K
        # destination -> source row map; source row 0 is a zero row, token t is row t + 1
        self.d_src_tok = _as_tt(
            (torch.arange(pmax).div(K, rounding_mode="floor") + 1).float().reshape(1, pmax),
            md,
            dtype=ttnn.float32,
            layout=rm,
        )
        self.d_src_zero = _as_tt(
            torch.zeros(1, self._max_buf(self.policy.moe_dispatch_max_rows)), md, dtype=ttnn.float32, layout=rm
        )
        self.d_zero_row = _as_tt(torch.zeros(1, 1, 1, H), md, layout=rm)
        R = self.policy.moe_scatter_rows
        if R > 1:
            self.d_src_zero_rows = _as_tt(
                torch.zeros(R, self._max_buf(self.policy.moe_dispatch_max_rows)), md, dtype=ttnn.float32, layout=rm
            )
        # exact prefix sums as matmuls with 0/1 triangular constants (fp32 accumulation, HiFi4):
        # within 32-pair blocks (inclusive), across blocks (exclusive), across experts (exclusive)
        gmax = pmax // TILE
        self.d_tri32 = _as_tt(
            torch.tril(torch.ones(TILE, TILE)).reshape(1, 1, TILE, TILE).expand(1, gmax, TILE, TILE).contiguous(), md
        )
        self.d_blocks_strict = _as_tt(torch.tril(torch.ones(gmax, gmax), -1), md)
        self.d_experts_strict = _as_tt(torch.triu(torch.ones(E, E), 1), md)  # row-vector @ U: exclusive prefix
        self.d_scan_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.d_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity={"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[
                self.policy.dispatch_fidelity
            ],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _max_buf(self, T):
        # every expert region starts tile-aligned: <= T*K rows + 31 padding rows per expert
        return ((T * self.K + self.E * 31) + TILE - 1) // TILE * TILE

    def _route_topk(self, x):
        """Prefill routing as (indices uint16 [T, K], normalized weights [T, K]) -- same fp32
        sigmoid + selection-bias top-k as the dense router."""
        scores = self._router_scores(x)
        biased = ttnn.add(scores, self.bias)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1)
        ttnn.deallocate(vals)
        ttnn.deallocate(biased)
        sel = ttnn.gather(scores, dim=-1, index=idx)
        ttnn.deallocate(scores)
        w = ttnn.div(sel, ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6))
        if self.scale != 1.0:
            w = ttnn.multiply(w, self.scale)
        ttnn.deallocate(sel)
        return idx, w

    def _pair_cumsum(self, onehot, P):
        """Inclusive cumsum over pairs (dim 0) of the 0/1 [P, E] matrix: tril(32) per 32-pair block
        plus the exclusive prefix of block totals -- matmuls, exact in fp32 accumulation."""
        E, G = self.E, P // TILE
        oh = ttnn.typecast(ttnn.reshape(onehot, (1, G, TILE, E)), ttnn.bfloat16)
        tri = ttnn.slice(self.d_tri32, (0, 0, 0, 0), (1, G, TILE, TILE))
        scan_cfg = None
        if self.policy.moe_scan_multicore:
            # the default config runs the G batched 1x1x2-tile matmuls on 2 cores (262 us at G=256);
            # batch-parallel over 8x8 cores: 22 us (repro/r2_prefill_ops_bench.py scan)
            scan_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=E // TILE,
                per_core_M=1,
                per_core_N=E // TILE,
            )
        local = ttnn.matmul(
            tri, oh, compute_kernel_config=self.d_scan_ckc, dtype=ttnn.float32, program_config=scan_cfg
        )  # [1, G, 32, E]
        ttnn.deallocate(oh)
        tot = ttnn.typecast(
            ttnn.reshape(ttnn.slice(local, (0, 0, TILE - 1, 0), (1, G, TILE, E)), (G, E)), ttnn.bfloat16
        )  # <= 32
        strict = ttnn.slice(self.d_blocks_strict, (0, 0), (G, G))
        pre = ttnn.matmul(strict, tot, compute_kernel_config=self.d_scan_ckc, dtype=ttnn.float32)  # [G, E]
        ttnn.deallocate(tot)
        cum = ttnn.reshape(ttnn.add(local, ttnn.reshape(pre, (1, G, 1, E))), (P, E))
        ttnn.deallocate(local)
        ttnn.deallocate(pre)
        return cum

    def _dispatch_forward(self, x):
        """x [1,1,T,H] -> [1,1,T,H]: tokens sorted into tile-aligned per-expert regions (offsets
        and in-expert ranks from fp32 cumsums, exact for these sizes), one fused routed-expert FFN
        over all experts, then each token's K expert rows gathered back and weighted."""
        E, K, H = self.E, self.K, self.H
        T = x.shape[-2]
        P, maxbuf = T * K, self._max_buf(T)
        rm = lambda a: ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
        idx, w = self._route_topk(x)  # [1,1,T,K] uint16 / fp32
        e_col = ttnn.reshape(ttnn.typecast(idx, ttnn.float32), (P, 1))
        ttnn.deallocate(idx)
        onehot = ttnn.eq(e_col, self.d_arange_e)  # [P, E] pair -> expert
        ttnn.deallocate(e_col)
        cum = self._pair_cumsum(onehot, P)  # running count of each expert over pairs (exact)
        counts = ttnn.slice(cum, (P - 1, 0), (P, E))  # [1, E]
        tiles = ttnn.floor(ttnn.multiply(ttnn.add(counts, TILE - 1.0), 1.0 / TILE))  # region size in tiles (<= 65)
        region = ttnn.multiply(
            ttnn.matmul(
                ttnn.typecast(tiles, ttnn.bfloat16),
                self.d_experts_strict,
                compute_kernel_config=self.d_scan_ckc,
                dtype=ttnn.float32,
            ),
            float(TILE),
        )  # exclusive prefix over experts: region start row
        ttnn.deallocate(tiles)
        dest = ttnn.sum(ttnn.multiply(onehot, ttnn.add(ttnn.add(cum, region), -1.0)), dim=-1, keepdim=True)  # [P, 1]
        ttnn.deallocate(onehot)
        ttnn.deallocate(cum)
        dest_u = ttnn.typecast(rm(ttnn.reshape(dest, (1, P))), ttnn.uint32)
        ttnn.deallocate(dest)
        src_tok = ttnn.slice(self.d_src_tok, (0, 0), (1, P))
        src_init = ttnn.slice(self.d_src_zero, (0, 0), (1, maxbuf))
        R = self.policy.moe_scatter_rows
        if R > 1:
            # scatter parallelizes over rows only (1 row = 1 core: 404 us): the pairs are split over R
            # rows, each scattered into its own zero row; every destination is written exactly once,
            # so the fp32 row sum (values <= T + 1, exact) is the placement
            init = ttnn.slice(self.d_src_zero_rows, (0, 0), (R, maxbuf))
            part = ttnn.scatter(init, -1, ttnn.reshape(dest_u, (R, P // R)), ttnn.reshape(src_tok, (R, P // R)))
            part_t = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
            ttnn.deallocate(part)
            red = ttnn.sum(part_t, dim=0, keepdim=True)
            ttnn.deallocate(part_t)
            src = ttnn.typecast(rm(red), ttnn.uint32)
            ttnn.deallocate(red)
        else:
            src = ttnn.typecast(ttnn.scatter(src_init, -1, dest_u, src_tok), ttnn.uint32)
        table = ttnn.reshape(ttnn.concat([self.d_zero_row, rm(x)], dim=2), (T + 1, H))
        buf = ttnn.reshape(ttnn.embedding(src, table, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16), (maxbuf, H))
        ttnn.deallocate(src)
        ttnn.deallocate(table)
        region_u = ttnn.reshape(ttnn.typecast(rm(region), ttnn.uint32), (E,))
        counts_u = ttnn.reshape(ttnn.typecast(rm(counts), ttnn.uint32), (E,))
        if self.policy.dispatch_ffn == "hybrid" and T > self.policy.dispatch_hybrid_threshold:
            out = ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
                buf,
                region_u,
                counts_u,
                self.d_gidx,
                self.d_gates,
                self.d_ups,
                self.d_downs,
                T,
                hybrid_token_threshold=self.policy.dispatch_hybrid_threshold,
                compute_kernel_config=self.d_ckc,
            )
        else:
            out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                buf,
                region_u,
                counts_u,
                self.d_gidx,
                self.d_gates,
                self.d_ups,
                self.d_downs,
                T,
                compute_kernel_config=self.d_ckc,
            )  # bfp8 TILE [maxbuf, H]
        ttnn.deallocate(buf)
        if self.policy.moe_out_direct_untilize:
            out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)  # BFP8 tile -> BF16 RM in one untilize (bit-identical)
        else:
            out_rm = ttnn.to_layout(ttnn.typecast(out, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out)
        comb = ttnn.reshape(
            ttnn.embedding(dest_u, out_rm, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16), (1, 1, T, K, H)
        )
        ttnn.deallocate(out_rm)
        ttnn.deallocate(dest_u)
        w5 = ttnn.reshape(rm(ttnn.typecast(w, ttnn.bfloat16)), (1, 1, T, K, 1))
        ttnn.deallocate(w)
        y = ttnn.experimental.deepseek_prefill.post_combine_reduce(comb, w5, expert_dim=3, output_memory_config=DRAM)
        ttnn.deallocate(comb)
        return ttnn.reshape(y, (1, 1, T, H))

    def _mm_cfg(self, role, rows, n, groups):
        p = self.policy
        cfg = {
            ("gate_up", True): p.moe_gate_up_cfg,
            ("down", True): p.moe_down_cfg,
            ("gate_up", False): p.moe_gate_up_prefill_cfg,
            ("down_prefill", False): p.moe_down_prefill_cfg,
            ("down_prefill", True): p.moe_down_prefill_cfg,  # single-group sparse prefill (not reached today)
            ("down", False): None,
        }[(role, groups == 1 and rows <= TILE)]
        if cfg is None:
            return super()._mm_cfg(role, rows, n, groups)
        return _mcast_1d_config(cfg, rows, n)

    # ---------------- decode: compact (indexed) experts for a small active-user bucket ----------------
    def _init_compact(self):
        """Constants of the compact decode path for buckets 1, 2, 4, .. <= moe_compact_max_users."""
        import torch

        from .functional_decoder import _as_tt

        md, K, T = self.mesh_device, self.K, TILE
        self.c_buckets = [1 << i for i in range(8) if (1 << i) <= self.policy.moe_compact_max_users]
        assert self.c_buckets[-1] * K <= self.E, "bucket * top_k must not exceed the expert count"
        # sparsity is a required operand of sparse_matmul but is not read in indexed mode
        self.c_sparsity = _as_tt(torch.ones(1, 1, 1, self.E), md, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.c_tril = _as_tt(torch.tril(torch.ones(T, T)), md)  # inclusive prefix count of active slots
        self.c_arange_e = _as_tt(torch.arange(self.E, dtype=torch.float32).reshape(self.E, 1), md)
        self.c_rank, self.c_rep, self.c_tilek = {}, {}, {}
        for b in self.c_buckets[1:]:
            self.c_rank[b] = _as_tt(torch.arange(1, b + 1, dtype=torch.float32).reshape(1, b), md, dtype=ttnn.float32)
            rep = torch.zeros(b, b * K)  # [b, b*K]: user j -> its K compact slots
            tilek = torch.zeros(K, b * K)  # [K, b*K]: k-th choice -> slot j*K + k of every user j
            for j in range(b):
                rep[j, j * K : (j + 1) * K] = 1.0
                tilek[:, j * K : (j + 1) * K] = torch.eye(K)
            self.c_rep[b], self.c_tilek[b] = _as_tt(rep, md), _as_tt(tilek, md)
        self.c_exact_ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _compact_route(self, x, slot_active, b):
        """Decode routing for <= b active slots -> (ids uint16 ROW_MAJOR [1,1,1,b*K]: the experts
        selected by the active slots, slot j's K choices at compact slots j*K..j*K+K-1 (duplicates
        allowed); wc bf16 [1,1,T,b*K]: wc[r, s] = routing weight of row r for compact slot s, zero
        unless s belongs to row r).  Same selection/weights as the 64-wide path (``_route_tile_mask``);
        ids and weights come from exact 0/1 / small-integer matmuls (no gather, no host reads)."""
        K, n = self.K, b * self.K
        scores = self._router_scores(x)
        biased = ttnn.add(scores, self.bias)
        vals, idx = ttnn.topk(biased, k=K, dim=-1)
        kth = ttnn.min(vals, dim=-1, keepdim=True)
        ttnn.deallocate(vals)
        sel = ttnn.multiply(scores, ttnn.ge(biased, kth))
        ttnn.deallocate(biased)
        ttnn.deallocate(kth)
        ttnn.deallocate(scores)
        dense = ttnn.div(sel, ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6))
        ttnn.deallocate(sel)
        if self.scale != 1.0:
            dense = ttnn.multiply(dense, self.scale)
        dense = ttnn.typecast(dense, ttnn.bfloat16)  # [1,1,T,E], as the 64-wide path
        idx_f = ttnn.typecast(idx, ttnn.bfloat16)  # expert ids < 256 are exact in bf16
        ttnn.deallocate(idx)
        ex = self.c_exact_ckc
        if b == 1:
            own = slot_active  # [T, 1]: the single compact user is the active row
            ids = ttnn.matmul(slot_active, idx_f, transpose_a=True, compute_kernel_config=ex)
        else:
            # st[r, j] = 1 iff row r is the j-th active slot; own[r, s] = st[r, s // K]
            cnt = ttnn.matmul(self.c_tril, slot_active, compute_kernel_config=ex, dtype=ttnn.float32)
            st = ttnn.typecast(
                ttnn.multiply(ttnn.eq(cnt, self.c_rank[b]), ttnn.typecast(slot_active, ttnn.float32)), ttnn.bfloat16
            )
            ttnn.deallocate(cnt)
            own = ttnn.matmul(st, self.c_rep[b], compute_kernel_config=ex)
            ttnn.deallocate(st)
            idk = ttnn.matmul(idx_f, self.c_tilek[b], compute_kernel_config=ex)  # [T, n]: idx[r, s % K]
            sel_ids = ttnn.multiply(idk, own)
            ttnn.deallocate(idk)
            ids = ttnn.matmul(slot_active, sel_ids, transpose_a=True, compute_kernel_config=ex)
            ttnn.deallocate(sel_ids)
        ttnn.deallocate(idx_f)
        onehot = ttnn.eq(self.c_arange_e, ids)  # [E, n] bf16 (ids < 256 are exact in bf16)
        wsel = ttnn.matmul(dense, onehot, compute_kernel_config=ex)  # [T, n]: dense[r, ids[s]]
        ttnn.deallocate(onehot)
        ttnn.deallocate(dense)
        wc = ttnn.where(own, wsel, 0.0)  # NaN-safe: idle rows are undefined
        ttnn.deallocate(wsel)
        if b != 1:
            ttnn.deallocate(own)
        ids_u = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(ids)
        ids_u16 = ttnn.typecast(ids_u, ttnn.uint16)
        ttnn.deallocate(ids_u)
        return ids_u16, wc

    # ---------------- decode routing v3 (Policy.moe_route_v3) ----------------
    def _init_route_v3(self, sd):
        """Constants of the broadcast-free decode routing (T = 32 rows, E experts, top-K)."""
        import torch

        from .functional_decoder import _as_tt

        md, E, K, T = self.mesh_device, self.E, self.K, TILE
        assert self.scale == 1.0, "moe_route_v3 folds routed_scaling_factor 1.0"
        f32, bf = ttnn.float32, ttnn.bfloat16
        mem = L1 if self.policy.moe_route_v3_l1 else DRAM
        bias = sd["feed_forward.expert_bias"].float().reshape(1, E)
        self.v3_bias = _as_tt(bias.expand(T, E).contiguous(), md, dtype=f32, mem=mem)  # full-width: no row broadcast
        key = (id(md), E, K, tuple(self.c_buckets) if self.policy.moe_compact_max_users else (), mem is L1)
        shared = _ROUTE_V3_CONSTS.get(key)
        if shared is not None and shared["md"] is md:
            self.__dict__.update({k: v for k, v in shared.items() if k != "md"})
            return
        for k in [k for k, v in _ROUTE_V3_CONSTS.items() if v["md"] is not md]:
            del _ROUTE_V3_CONSTS[k]  # constants of a previous (closed) device
        _as = lambda t, dtype=bf: _as_tt(t, md, dtype=dtype, mem=mem)
        self.v3_eps = _as(torch.full((T, 1), 1e-6), f32)  # tensor eps: no scalar-op path
        rep_k = torch.zeros(K, K * E)  # X = idx @ rep_k: X[r, k*E + e] = idx[r, k]
        sum_k = torch.zeros(K * E, E)  # mask = OH @ sum_k: sum over the K one-hot blocks
        for k in range(K):
            rep_k[k, k * E : (k + 1) * E] = 1.0
            sum_k[k * E : (k + 1) * E] = torch.eye(E)
        self.v3_rep_k, self.v3_sum_k = _as(rep_k), _as(sum_k)
        self.v3_cols = _as(torch.arange(E, dtype=torch.float32).repeat(K).expand(T, K * E).contiguous())
        self.v3_ones_e = _as(torch.ones(1, E))  # active [T,1] @ ones -> full-width [T, E]
        self.v3_ex = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # K*E-deep contractions: in0_block_w = all K tiles (default config: 1 tile/step, 5.2 vs 2.1 us)
        kt = K * E // TILE
        self.v3_deep_cfg = lambda n: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(-(-n // TILE), 1),
            in0_block_w=kt,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.v3_rep_e, self.v3_pick, self.v3_pat = None, {}, {}
        if self.policy.moe_compact_max_users:
            rep_e = torch.zeros(E, K * E)  # dense @ rep_e: K copies of dense side by side
            for k in range(K):
                rep_e[:, k * E : (k + 1) * E] = torch.eye(E)
            self.v3_rep_e = _as(rep_e)
            for b in self.c_buckets:
                pick = torch.zeros(K * E, b * K)  # P @ pick: column j*K + k sums block k (every user j)
                pat = torch.zeros(T, b * K)  # pat[r, s] = 1 iff compact slot s belongs to row r (= user s // K)
                for j in range(b):
                    pat[j, j * K : (j + 1) * K] = 1.0
                    for k in range(K):
                        pick[k * E : (k + 1) * E, j * K + k] = 1.0
                self.v3_pick[b] = _as(pick)
                self.v3_pat[b] = _as(pat)
        _ROUTE_V3_CONSTS[key] = {
            "md": md,
            **{k: v for k, v in self.__dict__.items() if k.startswith("v3_") and k != "v3_bias"},
        }

    def _route_v3_select(self, x):
        """-> (scores fp32 [T,E], topk indices [T,K], one-hot OH bf16 [T,K*E], mask fp32 [T,E]).
        mask[r, e] = 1 iff e is one of row r's topk indices; identical to (biased >= k-th value)
        except on exact fp32 ties of the k-th / (k+1)-th biased scores (where the threshold would
        select more than K experts)."""
        ex = self.v3_ex
        scores = self._router_scores(x)
        biased = ttnn.add(scores, self.v3_bias, memory_config=L1)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1, memory_config=L1)
        ttnn.deallocate(vals)
        ttnn.deallocate(biased)
        idx_f = ttnn.typecast(idx, ttnn.bfloat16, memory_config=L1)  # ids < 256 exact in bf16
        X = ttnn.matmul(idx_f, self.v3_rep_k, compute_kernel_config=ex, memory_config=L1)
        ttnn.deallocate(idx_f)
        oh = ttnn.eq(X, self.v3_cols, memory_config=L1)
        ttnn.deallocate(X)
        mask = ttnn.matmul(
            oh,
            self.v3_sum_k,
            compute_kernel_config=ex,
            program_config=self.v3_deep_cfg(self.E),
            dtype=ttnn.float32,
            memory_config=L1,
        )
        return scores, idx, oh, mask

    def _route_v3_dense(self, sel):
        """bf16 [T,E] normalized weights from the selected scores, same fp32 math as
        _route_tile_mask (sum, +eps, div; routed_scaling_factor 1.0)."""
        ssum = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1)
        den = ttnn.add(ssum, self.v3_eps, memory_config=L1)
        ttnn.deallocate(ssum)
        dense = ttnn.div(sel, den, dtype=ttnn.bfloat16, memory_config=L1)  # == typecast(div) bitwise
        ttnn.deallocate(den)
        return dense

    def _route_v3_wide(self, x, slot_active):
        """64-wide decode routing [T,E] bf16 with idle rows (slot_active = 0) zeroed (NaN-safe)."""
        scores, idx, oh, mask = self._route_v3_select(x)
        ttnn.deallocate(idx)
        ttnn.deallocate(oh)
        sel = ttnn.multiply(scores, mask, memory_config=L1)  # == scores * (biased >= kth) bitwise
        ttnn.deallocate(mask)
        ttnn.deallocate(scores)
        dense = self._route_v3_dense(sel)
        ttnn.deallocate(sel)
        if slot_active is not None and self.policy.mask_idle_routing:
            act = ttnn.matmul(
                slot_active, self.v3_ones_e, compute_kernel_config=self.v3_ex, memory_config=L1
            )  # [T,E] 0/1
            masked = ttnn.where(act, dense, 0.0, memory_config=L1)  # full width (no [T,1] broadcast), NaN-safe
            ttnn.deallocate(act)
            ttnn.deallocate(dense)
            dense = masked
        return dense

    def _compact_route_v3(self, x, b):
        """_compact_route for the bucket contract (only rows < b active): compact slot j*K + k =
        row j's k-th topk expert; wc[r, j*K + k] = row r's weight of that expert iff r == j (rows
        that are idle get their own garbage weights, which only reach their own, undefined, rows)."""
        K, n = self.K, b * self.K
        ex = self.v3_ex
        scores, idx, oh, mask = self._route_v3_select(x)
        ids = ttnn.untilize_with_unpadding(idx, output_tensor_end=(0, 0, b - 1, K - 1), memory_config=L1)  # [b, K]
        ttnn.deallocate(idx)
        if b > 1:
            ids = ttnn.reshape(ids, (1, 1, 1, n))  # row-major [b, K] -> [1, b*K] (uint16 reshape unsupported)
        ids_u16 = ttnn.typecast(ids, ttnn.uint16, memory_config=L1)
        ttnn.deallocate(ids)
        sel = ttnn.multiply(scores, mask, memory_config=L1)  # == scores * (biased >= kth) bitwise
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        dense = self._route_v3_dense(sel)
        ttnn.deallocate(sel)
        drep = ttnn.matmul(dense, self.v3_rep_e, compute_kernel_config=ex, memory_config=L1)  # exact copies
        ttnn.deallocate(dense)
        p = ttnn.multiply(oh, drep, memory_config=L1)  # one nonzero (the selected weight) per K-block
        ttnn.deallocate(oh)
        ttnn.deallocate(drep)
        wc = ttnn.matmul(
            p, self.v3_pick[b], compute_kernel_config=ex, program_config=self.v3_deep_cfg(n), memory_config=L1
        )
        ttnn.deallocate(p)
        if b > 1:
            wcm = ttnn.multiply(wc, self.v3_pat[b], memory_config=L1)
            ttnn.deallocate(wc)
            wc = wcm
        return ids_u16, wc

    def _compact_forward(self, x, slot_active, b):
        T, I, H, K = x.shape[-2], self.I, self.H, self.K
        n = b * K
        p = self.policy
        if p.moe_route_v3 and T == TILE:
            ids, wc = self._compact_route_v3(x, b)
        else:
            ids, wc = self._compact_route(x, slot_active, b)
        x = ttnn.reshape(x, (1, 1, T, H))
        kw = dict(
            sparsity=self.c_sparsity,
            indices=ids,
            memory_config=L1,
            output_tile=ttnn.Tile([32, 32]),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        gu_cfg = p.moe_gate_up_compact_cfg or p.moe_gate_up_cfg
        # packed [up | gate] (one indexed matmul + 2 small slices beats two half-width matmuls)
        ug = ttnn.sparse_matmul(x, self.w_up_gate, program_config=_mcast_1d_config(gu_cfg, T, 2 * I), **kw)
        ug = ttnn.reshape(ug, (1, n, T, 2 * I))  # [1,1,1,n,T,2I] -> [1,n,T,2I]
        up = ttnn.slice(ug, (0, 0, 0, 0), (1, n, T, I), memory_config=L1)
        gate = ttnn.slice(ug, (0, 0, 0, I), (1, n, T, 2 * I), memory_config=L1)
        ttnn.deallocate(ug)
        act = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            program_config=_mcast_1d_config(p.moe_down_compact_cfg or p.moe_down_cfg, T, H),
            is_input_a_sparse=True,
            **kw,
        )  # [1, n, T, H]
        ttnn.deallocate(act)
        ttnn.deallocate(ids)
        wt = ttnn.permute(ttnn.reshape(wc, (1, 1, T, n)), (0, 3, 2, 1), memory_config=L1)  # [1, n, T, 1]
        ttnn.deallocate(wc)
        down_w = ttnn.multiply(down, wt, memory_config=L1)
        ttnn.deallocate(down)
        ttnn.deallocate(wt)
        out = ttnn.experimental.fast_reduce_nc(down_w, dims=[1], memory_config=L1)
        ttnn.deallocate(down_w)
        return ttnn.reshape(out, (1, 1, T, H))

    def _experts(self, x, routing, sparsity, rows_per_group, groups):
        if groups != 1 or not self.policy.moe_decode_l1 or "moe_split_gate_up" in self.f:
            return super()._experts(x, routing, sparsity, rows_per_group, groups)
        # decode (one 32-row group): stage-2 graph with every intermediate in L1
        T, I, H, E = rows_per_group, self.I, self.H, self.E
        up_gate = ttnn.sparse_matmul(
            x,
            self.w_up_gate,
            sparsity=sparsity,
            nnz=None,
            memory_config=L1,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._mm_cfg("gate_up", T, 2 * I, 1),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        up_gate = ttnn.reshape(up_gate, (1, E, T, 2 * I))
        up = ttnn.slice(up_gate, (0, 0, 0, 0), (1, E, T, I), memory_config=L1)
        gate = ttnn.slice(up_gate, (0, 0, 0, I), (1, E, T, 2 * I), memory_config=L1)
        ttnn.deallocate(up_gate)
        act = ttnn.multiply(up, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            sparsity=sparsity,
            nnz=None,
            memory_config=L1,
            output_tile=ttnn.Tile([32, 32]),
            program_config=self._mm_cfg("down", T, H, 1),
            compute_kernel_config=self.ckc,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(act)
        down = ttnn.reshape(down, (1, E, T, H))
        weights = ttnn.permute(routing, (0, 3, 2, 1), memory_config=L1)
        down_w = ttnn.multiply(down, weights, memory_config=L1)
        ttnn.deallocate(down)
        out = ttnn.experimental.fast_reduce_nc(down_w, dims=[1], memory_config=L1)
        ttnn.deallocate(down_w)
        return ttnn.reshape(out, (1, 1, T, H))

    def _route_tile_mask(self, x):
        if not self.policy.router_kth_slice:
            return super()._route_tile_mask(x)
        # FusedMoE._route_tile_mask with the k-th largest biased score read from the (descending)
        # topk values instead of a min-reduce over the padded [T, k] tile (FillPad + reduce)
        scores = self._router_scores(x)
        biased = ttnn.add(scores, self.bias)
        vals, idx = ttnn.topk(biased, k=self.K, dim=-1)
        ttnn.deallocate(idx)
        kth = ttnn.slice(vals, (0, 0, 0, self.K - 1), (1, 1, vals.shape[-2], self.K))
        ttnn.deallocate(vals)
        sel = ttnn.multiply(scores, ttnn.ge(biased, kth))
        ttnn.deallocate(biased)
        ttnn.deallocate(kth)
        ttnn.deallocate(scores)
        denom = ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-6)
        w = ttnn.div(sel, denom)
        if self.scale != 1.0:
            w = ttnn.multiply(w, self.scale)
        ttnn.deallocate(sel)
        return ttnn.typecast(w, ttnn.bfloat16)

    def forward(self, x, slot_active=None, x_sharded=None, active_slots=None):
        """``x_sharded``: the same decode rows width-sharded on the ffn_norm grid (router_ds_cores).
        ``active_slots``: static decode bucket b (one trace per bucket): at most b slots are active
        (any positions for the compact MoE path).  None = up to all 32 slots."""
        self._x_sharded = x_sharded
        try:
            return self._forward(x, slot_active, active_slots)
        finally:
            self._x_sharded = None

    def _forward(self, x, slot_active=None, active_slots=None):
        T = x.shape[-2]
        p = self.policy
        if T <= TILE and active_slots is not None and slot_active is not None and p.moe_compact_max_users:
            b = next((c for c in self.c_buckets if c >= active_slots), None)
            if b is not None:
                return self._compact_forward(x, slot_active, b)
        if T > TILE and p.moe_dispatch_min_rows and T >= p.moe_dispatch_min_rows:
            if T <= p.moe_dispatch_max_rows:
                return self._dispatch_forward(x)
            parts = [
                self._dispatch_forward(
                    ttnn.slice(x, (0, 0, s0, 0), (1, 1, min(s0 + p.moe_dispatch_max_rows, T), self.H))
                )
                for s0 in range(0, T, p.moe_dispatch_max_rows)
            ]
            out = ttnn.concat(parts, dim=2)
            for part in parts:
                ttnn.deallocate(part)
            return out
        if T > TILE:
            return super().forward(x)
        if p.moe_route_v3 and T == TILE:
            routing = self._route_v3_wide(x, slot_active)  # idle rows already zeroed
            slot_active = None
        else:
            routing = self.route(x)
        if slot_active is not None and self.policy.mask_idle_routing:
            # idle decode slots carry garbage rows: drop their expert selections so the union
            # sparsity (and so the weights read) only covers active users
            masked = ttnn.where(slot_active, routing, 0.0)  # NaN-safe (idle rows are undefined)
            ttnn.deallocate(routing)
            routing = masked
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        out = self._experts(ttnn.reshape(x, (1, 1, T, self.H)), routing, sparsity, T, 1)
        ttnn.deallocate(routing)
        return out


class OptimizedDecoder(FusedDecoder):
    """``FusedDecoder`` + ``Policy``.  ``from_state_dict(..., policy=DEFAULT_POLICY, fusions=DEFAULT_FUSIONS)``."""

    accepts_slot_mask = True  # decode_forward(slot_mask=[1, 1, B, H]) (see decode_forward)

    @property
    def bucketed_rope(self):
        """Decode callers pass ``rope.decode_cos_sin(rope_pos, batch=active_slots)`` (b-user cos/sin)."""
        return self.is_attention and self.policy.attn_bucket_heads

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=TILE,
        policy=DEFAULT_POLICY,
        fusions=DEFAULT_FUSIONS,
        **kwargs,
    ):
        return cls(state_dict, hf_config, layer_idx, mesh_device, max_batch_size, policy, fusions, **kwargs)

    def __init__(self, sd, hf_config, layer_idx, mesh_device, max_batch_size, policy, fusions, sdpa_chunk=256):
        from .functional_decoder import _as_tt

        self.policy = policy
        self.fusions = frozenset(fusions)
        args = Lfm2MoeArgs.from_hf_config(hf_config)
        self.args = args
        self.layer_idx = layer_idx
        self.kind = args.layer_kind(layer_idx)
        self.mesh_device = mesh_device
        self.eps = args.norm_eps
        self.ckc_norm = _hifi4(mesh_device)
        self.operator_norm = _as_tt(
            sd["operator_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.ffn_norm = _as_tt(sd["ffn_norm.weight"].reshape(1, 1, -1, TILE), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.is_attention = args.is_attention_layer(layer_idx)
        if self.is_attention:
            self.mixer = OptimizedAttention(
                mesh_device, args, sd, max_batch_size, policy, self.fusions, policy.sdpa_prefill_chunk
            )
        else:
            self.mixer = OptimizedShortConv(mesh_device, args, sd, max_batch_size, policy, self.fusions)
        if args.is_moe_layer(layer_idx):
            self.ffn = OptimizedMoE(mesh_device, args, sd, policy, self.fusions)
        else:
            self.ffn = OptimizedDenseMLP(mesh_device, sd, policy, self.fusions)
        # prefill callers build ConvPrefillInputs with windowed=decoder.conv_prefill_windowed
        self.conv_prefill_windowed = (not self.is_attention) and policy.conv_prefill_windowed_tail
        self.op_norm_sh = self.ffn_norm_sh = None
        if policy.sharded_decode_norm:
            consumer = self.mixer.ds_qkv if self.is_attention else self.mixer.ds_in
            assert consumer is not None, "sharded_decode_norm needs the DRAM-sharded qkv / conv in_proj"
            H = args.hidden_size
            self.op_norm_sh = ShardedNorm(mesh_device, H, consumer.cores, self.eps, self.ckc_norm)
            assert self.op_norm_sh.mem == consumer.in_mem
            self.ffn_norm_sh = ShardedNorm(mesh_device, H, policy.ffn_norm_cores, self.eps, self.ckc_norm)

    def _decode_ffn_norm(self, h):
        if self.ffn_norm_sh is None:
            return self._norm(h, self.ffn_norm, DRAM)
        hs = self.ffn_norm_sh(h, self.ffn_norm)
        hn = ttnn.to_memory_config(hs, L1)
        ttnn.deallocate(hs)
        return hn

    def _ffn_block(self, h, slot_active=None, active_slots=None):
        decode = h.shape[-2] <= TILE
        hs = None
        if decode and self.ffn_norm_sh is not None and getattr(self.ffn, "router_ds", None) is not None:
            hs = self.ffn_norm_sh(h, self.ffn_norm)  # router reads the sharded copy directly
            hn = ttnn.to_memory_config(hs, L1)
        else:
            hn = self._decode_ffn_norm(h) if decode else self._norm(h, self.ffn_norm, DRAM)
        if isinstance(self.ffn, FusedDenseMLP):
            out = self.ffn.forward(hn, residual=h)
            ttnn.deallocate(hn)
            return out
        f = self.ffn.forward(hn, slot_active=slot_active, x_sharded=hs, active_slots=active_slots)
        ttnn.deallocate(hn)
        if hs is not None:
            ttnn.deallocate(hs)
        # (h may be the width-sharded attention residual: attn_residual_sharded)
        out = ttnn.add(h, f, memory_config=L1 if (decode and self.policy.decode_residual_l1) else DRAM)
        ttnn.deallocate(f)
        return out

    def decode_forward(
        self,
        x,
        *,
        rope=None,
        rope_cos=None,
        rope_sin=None,
        cur_pos=None,
        page_table=None,
        slot_active=None,
        active_slots=None,
        slot_mask=None,
    ):
        """``active_slots``: static decode bucket b (one trace per bucket); the caller guarantees that
        only slots 0..b-1 can be active (cur_pos -1 for the rest).  None = all 32 slots.
        ``slot_mask``: optional ``slot_active`` broadcast to ``[1, 1, B, H]`` (built once per step by the
        caller): the ShortConv state updates then run as full-width ``where`` (3.1 vs 8.5 us each)."""
        resid_sh = (
            self.is_attention
            and self.policy.attn_residual_sharded
            and self.policy.attn_pad_free
            and self.op_norm_sh is not None
            and isinstance(self.ffn, OptimizedMoE)
            and self.ffn_norm_sh is not None
            and self.ffn_norm_sh.mem == self.op_norm_sh.mem
        )
        xs = ttnn.to_memory_config(x, self.op_norm_sh.mem) if resid_sh else None  # residual, sharded once
        if self.op_norm_sh is not None:
            xn = self.op_norm_sh(xs if resid_sh else x, self.operator_norm)  # already in the qkv / in_proj input layout
        else:
            xn = self._norm(x, self.operator_norm, L1)
        if self.is_attention:
            h = self.mixer.decode_forward(
                xn,
                rope=rope,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                cur_pos=cur_pos,
                page_table=page_table,
                residual=xs if resid_sh else x,
                active_slots=active_slots,
            )
            if resid_sh:
                ttnn.deallocate(xs)
        else:
            h = self.mixer.decode_forward(xn, slot_active if slot_mask is None else slot_mask, residual=x)
        ttnn.deallocate(xn)
        out = self._ffn_block(h, slot_active, active_slots)
        ttnn.deallocate(h)
        return out


DECODE_BUCKETS = (1, 2, 4, 8, 16, TILE)


def decode_bucket(active_slot_ids, buckets=DECODE_BUCKETS):
    """Host-side: the smallest bucket covering the highest active slot (None = no bucket, all 32).
    Assign new requests to the lowest free slot; then no per-slot state (KV page rows, ShortConv
    state rows) ever has to move, and holes below the highest active slot are allowed."""
    top = max(active_slot_ids, default=-1) + 1
    b = next(b for b in buckets if b >= max(top, 1))
    return None if b >= TILE else b


def validate_decode_bucket(cur_pos, active_slots):
    """Host-side check of the ``active_slots`` contract on the host ``cur_pos`` [B] (-1 = idle): a
    slot >= b that is active would silently get zero attention output (SDPA reads rows < b) and
    more than b active rows would overflow the compact MoE ids.  Raise instead."""
    if active_slots is None:
        return
    active = [i for i, p in enumerate(cur_pos.tolist()) if p >= 0]
    if active and (max(active) >= active_slots or len(active) > active_slots):
        raise ValueError(f"decode bucket {active_slots}: active slots {active} must all be < {active_slots}")


def make_optimized(sd, cfg, args, layer_idx, mesh_device, max_seq_len, policy=DEFAULT_POLICY, fusions=DEFAULT_FUSIONS):
    dec = OptimizedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, policy=policy, fusions=fusions
    )
    rope = make_rope(mesh_device, args, max_seq_len, fusions) if dec.is_attention else None
    return dec, rope


__all__ = [
    "Policy",
    "DEFAULT_POLICY",
    "BASELINE_POLICY",
    "OptimizedDecoder",
    "make_optimized",
    "replace",
    "DECODE_BUCKETS",
    "decode_bucket",
    "validate_decode_bucket",
]
