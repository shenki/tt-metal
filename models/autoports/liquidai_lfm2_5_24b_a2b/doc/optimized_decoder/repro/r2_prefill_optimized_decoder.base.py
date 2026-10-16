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
    # decode MoE for a static small active-user bucket (decode_bucket <= this; 0 = off): the
    # bucket*K selected experts are computed compactly with sparse_matmul indexed (gather) mode
    # instead of the 64-expert-wide path; requires at most decode_bucket active slots
    moe_compact_max_users: int = 0
    moe_compact_split_gate_up: bool = False  # compact path: separate up / gate indexed matmuls (no slices)
    moe_gate_up_compact_cfg: tuple = None  # (grid_x, grid_y, in0_block_w, out_subblock_w); None = moe_gate_up_cfg
    moe_down_compact_cfg: tuple = None  # None = moe_down_cfg

    def summary(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


BASELINE_POLICY = Policy(mask_idle_routing=False)  # == fused decoder
DEFAULT_POLICY = Policy(
    kv_cache="bfp8",
    sdpa_decode_k_chunk=512,
    expert_gate_up="bfp8",
    expert_down="bfp8",
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
    ds_dense_w2=(46, 8, 2),
    sharded_decode_norm=True,
    moe_gate_up_prefill_cfg=(8, 8, 8, 1),
    moe_down_prefill_cfg=(8, 6, 8, 2),
    moe_dispatch_min_rows=96,
    sdpa_prefill_chunk=128,
    dense_prefill_2d=True,
    router_bf16=True,
    router_ksplit=8,
    dispatch_ffn="hybrid",
    dispatch_fidelity="lofi",
    dispatch_hybrid_threshold=128,
    conv_prefill_windowed_tail=True,
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
        xs = ttnn.to_memory_config(x, self.mem)
        y = ttnn.rms_norm(
            xs,
            epsilon=self.eps,
            weight=weight,
            program_config=self.cfg,
            memory_config=self.mem,
            compute_kernel_config=self.ckc,
        )
        ttnn.deallocate(xs)
        return y


def prefill_2d_config(mesh_device, m, k, n):
    """2D-mcast matmul config for large-M prefill over the full compute grid (works with
    interleaved and DRAM-width-sharded weights)."""
    grid = mesh_device.compute_with_storage_grid_size()
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    gy, gx = min(grid.y, mt), min(grid.x, nt)
    per_core_m, per_core_n = math.ceil(mt / gy), math.ceil(nt / gx)
    # circular buffers must fit L1: double-buffered bf16 in0 / (<= bf16) in1 blocks, and the bf16
    # output block with its fp32 partials; shrink the output block (divisors of the per-core tile
    # counts) first, then in0_block_w
    budget = 1_300_000
    divs = lambda v: [d for d in range(v, 0, -1) if v % d == 0]
    legal_bw = [b for b in (8, 4, 2, 1) if kt % b == 0]
    choice = None
    for bw in legal_bw:
        for ob_w in divs(per_core_n):
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


class DramShardedLinear:
    """``x @ W`` with ``W`` width-sharded over the DRAM banks.

    Decode (M = 32): activation resharded to a width-sharded L1 grid of ``cores`` cores,
    ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` (``in0_block_w``, readers per
    bank), output back to interleaved.  Prefill: the same weight with a 2D mcast config."""

    def __init__(self, mesh_device, w, dtype, cfg):
        self.mesh_device = mesh_device
        banks = mesh_device.dram_grid_size().x
        k, n = w.shape
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
        self.out_mem = width(per_core_n * TILE)
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

    def prefill(self, x, ckc, mem=DRAM):
        m = x.shape[-2]
        if m not in self._prefill_cfgs:
            self._prefill_cfgs[m] = prefill_2d_config(self.mesh_device, m, self.k, self.n)
        return ttnn.linear(
            x, self.weight, program_config=self._prefill_cfgs[m], compute_kernel_config=ckc, memory_config=mem
        )

    def __call__(self, x, ckc, mem):
        return self.decode(x, ckc, mem) if x.shape[-2] <= TILE else self.prefill(x, ckc, mem)


def _add_residual(out, residual):
    if residual is None:
        return out
    h = ttnn.add(residual, out, memory_config=DRAM)
    ttnn.deallocate(out)
    return h


class OptimizedAttention(FusedAttention):
    def __init__(self, mesh_device, args, sd, max_batch_size, policy: Policy, fusions, sdpa_chunk=256):
        super().__init__(mesh_device, args, sd, max_batch_size, DTYPES[policy.attn_qkv], fusions, sdpa_chunk=sdpa_chunk)
        self.policy = policy
        if policy.attn_out != policy.attn_qkv:
            from .functional_decoder import _as_tt

            self.wo = _as_tt(sd["self_attn.out_proj.weight"].T, mesh_device, dtype=DTYPES[policy.attn_out])
        self.ckc = _ckc(mesh_device, policy.attn_fidelity)
        self.ckc_sdpa_decode = _ckc(mesh_device, policy.sdpa_decode_fidelity)
        self.ds_qkv = self.ds_wo = None
        if policy.ds_wqkv:
            import torch

            wqkv = torch.cat(
                [sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"], sd["self_attn.v_proj.weight"]], dim=0
            ).T
            ttnn.deallocate(self.wqkv)
            self.ds_qkv = DramShardedLinear(mesh_device, wqkv, DTYPES[policy.attn_qkv], policy.ds_wqkv)
        if policy.ds_wo:
            ttnn.deallocate(self.wo)
            self.ds_wo = DramShardedLinear(
                mesh_device, sd["self_attn.out_proj.weight"].T, DTYPES[policy.attn_out], policy.ds_wo
            )

    def _qkv_linear(self, x, mem):
        if self.ds_qkv is None:
            return super()._qkv_linear(x, mem)
        return self.ds_qkv(x, self.ckc, mem)

    def _out_project(self, attn, residual):
        if self.ds_wo is None:
            return super()._out_project(attn, residual)
        return _add_residual(self.ds_wo(attn, self.ckc, DRAM), residual)

    def _sdpa_decode(self, q, cur_pos, page_table):
        p = self.policy
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=0,
            k_chunk_size=p.sdpa_decode_k_chunk,
            exp_approx_mode=p.sdpa_decode_exp_approx,
        )
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
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

    def allocate_kv_cache(self, num_blocks, block_size, dtype=None):
        super().allocate_kv_cache(num_blocks, block_size, dtype=dtype or DTYPES[self.policy.kv_cache])


class OptimizedShortConv(FusedShortConv):
    def __init__(self, mesh_device, args, sd, max_batch_size, policy: Policy, fusions):
        super().__init__(mesh_device, args, sd, max_batch_size, DTYPES[policy.conv_in], fusions)
        if policy.conv_out != policy.conv_in:
            from .functional_decoder import _as_tt

            self.out_proj = _as_tt(sd["conv.out_proj.weight"].T, mesh_device, dtype=DTYPES[policy.conv_out])
        self.ckc = _ckc(mesh_device, policy.conv_fidelity)
        self.ds_in = self.ds_out = None
        if policy.ds_conv_in:
            ttnn.deallocate(self.in_proj)
            self.ds_in = DramShardedLinear(
                mesh_device, sd["conv.in_proj.weight"].T, DTYPES[policy.conv_in], policy.ds_conv_in
            )
        if policy.ds_conv_out:
            ttnn.deallocate(self.out_proj)
            self.ds_out = DramShardedLinear(
                mesh_device, sd["conv.out_proj.weight"].T, DTYPES[policy.conv_out], policy.ds_conv_out
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
        return _add_residual(self.ds_out(y, self.ckc, DRAM), residual)


class OptimizedDenseMLP(FusedDenseMLP):
    def __init__(self, mesh_device, sd, policy: Policy, fusions):
        super().__init__(mesh_device, sd, DTYPES[policy.dense_w13], fusions)
        if policy.dense_w2 != policy.dense_w13:
            from .functional_decoder import _as_tt

            self.w2 = _as_tt(sd["feed_forward.w2.weight"].T, mesh_device, dtype=DTYPES[policy.dense_w2])
        self.ckc = _ckc(mesh_device, policy.dense_fidelity)
        self.policy, self.mesh_device, self._w13_cfgs = policy, mesh_device, {}
        self.ds_w2 = None
        if policy.ds_dense_w2:
            ttnn.deallocate(self.w2)
            self.ds_w2 = DramShardedLinear(
                mesh_device, sd["feed_forward.w2.weight"].T, DTYPES[policy.dense_w2], policy.ds_dense_w2
            )

    def forward(self, x, residual=None):
        T = x.shape[-2]
        if (
            T <= TILE
            or not self.policy.dense_prefill_2d
            or "dense_minimal_swiglu" in self.f
            or "dense_merged_w13" in self.f
        ):
            return super().forward(x, residual)
        cfg = self._w13_cfgs.get(T)
        if cfg is None:
            cfg = self._w13_cfgs[T] = prefill_2d_config(self.mesh_device, T, x.shape[-1], self.I)
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
        return _add_residual(self.ds_w2(h, self.ckc, DRAM), residual)


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


class OptimizedMoE(FusedMoE):
    def __init__(self, mesh_device, args, sd, policy: Policy, fusions):
        super().__init__(mesh_device, args, sd, DTYPES[policy.expert_gate_up], fusions)
        self.policy = policy
        if policy.expert_down != policy.expert_gate_up:
            from .functional_decoder import _as_tt

            _, _, down = stacked_expert_weights(sd, self.E)
            self.w_down = _as_tt(down.transpose(1, 2).unsqueeze(0), mesh_device, dtype=DTYPES[policy.expert_down])
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
        if policy.moe_compact_max_users:
            self._init_compact()

    def _router_scores(self, x):
        if not self.policy.router_bf16:
            return super()._router_scores(x)
        T = x.shape[-2]
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
        local = ttnn.matmul(tri, oh, compute_kernel_config=self.d_scan_ckc, dtype=ttnn.float32)  # [1, G, 32, E]
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
        if self.policy.moe_compact_split_gate_up:
            _, I = self.H, self.I
            self.c_w_up = ttnn.slice(self.w_up_gate, (0, 0, 0, 0), (1, self.E, self.H, I))
            self.c_w_gate = ttnn.slice(self.w_up_gate, (0, 0, 0, I), (1, self.E, self.H, 2 * I))

    def _compact_route(self, x, slot_active, b):
        """-> (ids uint16 ROW_MAJOR [1,1,1,b*K]: the selected experts of the <= b active slots in
        slot order (duplicates allowed), wc bf16 [1,1,T,b*K]: wc[r, s] = routing weight of row r
        for compact slot s, zero unless slot s belongs to row r's user).  All on device."""
        T, K = x.shape[-2], self.K
        idx, w = self._route_topk(x)  # [1,1,T,K] uint16 / fp32 (top-k order)
        idx_f = ttnn.typecast(idx, ttnn.bfloat16)  # expert ids < 256 are exact in bf16
        ttnn.deallocate(idx)
        w = ttnn.typecast(w, ttnn.bfloat16)
        if b == 1:
            # the single active row's ids (an exact matmul: ttnn.sum's reduce rounds ids > 31)
            ids = ttnn.matmul(
                slot_active, idx_f, transpose_a=True, compute_kernel_config=self.c_exact_ckc, dtype=ttnn.float32
            )
            wc = ttnn.multiply(w, slot_active)
        else:
            # st[r, j] = 1 iff row r is the j-th active slot (exact 0/1 and small-integer matmuls)
            cnt = ttnn.matmul(self.c_tril, slot_active, compute_kernel_config=self.c_exact_ckc, dtype=ttnn.float32)
            st = ttnn.typecast(
                ttnn.multiply(ttnn.eq(cnt, self.c_rank[b]), ttnn.typecast(slot_active, ttnn.float32)), ttnn.bfloat16
            )
            ttnn.deallocate(cnt)
            ids = ttnn.matmul(
                st, idx_f, transpose_a=True, compute_kernel_config=self.c_exact_ckc, dtype=ttnn.float32
            )  # [b, K]
            wk = ttnn.matmul(w, self.c_tilek[b], compute_kernel_config=self.c_exact_ckc)  # [T, b*K]: w[r, s % K]
            own = ttnn.matmul(st, self.c_rep[b], compute_kernel_config=self.c_exact_ckc)  # [T, b*K]: slot s is row r's
            wc = ttnn.multiply(wk, own)
            ttnn.deallocate(st)
            ttnn.deallocate(wk)
            ttnn.deallocate(own)
        ttnn.deallocate(idx_f)
        ttnn.deallocate(w)
        ids_u = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(ids)
        ids_u16 = ttnn.typecast(ttnn.reshape(ids_u, (1, 1, 1, b * K)), ttnn.uint16)
        ttnn.deallocate(ids_u)
        return ids_u16, wc

    def _compact_forward(self, x, slot_active, b):
        T, I, H, K = x.shape[-2], self.I, self.H, self.K
        n = b * K
        p = self.policy
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
        if p.moe_compact_split_gate_up:
            up = ttnn.sparse_matmul(x, self.c_w_up, program_config=_mcast_1d_config(gu_cfg, T, I), **kw)
            gate = ttnn.sparse_matmul(x, self.c_w_gate, program_config=_mcast_1d_config(gu_cfg, T, I), **kw)
            up, gate = ttnn.reshape(up, (1, n, T, I)), ttnn.reshape(gate, (1, n, T, I))
        else:
            ug = ttnn.sparse_matmul(x, self.w_up_gate, program_config=_mcast_1d_config(gu_cfg, T, 2 * I), **kw)
            ug = ttnn.reshape(ug, (1, n, T, 2 * I))
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

    def forward(self, x, slot_active=None, decode_bucket=None):
        """``decode_bucket``: static upper bound on the active decode slots (None: any)."""
        T = x.shape[-2]
        p = self.policy
        if T <= TILE and decode_bucket is not None and slot_active is not None and p.moe_compact_max_users:
            b = next((c for c in self.c_buckets if c >= decode_bucket), None)
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
        routing = self.route(x)
        if slot_active is not None and self.policy.mask_idle_routing:
            # idle decode slots carry garbage rows: drop their expert selections so the union
            # sparsity (and so the weights read) only covers active users
            masked = ttnn.multiply(routing, slot_active)
            ttnn.deallocate(routing)
            routing = masked
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        out = self._experts(ttnn.reshape(x, (1, 1, T, self.H)), routing, sparsity, T, 1)
        ttnn.deallocate(routing)
        return out


class OptimizedDecoder(FusedDecoder):
    """``FusedDecoder`` + ``Policy``.  ``from_state_dict(..., policy=DEFAULT_POLICY, fusions=DEFAULT_FUSIONS)``."""

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

    def _ffn_block(self, h, slot_active=None, decode_bucket=None):
        decode = h.shape[-2] <= TILE
        hn = self._decode_ffn_norm(h) if decode else self._norm(h, self.ffn_norm, DRAM)
        if isinstance(self.ffn, FusedDenseMLP):
            out = self.ffn.forward(hn, residual=h)
            ttnn.deallocate(hn)
            return out
        f = self.ffn.forward(hn, slot_active=slot_active, decode_bucket=decode_bucket)
        ttnn.deallocate(hn)
        out = ttnn.add(h, f, memory_config=DRAM)
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
        decode_bucket=None,
    ):
        """``decode_bucket``: static bound on the active slots of this (traced) decode step; None = any."""
        if self.op_norm_sh is not None:
            xn = self.op_norm_sh(x, self.operator_norm)  # already in the qkv / in_proj input layout
        else:
            xn = self._norm(x, self.operator_norm, L1)
        if self.is_attention:
            h = self.mixer.decode_forward(
                xn, rope=rope, rope_cos=rope_cos, rope_sin=rope_sin, cur_pos=cur_pos, page_table=page_table, residual=x
            )
        else:
            h = self.mixer.decode_forward(xn, slot_active, residual=x)
        ttnn.deallocate(xn)
        out = self._ffn_block(h, slot_active, decode_bucket)
        ttnn.deallocate(h)
        return out


def make_optimized(sd, cfg, args, layer_idx, mesh_device, max_seq_len, policy=DEFAULT_POLICY, fusions=DEFAULT_FUSIONS):
    dec = OptimizedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, policy=policy, fusions=fusions
    )
    rope = make_rope(mesh_device, args, max_seq_len, fusions) if dec.is_attention else None
    return dec, rope


__all__ = ["Policy", "DEFAULT_POLICY", "BASELINE_POLICY", "OptimizedDecoder", "make_optimized", "replace"]
