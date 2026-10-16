# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared decoder-layer test harness: weights, inputs, paged-cache bookkeeping and the
prefill/decode drivers used by the functional-decoder tests.

Host-side work (input construction, page tables, selection tensors, PCC) happens here at the
explicit test boundary; the decoder's prefill/decode passes themselves run only device ops.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn

from ..tt.functional_decoder import ConvPrefillInputs
from ..tt.model_config import HF_MODEL_ID, HF_REVISION, TILE

HERE = Path(__file__).parent
STATS_PATH = HERE / "weight_stats.json"
BLOCK_SIZE = 64
PAGE_TABLE_ALIGN = 8


def pad_page_table_row(blocks, width=None):
    """int32 page-table row padded with scratch block 0 to ``width`` (default: next multiple of 8)."""
    width = width or math.ceil(max(len(blocks), 1) / PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN
    row = torch.zeros(width, dtype=torch.int32)
    row[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return row


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def local_model_path():
    """Snapshot dir of the pinned revision if every shard is present locally, else None."""
    from huggingface_hub import snapshot_download

    try:
        p = Path(snapshot_download(HF_MODEL_ID, revision=HF_REVISION, local_files_only=True))
    except Exception:
        return None
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        return None
    files = set(json.load(open(idx))["weight_map"].values())
    return p if all((p / f).exists() for f in files) else None


def hf_config():
    from transformers import AutoConfig

    path = local_model_path()
    if path is not None:
        return AutoConfig.from_pretrained(path)
    return AutoConfig.from_pretrained(HF_MODEL_ID, revision=HF_REVISION)


def synthetic_layer_state_dict(layer_idx, seed=0):
    """Deterministic weights drawn from recorded real-weight stats (same keys/shapes/dtypes)."""
    stats = json.load(open(STATS_PATH))
    gen = torch.Generator().manual_seed(seed + 1000 * layer_idx)
    src = stats["layers"][stats["layer_kind_source"][str(layer_idx)]]
    sd = {}
    for name, s in src.items():
        t = torch.randn(s["shape"], generator=gen) * s["std"] + s["mean"]
        sd[name] = t.to(getattr(torch, s["dtype"].replace("torch.", "")))
    return sd


def layer_state_dict(layer_idx, real):
    if real:
        from ..tt.weights import load_layer_state_dict

        path = local_model_path()
        assert path is not None, "real weights requested but the pinned checkpoint is not fully downloaded"
        return load_layer_state_dict(path, layer_idx)
    return synthetic_layer_state_dict(layer_idx)


def layer_inputs(num_tokens, hidden, seed, real_embed=None):
    """Decoder-input-like activations: embedding rows of random tokens when real embeddings are
    available, else gaussian rows with the recorded embedding std."""
    gen = torch.Generator().manual_seed(seed)
    if real_embed is not None:
        ids = torch.randint(0, real_embed.shape[0], (num_tokens,), generator=gen)
        return real_embed[ids].float()
    std = json.load(open(STATS_PATH))["embed_std"]
    return torch.randn(num_tokens, hidden, generator=gen) * std


@dataclass
class User:
    slot: int
    prompt_len: int
    decode_steps: int
    seed: int
    x: torch.Tensor = None  # [prompt_len + decode_steps, H]
    blocks: list = field(default_factory=list)
    ref: torch.Tensor = None
    tt_prefill: torch.Tensor = None
    tt_decode: list = field(default_factory=list)
    join_delay: int = 0  # decode replays (for other slots) between this request's prefill and its first decode
    delta: tuple = (1.0, 1.0)  # (prefill, decode) PCC of out - x, filled by the test


class PagedLayout:
    """Random block assignment per user; block 0 is a scratch block owned by nobody."""

    def __init__(self, users, max_seq_len, block_size=BLOCK_SIZE, seed=0, reserve_seqs=None):
        """``reserve_seqs``: allocate pages for this many full-length sequences (>= len(users)),
        e.g. 32 to prove the full batch x full context cache fits."""
        self.block_size = block_size
        self.blocks_per_seq = math.ceil(max_seq_len / block_size)
        # page-table rows must be a multiple of 32 bytes (8 int32 entries); padding entries -> scratch block 0
        self.max_blocks_per_seq = math.ceil(self.blocks_per_seq / PAGE_TABLE_ALIGN) * PAGE_TABLE_ALIGN
        need = self.blocks_per_seq * max(len(users), reserve_seqs or 0)
        self.num_blocks = max(need + 1, self.max_blocks_per_seq)  # update_cache: table width <= num_blocks
        perm = (torch.randperm(need, generator=torch.Generator().manual_seed(seed)) + 1).tolist()
        for i, u in enumerate(users):
            u.blocks = perm[i * self.blocks_per_seq : (i + 1) * self.blocks_per_seq]

    def user_table(self, user):
        return self._row(user.blocks).reshape(1, -1)

    def _row(self, blocks):
        return pad_page_table_row(blocks, self.max_blocks_per_seq)

    def decode_table(self, users, batch=TILE):
        t = torch.zeros(batch, self.max_blocks_per_seq, dtype=torch.int32)
        for u in users:
            t[u.slot] = self._row(u.blocks)
        return t


def to_dev(t, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def from_dev(t, mesh_device):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def run_prefill(decoder, rope, mesh_device, layout, user, chunk_size, continuation_from=0, after_chunk=None):
    """Chunked prefill of ``user.x[continuation_from:prompt_len]`` -> logical rows [n, H].

    ``after_chunk()`` (optional) runs after every chunk, e.g. decode replays for other slots.
    All per-chunk device inputs are released before it runs (trace-allocation safety)."""
    H = user.x.shape[-1]
    outs = []
    pos = continuation_from
    while pos < user.prompt_len:
        n = min(chunk_size, user.prompt_len - pos)
        s_pad = math.ceil(n / TILE) * TILE
        x = torch.zeros(1, 1, s_pad, H)
        x[0, 0, :n] = user.x[pos : pos + n]
        tt_x = to_dev(x, mesh_device)
        kw = {}
        if decoder.is_attention:
            first_block = pos // layout.block_size
            nblk = math.ceil(s_pad / layout.block_size)
            chunk_table = pad_page_table_row(user.blocks[first_block : first_block + nblk]).reshape(1, -1)
            kw = dict(
                rope=rope,
                chunk_start=pos,
                page_table=to_dev(layout.user_table(user), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_page_table=to_dev(chunk_table, mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_start_tensor=to_dev(
                    torch.tensor([pos], dtype=torch.int32), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
            )
        else:
            kw = dict(
                conv_inputs=ConvPrefillInputs.create(
                    mesh_device,
                    slot=user.slot,
                    valid_len=n,
                    padded_len=s_pad,
                    continuation=pos > 0,
                    windowed=getattr(decoder, "conv_prefill_windowed", False),
                )
            )
        out = decoder.prefill_forward(tt_x, **kw)
        outs.append(from_dev(out, mesh_device)[0, 0, :n])
        ttnn.deallocate(out)
        del tt_x, kw
        pos += n
        if after_chunk is not None:
            after_chunk()
    return torch.cat(outs, dim=0)


class DecodeInputs:
    """Persistent decode inputs (stable addresses for trace capture); refreshed by host copies."""

    def __init__(self, mesh_device, H, max_blocks_per_seq, batch=TILE, active_slots=None):
        self.mesh_device = mesh_device
        self.active_slots = active_slots  # static decode bucket (optimized decoder): only slots < b active
        self.x = to_dev(torch.zeros(1, 1, batch, H), mesh_device)
        self.cur_pos = to_dev(
            torch.full((batch,), -1, dtype=torch.int32), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.rope_pos = to_dev(
            torch.zeros(1, batch, dtype=torch.int32), mesh_device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.slot_active = to_dev(torch.zeros(1, 1, batch, 1), mesh_device)
        self.slot_mask = to_dev(torch.zeros(1, 1, batch, H), mesh_device)  # slot_active broadcast over H
        self.page_table = to_dev(
            torch.zeros(batch, max_blocks_per_seq, dtype=torch.int32), mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )

    def update(self, x, cur_pos, page_table):
        if self.active_slots is not None:
            from ..tt.optimized_decoder import validate_decode_bucket

            validate_decode_bucket(cur_pos, self.active_slots)
        host = lambda t, dt, lay: ttnn.from_torch(
            t, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(host(x, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.x)
        ttnn.copy_host_to_device_tensor(host(cur_pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.cur_pos)
        rp = cur_pos.clamp(min=0).reshape(1, -1)
        ttnn.copy_host_to_device_tensor(host(rp, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.rope_pos)
        ttnn.copy_host_to_device_tensor(host(page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.page_table)
        active = (cur_pos >= 0).to(torch.float32).reshape(1, 1, -1, 1)
        ttnn.copy_host_to_device_tensor(host(active, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.slot_active)
        wide = active.expand(*active.shape[:-1], self.slot_mask.shape[-1]).contiguous()
        ttnn.copy_host_to_device_tensor(host(wide, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.slot_mask)


def decode_step(decoder, rope, inp: DecodeInputs):
    bucket = {} if inp.active_slots is None else {"active_slots": inp.active_slots}
    if getattr(decoder, "accepts_slot_mask", False):
        bucket["slot_mask"] = inp.slot_mask
    if decoder.is_attention:
        if inp.active_slots is not None and getattr(decoder, "bucketed_rope", False):
            cos, sin = rope.decode_cos_sin(inp.rope_pos, batch=inp.active_slots)  # b-user cos/sin
        else:
            cos, sin = rope.decode_cos_sin(inp.rope_pos)
        return decoder.decode_forward(
            inp.x,
            rope=rope,
            rope_cos=cos,
            rope_sin=sin,
            cur_pos=inp.cur_pos,
            page_table=inp.page_table,
            slot_active=inp.slot_active,
            **bucket,
        )
    return decoder.decode_forward(inp.x, slot_active=inp.slot_active, **bucket)
