# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder correctness: the functional-decoder test matrix run through ``FusedDecoder``
with ``DEFAULT_FUSIONS`` (no functional fallback), plus perf / determinism / no-host-fallback.

Same harness and HF reference as ``test_functional_decoder.py`` (paged chunked prefill, traced
batch-32 decode, interleaved prefill/decode with one trace, full 131072 context).
"""

import pytest

from ..tt.fused_decoder import DEFAULT_FUSIONS, FusedDecoder, make_rope
from ..tt.model_config import TILE, Lfm2MoeArgs
from .harness import User
from .test_functional_decoder import LAYERS, _assert, hf_config, run_case, run_interleaved


def fused_factory(fusions=DEFAULT_FUSIONS):
    def factory(sd, cfg, args, layer_idx, mesh_device, max_seq_len):
        dec = FusedDecoder.from_state_dict(
            sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, fusions=fusions
        )
        assert dec.fusions == frozenset(fusions)
        rope = make_rope(mesh_device, args, max_seq_len, fusions) if dec.is_attention else None
        return dec, rope

    return factory


FUSED = fused_factory()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize(
    "prompt_len, chunk",
    [(1, 2048), (31, 2048), (32, 2048), (64, 2048), (65, 2048), (300, 128), (300, 2048), (2049, 2048)],
)
def test_fused_single_user(mesh_device, kind, prompt_len, chunk):
    users = [User(slot=5, prompt_len=prompt_len, decode_steps=4, seed=prompt_len)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=chunk, decoder_factory=FUSED))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_fused_batch32(mesh_device, kind):
    lens = [1 + (37 * i) % 200 for i in range(TILE)]
    users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
    _assert(run_case(mesh_device, LAYERS[kind], users, chunk_size=128, decoder_factory=FUSED))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_fused_slot_reuse(mesh_device, kind):
    fg = [
        User(slot=7, prompt_len=700, decode_steps=2, seed=1),
        User(slot=7, prompt_len=33, decode_steps=3, seed=2, join_delay=2),
    ]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=256, decoder_factory=FUSED))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_fused_increasing_lengths_same_instance(mesh_device, kind):
    lens = [2047, 2048, 2049, 4100, 32, 33]
    fg = [User(slot=9, prompt_len=n, decode_steps=2, seed=500 + n, join_delay=i % 3) for i, n in enumerate(lens)]
    _assert(run_interleaved(mesh_device, LAYERS[kind], fg, chunk_size=2048, decoder_factory=FUSED))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_fused_full_context(mesh_device, kind):
    ctx = Lfm2MoeArgs.from_hf_config(hf_config()).max_position_embeddings
    users = [
        User(slot=9, prompt_len=ctx - 3, decode_steps=3, seed=7),
        User(slot=30, prompt_len=77, decode_steps=1, seed=8),
    ]
    _assert(
        run_case(
            mesh_device, LAYERS[kind], users, chunk_size=2048, max_seq_len=ctx, reserve_seqs=TILE, decoder_factory=FUSED
        )
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_fused_perf_determinism_no_fallback(mesh_device, kind):
    from .test_perf_determinism import run_perf_determinism

    run_perf_determinism(mesh_device, kind, decoder_factory=FUSED)
