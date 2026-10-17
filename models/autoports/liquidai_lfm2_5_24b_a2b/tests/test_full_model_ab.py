# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise A/B of stage-7 graph changes on the reduced probe (layers 0, 2, 3 + terminal path), eager decode before any
capture (so the alternate variant's programs never compile next to live traces):

* D1: b-row page table / positions sliced once per step (``Lfm2Model.hoist_bucket_slices``) vs inside every attention layer.
"""

import os

import pytest
import torch

import ttnn

from ..tt.generator import build_generator, open_mesh
from .test_full_model import _start, text_ids


@pytest.fixture(scope="module")
def gen():
    os.environ["LFM2_EAGER_DEBUG"] = "1"
    mesh = open_mesh()
    g = build_generator(mesh_device=mesh, layer_indices=(0, 2, 3), max_seq_len=8192)
    yield g
    g.close()
    ttnn.close_mesh_device(mesh)


def _eager_step_logits(gen, slots, hoist):
    """Prefill each slot, then one eager decode step; returns the vocab-sharded logits of the active rows."""
    gen.reset()
    gen._refresh_table(gen.slot_table)
    for i, s in enumerate(slots):
        _start(gen, s, text_ids(gen, 30 + 7 * i, i))
    gen.model.hoist_bucket_slices = hoist
    try:
        lg = gen.model.decode_forward_device(gen_bucket(slots))
        out = gen.model.logits_to_host(lg)[0, 0, list(slots)]
        ttnn.deallocate(lg)
    finally:
        gen.model.hoist_bucket_slices = True
    return out


def gen_bucket(slots):
    from ..tt.optimized_decoder import decode_bucket

    return decode_bucket(slots)


@pytest.mark.parametrize("slots", [(0,), (0, 3), (0, 5, 7), tuple(range(32))], ids=["b1", "b4", "b8", "b32"])
def test_hoisted_bucket_slices_bitwise(gen, slots):
    a = _eager_step_logits(gen, slots, hoist=True)
    b = _eager_step_logits(gen, slots, hoist=False)
    assert torch.equal(a, b), (a - b).abs().max()
