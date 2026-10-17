# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-profile workload of the reduced full-model probe (layers 0, 2, 3: one layer of each kind + embedding + final
norm + LM head + split sampling), for Tracy / ``tt-perf-report`` (never on all 40 layers: profiler buffer limits).

Signposted windows: PERF_PREFILL (one traced 128-token prefill incl. the prefill sampling replay), PERF_DECODE
(DECODE_ITERS token-out steps = model trace + sampling/history trace replays, bucket 1) and PERF_DECODE_B8 (same, slots
0..7 active, bucket 8).  Run via
``doc/full_model/repro/tracy_probe.sh``."""

import torch

import ttnn

from ..tt.generator import build_generator, open_mesh

PROBE = (0, 2, 3)
DECODE_ITERS = 5

try:
    from tracy import signpost
except ImportError:  # profiler not built in

    def signpost(header, message=None):
        pass


def test_probe_profile():
    from .perf_full_model import _prompt

    mesh = open_mesh()
    try:
        gen = build_generator(mesh_device=mesh, layer_indices=PROBE, max_seq_len=8192)
        ids = _prompt(gen, 128)
        gen.generate(ids, 8)  # captures bucket 1, warms everything once more
        gen.reset()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush: build / warm-up ops would overflow the device profiler buffers
        signpost("PERF_PREFILL")
        first = gen.prefill_forward(
            torch.tensor(ids).reshape(1, -1), page_table=gen.slot_table[:1], prompt_lens=[len(ids)], slots=[0]
        )
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        pos = torch.full((32,), -1)
        pos[0] = len(ids)
        gen.decode_forward(tokens=first, start_pos=pos, page_table=gen.slot_table, read_from_device=False)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE")
        for _ in range(DECODE_ITERS):
            gen.decode_forward(read_from_device=False, record_history=True)  # token-out: sampling + history trace
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        # bucket 8: slots 0..7 active (the few-users regime)
        gen.reset()
        for slot in range(8):
            t = torch.tensor(ids[: 100 + slot]).reshape(1, -1)
            f = gen.prefill_forward(
                t, page_table=gen.slot_table[slot : slot + 1], prompt_lens=[t.shape[1]], slots=[slot]
            )
            pos = gen.pos_host.clone()
            pos[slot] = t.shape[1]
            gen._refresh_positions(pos)
            gen._set_slot_tokens([slot], [int(f[0])])
        gen.decode_forward(read_from_device=False, record_history=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE_B8")
        for _ in range(DECODE_ITERS):
            gen.decode_forward(read_from_device=False, record_history=True)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_B8_END")
        ttnn.ReadDeviceProfiler(mesh)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)
