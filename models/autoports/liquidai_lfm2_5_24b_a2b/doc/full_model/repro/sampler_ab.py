# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sampler A/B on the reduced probe (one model, several Lfm2Generator samplers over it): wall time of the traced
sampling replay alone (100 replays) and of model + sampling, plus token equality over a 64-token greedy generation.
Variants: TTSampling split (32256-wide shards, pad_logits_to_power_of_2 False / True) and force-argmax."""

import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.perf_full_model import _prompt, _time_replays
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator import Lfm2Generator, build_generator, open_mesh

mesh = open_mesh()
try:
    # base generator = the unpadded split sampler; every later variant is built only after the previous one's last
    # replay (no earlier generator's trace is replayed while a later generator's buffers are live)
    base = build_generator(mesh_device=mesh, layer_indices=(0, 2, 3), max_seq_len=8192, pad_logits_to_power_of_2=False)
    ids = _prompt(base, 128)
    ref = None
    for name, kw in (
        ("split_unpadded", dict()),
        ("split_pad_pow2", dict(pad_logits_to_power_of_2=True)),
        ("force_argmax", dict(sampling_strategy="argmax")),
    ):
        g = base if name == "split_unpadded" else Lfm2Generator(base.model, **kw)
        if g is not base:
            g.kv, g.owns_cache, g.slot_table, g.bps = base.kv, True, base.slot_table, base.bps
            g.prepare()
        g.reset()
        out = g.generate(ids, 64, stop_on_eos=False)
        ref = ref or out
        tid, _, sid = g.traces[1]
        s = _time_replays(g, [sid], 100)
        ms = _time_replays(g, [tid, sid], 100)
        print(
            f"SAMPLER {name:16s} sampling {s*1e3:7.1f} us  model+sampling {ms*1e3:7.1f} us  same tokens {out == ref}",
            flush=True,
        )
        if g is not base:
            g.release_traces()
    base.close()
finally:
    ttnn.close_mesh_device(mesh)
