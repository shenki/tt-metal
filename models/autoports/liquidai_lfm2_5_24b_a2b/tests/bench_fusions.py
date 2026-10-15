# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B one fusion set: PCC (32 users, mixed lengths, traced decode) + warmed wall-clock perf.

    LFM2_FUSIONS=default|all|none|{all,default}-<f1>,<f2>|only:<f1>,<f2> \
    pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_fusions.py -k <kind>

Appends one JSON line per kind to ``$LFM2_BENCH_OUT`` (default
``generated/lfm2_bench_fusions.jsonl``).  Wall clock is used for quick A/B decisions only; the
reported before/after numbers come from Tracy + tt-perf-report.
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

from ..tt.fused_decoder import ALL_FUSIONS, DEFAULT_FUSIONS
from ..tt.model_config import TILE
from .harness import User
from .test_functional_decoder import DELTA_PCC_BAR, LAYERS, _use_real, run_case
from .test_fused_decoder import fused_factory
from .test_perf_determinism import run_perf_determinism


def parse_fusions(spec):
    if spec in ("default", "", None):
        return DEFAULT_FUSIONS
    if spec == "all":
        return frozenset(ALL_FUSIONS)
    if spec == "none":
        return frozenset()
    if spec.startswith("default+"):
        add = set(spec[len("default+") :].split(","))
        assert add <= set(ALL_FUSIONS), add - set(ALL_FUSIONS)
        return DEFAULT_FUSIONS | add
    if spec.startswith("all-") or spec.startswith("default-"):
        base, drop = spec.split("-", 1)
        drop = set(drop.split(","))
        assert drop <= set(ALL_FUSIONS), drop - set(ALL_FUSIONS)
        return (frozenset(ALL_FUSIONS) if base == "all" else DEFAULT_FUSIONS) - drop
    if spec.startswith("only:"):
        keep = set(spec[5:].split(","))
        assert keep <= set(ALL_FUSIONS), keep - set(ALL_FUSIONS)
        return frozenset(keep)
    raise ValueError(spec)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_bench_fusions(mesh_device, kind):
    spec = os.environ.get("LFM2_FUSIONS", "default")
    fusions = parse_fusions(spec)
    factory = fused_factory(fusions)
    lens = [1 + (37 * i) % 200 for i in range(TILE)]
    users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
    mode = os.environ.get("LFM2_BENCH_MODE", "full")  # full | perf_only (diagnostics)
    res = [(None, 1.0, 1.0)]
    if mode == "full":
        res = run_case(mesh_device, LAYERS[kind], users, chunk_size=128, decoder_factory=factory)
    perf = run_perf_determinism(
        mesh_device, kind, decoder_factory=factory, decode_iters=int(os.environ.get("LFM2_BENCH_DECODE_ITERS", "20"))
    )
    row = {
        "kind": kind,
        "spec": spec,
        "fusions": sorted(fusions),
        "min_prefill_pcc": min(p for _, p, _ in res),
        "min_decode_pcc": min(d for _, _, d in res),
        "min_delta_prefill_pcc": min(getattr(u, "delta", (1.0, 1.0))[0] for u, _, _ in res),
        "min_delta_decode_pcc": min(getattr(u, "delta", (1.0, 1.0))[1] for u, _, _ in res),
        **perf,
    }
    logger.info(f"BENCH {json.dumps(row)}")
    out = Path(os.environ.get("LFM2_BENCH_OUT", "generated/lfm2_bench_fusions.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    assert row["min_prefill_pcc"] >= 0.995 and row["min_decode_pcc"] >= 0.995, row
    if _use_real():
        assert row["min_delta_prefill_pcc"] >= DELTA_PCC_BAR and row["min_delta_decode_pcc"] >= DELTA_PCC_BAR, row
