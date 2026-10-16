# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B one optimized-decoder Policy: PCC (32 users, mixed lengths, traced decode) + warmed perf
at batch 1 and batch 32.

    LFM2_POLICY='{"expert_gate_up": "bfp4", ...}' LFM2_POLICY_BASE=default|baseline \\
    pytest models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_policy.py -k <kind>

Wall clock is for quick screening; reported numbers come from Tracy (LFM2_BENCH_MODE=perf_only under
``python -m tracy``, signposts PERF_PREFILL / PERF_DECODE for the batch given by LFM2_BENCH_BATCH).
Appends a JSON line to ``$LFM2_BENCH_OUT``.
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

from ..tt.model_config import TILE
from ..tt.optimized_decoder import BASELINE_POLICY, DEFAULT_POLICY, make_optimized, replace
from .harness import User
from .test_functional_decoder import LAYERS, run_case
from .test_perf_determinism import run_perf_determinism


def policy_from_env():
    base = {"default": DEFAULT_POLICY, "baseline": BASELINE_POLICY}[os.environ.get("LFM2_POLICY_BASE", "default")]
    return replace(base, **json.loads(os.environ.get("LFM2_POLICY", "{}")))


def policy_factory(policy):
    return lambda sd, cfg, args, li, md, max_seq: make_optimized(sd, cfg, args, li, md, max_seq, policy=policy)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kind", list(LAYERS))
def test_bench_policy(mesh_device, kind):
    policy = policy_from_env()
    factory = policy_factory(policy)
    mode = os.environ.get("LFM2_BENCH_MODE", "full")
    row = {
        "kind": kind,
        "policy": policy.summary(),
        "tag": os.environ.get("LFM2_TAG", ""),
        "bucket": os.environ.get("LFM2_BENCH_BUCKET") == "1",
    }
    if mode == "full":
        if os.environ.get("LFM2_BENCH_LONG") == "1":  # one long prompt in 2048-row chunks (dispatch/hybrid paths)
            users = [User(slot=5, prompt_len=2049 + 1000, decode_steps=4, seed=77)]
            res = run_case(mesh_device, LAYERS[kind], users, chunk_size=2048, decoder_factory=factory)
        else:
            lens = [1 + (37 * i) % 200 for i in range(TILE)]
            users = [User(slot=i, prompt_len=lens[i], decode_steps=3, seed=100 + i) for i in range(TILE)]
            res = run_case(mesh_device, LAYERS[kind], users, chunk_size=128, decoder_factory=factory)
        row.update(
            min_prefill_pcc=min(p for _, p, _ in res),
            min_decode_pcc=min(d for _, _, d in res),
            min_delta_prefill_pcc=min(u.delta[0] for u, _, _ in res),
            min_delta_decode_pcc=min(u.delta[1] for u, _, _ in res),
        )
    iters = int(os.environ.get("LFM2_BENCH_DECODE_ITERS", "20"))
    batches = [int(b) for b in os.environ.get("LFM2_BENCH_BATCH", "1,32").split(",")]
    for b in batches:
        # LFM2_BENCH_BUCKET=1: decode with the static bucket active_slots = b (users in slots 0..b-1)
        bucket = b if os.environ.get("LFM2_BENCH_BUCKET") == "1" and b < TILE else None
        perf = run_perf_determinism(
            mesh_device, kind, decoder_factory=factory, decode_iters=iters, active_users=b, active_slots=bucket
        )
        row[f"decode_us_b{b}"] = perf["decode_us"]
        row["prefill_ms"] = perf["prefill_ms"]
    logger.info(f"BENCH {json.dumps(row)}")
    out = Path(os.environ.get("LFM2_BENCH_OUT", "generated/lfm2_bench_policy.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    if mode == "full":
        assert row["min_prefill_pcc"] >= 0.995 and row["min_decode_pcc"] >= 0.995, row
