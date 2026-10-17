# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repo-local driver for the packaged readiness runners (``readiness_check.run_prefill_check`` / ``run_teacher_forcing``
/ ``run_autoregressive``), unchanged runner code with two differences from their CLIs:

* the mesh is opened with the model's parameters (FABRIC_1D_RING + 8 KB router payload, ``l1_small_size`` for the CCL
  semaphores, a trace region); the packaged ``--mesh-device`` opener sets none of these;
* one generator (``build_generator(..., shared=True)``) serves all runners: the 40-layer build happens once.

Usage (from tt-metal/, environment per doc/full_model/README.md "Commands"):
    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.readiness_driver --runs prefill,tf,ar \
        --reference bringup/references/lfm2_5_24b_a2b_aime24_chat.refpt [--layers 0,2,3]
"""

import argparse
import json
import time
from pathlib import Path

from loguru import logger

MODEL_DIR = Path(__file__).resolve().parents[1]


def memory_report(gen):
    """Per-chip DRAM / L1 / trace-region usage of the built generator (full cache, warm-up and captures done)."""
    import ttnn

    def view(kind):
        v = ttnn.get_memory_view(gen.mesh, kind)
        return {
            "total_bytes_per_bank": v.total_bytes_per_bank,
            "allocated_bytes_per_bank": v.total_bytes_allocated_per_bank,
            "free_bytes_per_bank": v.total_bytes_free_per_bank,
            "largest_contiguous_free_per_bank": v.largest_contiguous_bytes_free_per_bank,
            "num_banks": v.num_banks,
        }

    out = {k: view(getattr(ttnn.BufferType, k)) for k in ("DRAM", "L1", "TRACE")}
    d = out["DRAM"]
    out["dram_allocated_gib_per_chip"] = d["allocated_bytes_per_bank"] * d["num_banks"] / 2**30
    out["dram_total_gib_per_chip"] = d["total_bytes_per_bank"] * d["num_banks"] / 2**30
    out["kv_blocks"] = gen.model.num_blocks
    out["kv_block_size"] = gen.model.block_size
    out["captured_buckets"] = [str(b) for b in gen.traces]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="prefill,tf,ar")
    ap.add_argument("--reference", type=Path, default=Path("bringup/references/lfm2_5_24b_a2b_aime24_chat.refpt"))
    ap.add_argument("--layers", default=None, help="comma list (reduced probe), default all 40")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ar-out", type=Path, default=MODEL_DIR / "readiness_autoregressive")
    ap.add_argument("--json", type=Path, default=None, help="write the results here")
    ap.add_argument("--trace-region-mb", type=int, default=768)
    a = ap.parse_args()

    from readiness_check.run_autoregressive import DEFAULT_PROMPT_FILE, run_autoregressive
    from readiness_check.run_prefill_check import run_prefill_check
    from readiness_check.run_teacher_forcing import run_teacher_forcing

    import ttnn

    from ..tt.generator import build_generator, open_mesh
    from ..tt.model_config import HF_MODEL_ID

    layers = None if a.layers is None else [int(x) for x in a.layers.split(",")]
    build_kwargs = dict(shared=True, layer_indices=layers)
    mesh = open_mesh(a.trace_region_mb << 20)
    results = {"layers": layers or "all", "reference": str(a.reference)}
    try:
        t0 = time.perf_counter()
        gen = build_generator(model_dir=MODEL_DIR, mesh_device=mesh, **build_kwargs)
        results["build_prepare_s"] = time.perf_counter() - t0
        results["model_build_s"] = gen.model.build_s
        for run in a.runs.split(","):
            t = time.perf_counter()
            if run == "prefill":
                r = run_prefill_check(
                    model_dir=MODEL_DIR,
                    reference_path=a.reference.resolve(),
                    mesh_device=mesh,
                    build_kwargs=build_kwargs,
                )
            elif run == "tf":
                r = run_teacher_forcing(
                    model_dir=MODEL_DIR,
                    reference_path=a.reference.resolve(),
                    mesh_device=mesh,
                    build_kwargs=build_kwargs,
                )
            elif run == "ar":
                r = {
                    k: str(v)
                    for k, v in run_autoregressive(
                        model_dir=MODEL_DIR,
                        hf_model_id=HF_MODEL_ID,
                        prompt_file=Path(DEFAULT_PROMPT_FILE),
                        mesh_device=mesh,
                        output_dir=a.ar_out.resolve(),
                        max_new_tokens=a.max_new_tokens,
                        build_kwargs=build_kwargs,
                    ).items()
                }
            elif run == "qual":
                from .qualitative import run_tt

                r = run_tt(gen)
            elif run == "memory":
                r = memory_report(gen)
            elif run == "perf":
                from .perf_full_model import run_perf

                r = run_perf(gen)
            else:
                raise ValueError(run)
            results[run] = r
            results[f"{run}_wall_s"] = time.perf_counter() - t
            results[f"{run}_counters"] = dict(gen.counters)
            results[f"{run}_last_perf"] = dict(gen.last_perf)
            logger.info(f"{run}: {r}")
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(results, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
