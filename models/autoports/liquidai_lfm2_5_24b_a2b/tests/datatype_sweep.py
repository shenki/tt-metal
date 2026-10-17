# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Datatype sweep driver (stage 8).

Parent (one config per child process, sequential: one device process at a time):
    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.datatype_sweep --only B0,E1 [--perf B0]
reads ``doc/datatype_sweep/configs/matrix.json`` (list of {id, priority, policy_overrides, plan_overrides,
layer_exceptions, head, cache, note}), writes ``configs/<id>.json`` (a ``tt.precision.PrecisionConfig`` file, the
selected-config schema), runs the child, evaluates ``thresholds.json`` and upserts the row in ``sweep_results.json`` /
``sweep_results.csv``.

Child (``--run-one configs/<id>.json --json logs/<id>.json``): builds through the normal construction path
(``build_generator(..., precision=<file>)``) and runs, in this order, the readiness prefill check and traced teacher
forcing on the main AIME24 reference (first request: incl. deferred captures), teacher forcing again (warm), teacher
forcing on the extended references, a memory report and optionally the stage-7 token-out perf harness (last).
Ranking metric: main-reference warm teacher-forcing decode t/s/u (``run_teacher_forcing``, traced bucket-1 decode).
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
SWEEP = MODEL_DIR / "doc" / "datatype_sweep"
REPO = MODEL_DIR.parents[2]
REFS = REPO / "bringup" / "references"
MAIN_REF = REFS / "lfm2_5_24b_a2b_aime24_chat.refpt"
EXT_REFS = [REFS / f"lfm2_5_24b_a2b_aime24_chat_p{i}.refpt" for i in (1, 2, 3)]
REGIME = (
    "readiness run_teacher_forcing (traced bucket-1 decode, token-out sampling path, per-step token readback), batch 1, "
    "AIME24 chat-template refs (main idx 0: 157-token prompt + 100 tokens; extended idx 1-3), 1x4 TP4, trace region 768 MiB"
)
HARDWARE = "QuietBox 4x Blackhole p150b"
MESH = "1x4 FABRIC_1D_RING, router payload 8192 B, l1_small 24576, trace region 768 MiB"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------
# child
# ------------------------------------------------------------------------------------------------
def run_one(cfg_path, out_json, perf):
    from loguru import logger
    from readiness_check.run_prefill_check import run_prefill_check
    from readiness_check.run_teacher_forcing import run_teacher_forcing

    import ttnn

    from ..tt.generator import build_generator, open_mesh
    from .readiness_driver import memory_report

    res = {"config": str(cfg_path)}
    kw = dict(shared=True, precision=str(cfg_path))
    mesh = open_mesh(768 << 20)
    try:
        t0 = time.perf_counter()
        gen = build_generator(model_dir=MODEL_DIR, mesh_device=mesh, **kw)
        res["build_prepare_s"] = time.perf_counter() - t0
        res["model_build_s"] = gen.model.build_s
        res["weight_cache"] = gen.model.weight_cache
        res["precision"] = {
            "config_id": gen.model.precision.config_id,
            "source": gen.model.precision_source,
            "policy": gen.model.precision.policy.summary(),
            "plan": gen.model.precision.plan.summary(),
            "layer_exceptions": gen.model.precision.layer_exceptions,
            "head": gen.model.precision.head,
        }
        res["prefill"] = run_prefill_check(
            model_dir=MODEL_DIR, reference_path=MAIN_REF, mesh_device=mesh, build_kwargs=kw
        )
        res["tf_first"] = run_teacher_forcing(
            model_dir=MODEL_DIR, reference_path=MAIN_REF, mesh_device=mesh, build_kwargs=kw
        )
        c0 = dict(gen.counters)
        res["tf"] = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=MAIN_REF, mesh_device=mesh, build_kwargs=kw)
        res["tf_counters"] = {k: v - c0.get(k, 0) for k, v in gen.counters.items() if v != c0.get(k, 0)}
        res["tf_last_perf"] = dict(gen.last_perf)
        res["tf_ext"] = {}
        for ref in EXT_REFS:
            res["tf_ext"][ref.name] = run_teacher_forcing(
                model_dir=MODEL_DIR, reference_path=ref, mesh_device=mesh, build_kwargs=kw
            )
        res["memory"] = memory_report(gen)
        if perf:
            from .perf_full_model import run_perf

            res["perf"] = run_perf(gen)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)
    Path(out_json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    logger.info(f"datatype sweep child done: {out_json}")


# ------------------------------------------------------------------------------------------------
# parent
# ------------------------------------------------------------------------------------------------
def compact_dtypes(policy, plan, head, exc):
    p = policy
    s = (
        f"attn={p['attn_qkv']}/{p['attn_out']} conv={p['conv_in']}/{p['conv_out']} dense={p['dense_w13']}/{p['dense_w2']} "
        f"exp_prefill={p['expert_gate_up']}/{p['expert_down']} exp_decode={p['expert_gate_up_decode'] or p['expert_gate_up']}"
        f"/{p['expert_down_decode'] or p['expert_down']} head={head['dtype']} kv={p['kv_cache']} ccl={plan['ccl_dtype']}"
        f"{'' if plan.get('ccl_moe', True) else '(no moe)'}"
    )
    return s + (f" exceptions={exc}" if exc else "")


def compact_fidelities(policy, head, exc):
    p = policy
    s = (
        f"attn={p['attn_fidelity']} conv={p['conv_fidelity']} dense={p['dense_fidelity']} expert={p['expert_fidelity']} "
        f"dispatch={p['dispatch_fidelity']} sdpa_decode={p['sdpa_decode_fidelity']} sdpa_prefill={p['sdpa_prefill_fidelity']} "
        f"head={head['fidelity']}"
    )
    return s


def make_row(entry, child, cmd, log, thresholds, error=None):
    row = {
        "config_id": entry["id"],
        "priority": entry.get("priority"),
        "note": entry.get("note", ""),
        "status": "error",
        "pass_reason": "",
        "policy_overrides": entry.get("policy_overrides", {}),
        "plan_overrides": entry.get("plan_overrides", {}),
        "layer_exceptions": entry.get("layer_exceptions", {}),
        "head": entry.get("head", {}),
        "measurement_regime": REGIME,
        "reference": {
            "main": str(MAIN_REF),
            "main_sha256": sha256(MAIN_REF),
            "ext": {r.name: sha256(r) for r in EXT_REFS},
        },
        "command": cmd,
        "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip(),
        "worktree_tt_md5": hashlib.md5(
            subprocess.check_output(["git", "diff", "HEAD", "--", str(MODEL_DIR / "tt")], cwd=REPO)
        ).hexdigest()[:12],
        "hardware": HARDWARE,
        "mesh": MESH,
        "env": {k: v for k, v in os.environ.items() if k.startswith("LFM2_")},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "log": str(log),
        "error": error,
    }
    if child is None:
        return row
    pr = child["precision"]
    row["dtype_policy"] = compact_dtypes(pr["policy"], pr["plan"], pr["head"], pr["layer_exceptions"])
    row["fidelity_policy"] = compact_fidelities(pr["policy"], pr["head"], pr["layer_exceptions"])
    row["policy"] = pr["policy"]
    row["plan"] = pr["plan"]
    row["head_resolved"] = pr["head"]
    tf, pf = child["tf"][0], child["prefill"][0]
    row.update(
        top1=tf["top1"],
        top5=tf["top5"],
        top100=tf["top100"],
        matches_top1=tf["matches_top1"],
        matches_top5=tf["matches_top5"],
        matches_top100=tf["matches_top100"],
        total=tf["total"],
        prefill_top1=pf["top1"],
        prefill_top5=pf["top5"],
        prefill_top100=pf["top100"],
        ttft_ms_first_request=child["tf_first"][0].get("ttft_ms"),
        ttft_ms_warm=tf.get("ttft_ms"),
        tf_decode_t_s_u=tf.get("decode_t/s/u"),
        tf_decode_ms=1e3 / tf["decode_t/s/u"] if tf.get("decode_t/s/u") else None,
        tf_decode_t_s_u_first=child["tf_first"][0].get("decode_t/s/u"),
    )
    ext = [e[0] for e in child["tf_ext"].values()]
    tot = tf["total"] + sum(e["total"] for e in ext)
    for k in ("top1", "top5", "top100"):
        row[f"ext_{k}"] = (tf[f"matches_{k}"] + sum(e[f"matches_{k}"] for e in ext)) / tot
    row["ext_per_ref"] = {n: {k: e[0][k] for k in ("top1", "top5", "top100")} for n, e in child["tf_ext"].items()}
    row["ext_total"] = tot
    c = child.get("tf_counters", {})
    row["trace_verified"] = {
        "enable_trace": True,
        "model_replays": c.get("model_replays"),
        "sampling_replays": c.get("sampling_replays"),
        "trace_captures_during_window": c.get("trace_captures", 0),
        "eager_decode_steps": c.get("eager_decode_steps", 0),
    }
    row["build_s"] = child["model_build_s"]
    row["cache"] = child.get("weight_cache")
    row["dram_allocated_gib"] = child["memory"]["dram_allocated_gib_per_chip"]
    if "perf" in child:
        p = child["perf"]
        row["token_out"] = {
            "ttft_ms": p["ttft_ms"],
            "ttft_ms_by_prompt_len": p["ttft_ms_by_prompt_len"],
            "token_out_tok_s_u": p["token_out_tok_s_u"],
            "token_out_ms_step": p["token_out_ms_step"],
            "teacher_forcing_tok_s_u": p["teacher_forcing_tok_s_u"],
            "batch_token_out_ms_step": p["batch_token_out_ms_step"],
        }
    t = thresholds
    fails = []
    for scope, pre in (("main", ""), ("extended", "ext_")):
        for k in ("top1", "top5", "top100"):
            if row[f"{pre}{k}"] + 1e-9 < t[f"{k}_min"]:
                fails.append(f"{scope} {k} {row[f'{pre}{k}']:.4f} < {t[f'{k}_min']}")
    tv = row["trace_verified"]
    if tv["model_replays"] != tf["total"] - 1 or tv["eager_decode_steps"]:
        fails.append(f"trace verification {tv}")
    row["status"] = "fail" if fails else "pass"
    row["pass_reason"] = "; ".join(fails) if fails else "all gates met"
    row["error"] = None
    return row


FLAT_KEYS = [
    "config_id", "priority", "status", "pass_reason", "dtype_policy", "fidelity_policy", "top1", "top5", "top100",
    "prefill_top1", "prefill_top5", "prefill_top100", "ext_top1", "ext_top5", "ext_top100", "ext_total",
    "ttft_ms_first_request", "ttft_ms_warm", "tf_decode_t_s_u", "tf_decode_ms", "tf_decode_t_s_u_first",
    "token_out.token_out_tok_s_u", "token_out.ttft_ms", "build_s.layers", "dram_allocated_gib", "measurement_regime",
    "command", "commit", "worktree_tt_md5", "hardware", "mesh", "timestamp", "note", "error",
]  # fmt: skip


def write_csv(rows):
    with open(SWEEP / "sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FLAT_KEYS)
        for r in rows:
            vals = []
            for k in FLAT_KEYS:
                v = r
                for part in k.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
                vals.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
            w.writerow(vals)


def upsert(row):
    path = SWEEP / "sweep_results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [r for r in rows if r["config_id"] != row["config_id"]] + [row]
    path.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    write_csv(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--matrix", default=str(SWEEP / "configs" / "matrix.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--perf", default="", help="comma list of config ids that also run the token-out perf harness")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--with-perf", action="store_true", help="(child) run the perf harness last")
    a = ap.parse_args()
    if a.run_one:
        return run_one(a.run_one, a.json, a.with_perf)

    from ..tt.precision import PrecisionConfig

    matrix = json.loads(Path(a.matrix).read_text())
    thresholds = json.loads((SWEEP / "thresholds.json").read_text())
    ids = a.only.split(",") if a.only else [e["id"] for e in matrix]
    perf_ids = set(filter(None, a.perf.split(",")))
    for cid in ids:
        entry = next(e for e in matrix if e["id"] == cid)
        cfg = SWEEP / "configs" / f"{cid}.json"
        PrecisionConfig(
            config_id=cid,
            policy_overrides=entry.get("policy_overrides", {}),
            plan_overrides=entry.get("plan_overrides", {}),
            layer_exceptions=entry.get("layer_exceptions", {}),
            head=entry.get("head", {}),
        ).to_json(cfg, note=entry.get("note", ""))
        out = SWEEP / "logs" / f"{cid}.json"
        log = SWEEP / "logs" / f"{cid}.log"
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.liquidai_lfm2_5_24b_a2b.tests.datatype_sweep",
            "--run-one",
            str(cfg),
            "--json",
            str(out),
        ]
        if cid in perf_ids:
            cmd.append("--with-perf")
        env = dict(os.environ)
        if not entry.get("cache", False):
            env["LFM2_WEIGHT_CACHE"] = "0"  # one-off candidate: no 54 GB cache write
        print(f"[sweep] {cid}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(
                    cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        child = json.loads(out.read_text()) if (rc == 0 and out.exists()) else None
        err = None if child is not None else f"child exit {rc} (see {log})"
        row = make_row(entry, child, " ".join(cmd[1:]), log, thresholds, error=err)
        row["wall_s"] = time.time() - t0
        upsert(row)
        print(
            f"[sweep] {cid}: {row['status']} top1/5/100 {row.get('top1')}/{row.get('top5')}/{row.get('top100')} "
            f"ext {row.get('ext_top1')}/{row.get('ext_top5')}/{row.get('ext_top100')} TF {row.get('tf_decode_t_s_u')} t/s/u "
            f"({row['pass_reason'] or err})",
            flush=True,
        )


if __name__ == "__main__":
    main()
