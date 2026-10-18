# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the plugin's tests/tt suite one test at a time against a live server whose adapter writes LFM2_VLLM_EVENTS and
attribute every test to the sampling path its steps took (device / host prefill and decode counts, resident decode
steps), so the suite result says which tests exercised the on-device sampler.

    python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.vllm_suite_sampling_paths --events <jsonl> --out <json>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--ignore", default="spec")
    ap.add_argument("--fail-dir", default=None, help="write the pytest output of every non-passing test here")
    a = ap.parse_args()
    plugin = Path(os.environ["VLLM_TT_PLUGIN_ROOT"])
    root = plugin / "tests" / "tt"
    common = [
        f"--tt-server-url={a.url}",
        "--tt-model-name=LiquidAI/LFM2.5-24B-A2B",
        "--tt-max-num-seqs=32",
        "--tt-chunked-prefill-budget=2048",
    ]
    ids = subprocess.run(
        [sys.executable, "-m", "pytest", str(root), "--collect-only", "-q", f"--ignore={root / a.ignore}", *common],
        capture_output=True,
        text=True,
        cwd=plugin,
    ).stdout.split("\n")
    ids = [i for i in ids if "::" in i]
    ev = Path(a.events)
    rows = []
    for nid in ids:
        n0 = sum(1 for _ in open(ev))
        r = subprocess.run(
            [sys.executable, "-m", "pytest", nid, "-q", "-rs", *common], capture_output=True, text=True, cwd=plugin
        )
        tail = [
            l for l in r.stdout.split("\n") if (" passed" in l or " failed" in l or " skipped" in l or " error" in l)
        ][-1:] or [""]
        new = [json.loads(l) for l in list(open(ev))[n0:]]
        pre = [e for e in new if e["kind"] == "prefill"]
        dec = [e for e in new if e["kind"] == "decode"]
        rows.append(
            dict(
                test=nid.split("tests/tt/")[-1],
                outcome="passed"
                if r.returncode == 0 and "skipped" not in tail[0]
                else ("skipped" if "skipped" in tail[0] else "failed"),
                summary=tail[0],
                prefill_device=sum(e["device_sampling"] for e in pre),
                prefill_host=sum(not e["device_sampling"] for e in pre),
                decode_device=sum(e["device_sampling"] for e in dec),
                decode_host=sum(not e["device_sampling"] for e in dec),
                decode_device_resident=sum(e["device_sampling"] and not e["reload_inputs"] for e in dec),
            )
        )
        if a.fail_dir and rows[-1]["outcome"] != "passed":
            Path(a.fail_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.fail_dir) / (rows[-1]["test"].replace("/", "_").replace("::", "__") + ".log")).write_text(r.stdout)
        print(rows[-1], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    from collections import Counter

    print(Counter(r["outcome"] for r in rows))
    print("tests with device-sampled steps:", sum(1 for r in rows if r["prefill_device"] + r["decode_device"] > 0))


if __name__ == "__main__":
    main()
