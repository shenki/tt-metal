# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The selected precision appears in the measured runtime rows: reads a reduced-probe ``decode_perf_report.csv``
(tt-perf-report, default construction path = selected_precision_config.json) and checks the dominant decode matmul
rows' weight dtype (``Input 1 Datatype``) and ``Math Fidelity`` against the selected config, plus the all-reduce
payload dtype (``Input 0 Datatype`` of AllReduceAsync).  Writes a JSON report next to the CSV.
Usage: python check_perf_rows.py <decode_perf_report.csv> <selected_precision_config.json> <out.json>"""
import json
import sys

import pandas as pd

csv, cfg_path, out = sys.argv[1:4]
cfg = json.load(open(cfg_path))
p, head = cfg["policy"], cfg["head"]
DT = {"bf16": "BFLOAT16", "bfp8": "BFLOAT8_B", "bfp4": "BFLOAT4_B"}
FI = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi4": "HiFi4"}
df = pd.read_csv(csv)
op = df["OP Code"].astype(str)

# probe layers 0 (conv_dense: conv fidelity exception hifi2), 2 (attn_moe), 3 (conv_moe)
expect = {
    "LM head 32 x 2048 x 32256": (op.str.contains("32 x 2048 x 32256"), head["dtype"], head["fidelity"]),
    "expert gate/up (sparse, decode)": (
        op.str.contains("SparseMatmul") & op.str.contains("x 2048 x 768"),
        p["expert_gate_up_decode"] or p["expert_gate_up"],
        p["expert_fidelity"],
    ),
    "expert down (sparse, decode)": (
        op.str.contains("SparseMatmul") & op.str.contains("x 384 x 2048"),
        p["expert_down_decode"] or p["expert_down"],
        p["expert_fidelity"],
    ),
    "attention qkv 32 x 2048 x 768": (
        op.str.contains("Matmul") & ~op.str.contains("Sparse") & op.str.contains("32 x 2048 x 768"),
        p["attn_qkv"],
        p["attn_fidelity"],
    ),
    "ShortConv in_proj 32 x 2048 x 1536 (layers 0 + 3)": (op.str.contains("32 x 2048 x 1536"), p["conv_in"], None),
    "dense w13 32 x 2048 x 6144": (op.str.contains("32 x 2048 x 6144"), p["dense_w13"], p["dense_fidelity"]),
}
report, ok = {}, True
for name, (mask, dt, fid) in expect.items():
    rows = df[mask]
    dts = sorted(set(rows["Input 1 Datatype"].astype(str)))
    fids = sorted({f.split()[0] for f in rows["Math Fidelity"].astype(str)})  # "LoFi BF16 x BFP4 => BF16" -> LoFi
    good = len(rows) > 0 and dts == [DT[dt]] and (fid is None or fids == [FI[fid]])
    if name.startswith("ShortConv"):  # layer 0 keeps HiFi2 (layer exception), layer 3 uses the policy fidelity
        exp_f = sorted(
            {FI[cfg["layer_exceptions"].get("0", {}).get("conv_fidelity", p["conv_fidelity"])], FI[p["conv_fidelity"]]}
        )
        good = good and fids == exp_f
        fid = "/".join(exp_f)
    report[name] = {
        "rows": int(len(rows)),
        "weight_dtype": dts,
        "math_fidelity": fids,
        "expected": [DT[dt], fid],
        "ok": bool(good),
    }
    ok &= good
ar = df[op.str.contains("AllReduceAsync")]
ar_dts = sorted(set(ar["Input 0 Datatype"].astype(str)))
report["decode all-reduce payload"] = {
    "rows": int(len(ar)),
    "payload_dtypes": ar_dts,
    "expected": DT[cfg["plan"]["ccl_dtype"]] + " (bucket-1 decode: every partial incl. the MoE compact sum)",
    "ok": ar_dts == [DT[cfg["plan"]["ccl_dtype"]]],
}
ok &= report["decode all-reduce payload"]["ok"]
report["all_ok"] = bool(ok)
json.dump(report, open(out, "w"), indent=1)
print(json.dumps(report, indent=1))
sys.exit(0 if ok else 1)
