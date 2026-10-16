"""Per-variant device op times between V_<name>/E_<name> signposts of the newest ops_perf_results csv."""
import glob
import os
import sys

import pandas as pd

csv = (
    sys.argv[1]
    if len(sys.argv) > 1
    else max(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
)
df = pd.read_csv(csv)
assert "DEVICE ID" in df.columns, csv
print("csv", csv)
cur = None
rows = []
for _, r in df.iterrows():
    if r["OP TYPE"] == "signpost":
        if str(r["OP CODE"]).startswith("V_"):
            cur, rows = r["OP CODE"][2:], []
        elif cur and str(r["OP CODE"]) == f"E_{cur}":
            tot = sum(t for _, t, _ in rows)
            print(f"== {cur}: {tot:.2f} us in {len(rows)} ops: " + ", ".join(f"{n}({c}) {t:.2f}" for n, t, c in rows))
            cur = None
        continue
    if cur:
        rows.append(
            (str(r["OP CODE"]).replace("DeviceOperation", ""), r["DEVICE KERNEL DURATION [ns]"] / 1000, r["CORE COUNT"])
        )
