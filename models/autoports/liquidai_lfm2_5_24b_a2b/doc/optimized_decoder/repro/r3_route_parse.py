# Per-signpost device-time sums of a Tracy ops csv: signposts "<name>" ... "<name>_END" around REPS replays.
import re
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
assert "DEVICE ID" in df.columns
detail = len(sys.argv) > 2
cur, rows, out = None, [], []
for _, r in df.iterrows():
    if r["OP TYPE"] == "signpost":
        name = str(r["OP CODE"])
        if name.endswith("_END"):
            if cur is not None:
                out.append((cur, rows))
            cur, rows = None, []
        else:
            cur, rows = name, []
        continue
    if cur is not None:
        rows.append(r)
for name, rs in out:
    m = re.search(r"#R(\d+)", name)
    reps = int(m.group(1)) if m else 1
    k = sum(r["DEVICE KERNEL DURATION [ns]"] for r in rs) / reps / 1e3
    g = sum(r["OP TO OP LATENCY [ns]"] for r in rs[1:]) / reps / 1e3 if len(rs) > 1 else 0.0
    print(f"{name:60s} ops/rep {len(rs)//reps:3d}  device {k:7.2f} us  gaps {g:6.2f} us  total {k+g:7.2f}")
    if "#D" in name:
        n = len(rs) // reps
        for i in range(n):
            grp = [rs[j * n + i] for j in range(reps)]
            d = sum(x["DEVICE KERNEL DURATION [ns]"] for x in grp) / reps / 1e3
            gg = sum(x["OP TO OP LATENCY [ns]"] for x in grp) / reps / 1e3
            r0 = grp[0]
            print(
                f"    {r0['OP CODE'][:34]:34s} cores {int(r0['CORE COUNT']):3d} {d:6.2f} us gap {gg:5.2f}  in0 {r0['INPUT_0_Y_PAD[LOGICAL]']}x{r0['INPUT_0_X_PAD[LOGICAL]']} {r0['INPUT_0_DATATYPE']} in1 {r0['INPUT_1_Y_PAD[LOGICAL]']}x{r0['INPUT_1_X_PAD[LOGICAL]']}"
            )
