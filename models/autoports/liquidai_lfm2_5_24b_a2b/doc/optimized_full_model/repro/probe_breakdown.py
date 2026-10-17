# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase breakdown of the probe token-out step from a tt-perf-report CSV (rows = op x replays, consecutive):
embedding/entry/cos-sin, decoder layers, final norm + LM head, position advance, sampling(+history) trace; device time
and op-to-op gaps.  Usage: python probe_breakdown.py <decode_perf_report.csv> [replays=5]"""
import sys

import pandas as pd

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
df = pd.read_csv(path)
df["gap"] = pd.to_numeric(df["Op-to-Op Gap"], errors="coerce").fillna(0)
g = [df.iloc[i : i + n] for i in range(0, len(df), n)]
seq = [(x["OP Code"].iloc[0], x["Device Time"].mean(), x["gap"].mean()) for x in g]
names = [s[0] for s in seq]
print(
    f"{path}: mean of {n} replays: device {sum(s[1] for s in seq):.1f} us, op-to-op gaps {sum(s[2] for s in seq):.1f} us, {len(seq)} ops"
)
hi = [i for i, s in enumerate(names) if "32256" in s][0]
fs = hi + 2
while fs < len(names) and names[fs].startswith("PlusOne"):
    fs += 1
first_layer = next(i for i, s in enumerate(names) if s.startswith("LayerNorm"))
phases = {
    "embedding + entry reshard + cos/sin (+ hoisted slices)": seq[:first_layer],
    "decoder layers": seq[first_layer : hi - 2],
    "final norm + LM head": seq[hi - 2 : hi + 2],
    "position advance": seq[hi + 2 : fs],
    "sampling trace": seq[fs:],
}
for k, v in phases.items():
    print(f"  {k}: device {sum(s[1] for s in v):.1f} us, gaps {sum(s[2] for s in v):.1f} us, {len(v)} ops")
    if k != "decoder layers":
        print("     " + ", ".join(f"{s[0].replace('DeviceOperation', '')} {s[1]:.1f}" for s in v))
