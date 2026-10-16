"""Print the attention part (up to the residual add before the ffn norm) of one decode iteration
from decode_perf_report.csv files: python r3_attn_ops.py <dir1> [<dir2> ...]"""
import sys

import pandas as pd


def attn_ops(d):
    df = pd.read_csv(f"{d}/decode_perf_report.csv").sort_values("ID")
    df = df.iloc[: len(df) // 5]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (r["OP Code"].split(" ")[0].replace("DeviceOperation", ""), float(r["Device Time"]), int(r["Cores"]))
        )
        if len(rows) > 3 and rows[-1][0] == "LayerNorm" and any(x[0] == "SdpaDecode" for x in rows):
            break  # ffn norm reached
    return rows[:-2], sum(float(x) for x in df["Device Time"])  # drop the ffn-norm I2S + LN


for d in sys.argv[1:]:
    rows, tot = attn_ops(d)
    print(f"== {d}  attention {sum(r[1] for r in rows):.1f} us in {len(rows)} ops; layer {tot:.1f} us")
    for n, t, c in rows:
        print(f"   {n:38s} {t:7.2f} {c:4d}")
