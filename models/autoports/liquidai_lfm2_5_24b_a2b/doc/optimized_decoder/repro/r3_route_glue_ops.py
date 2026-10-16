# Routing-glue op list from a decoder Tracy ops csv (bench_policy perf_only, PERF_DECODE window):
# the ops after the ffn_norm LayerNorm (the last LayerNorm of each decode step) up to the first
# SparseMatmul, plus the post-expert weight ops (after the last SparseMatmul up to FastReduceNC),
# averaged over the decode iterations.  Usage: r3_route_glue_ops.py <ops.csv[.xz]> [label]
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
assert "DEVICE ID" in df.columns
sp = df.index[df["OP TYPE"] == "signpost"]
names = df.loc[sp, "OP CODE"].astype(str).tolist()
s0 = sp[names.index("PERF_DECODE")]
s1 = sp[names.index("PERF_DECODE_END")]
d = df.loc[s0 + 1 : s1 - 1].reset_index(drop=True)
ops = d["OP CODE"].astype(str).tolist()
iters, i = [], 0
while i < len(ops):
    try:
        sm = ops.index("SparseMatmulDeviceOperation", i)
    except ValueError:
        break
    ln = max(j for j in range(i, sm) if ops[j] == "LayerNormDeviceOperation")
    sm2 = ops.index("SparseMatmulDeviceOperation", sm + 1)
    fr = ops.index("FastReduceNCDeviceOperation", sm2)
    iters.append((list(range(ln + 1, sm)), list(range(sm2 + 1, fr + 1))))
    i = fr + 1


def show(tag, idx_lists):
    n = len(idx_lists[0])
    assert all(len(l) == n for l in idx_lists), [len(l) for l in idx_lists]
    tot_k = tot_g = 0
    for p in range(n):
        rows = [d.loc[l[p]] for l in idx_lists]
        k = sum(r["DEVICE KERNEL DURATION [ns]"] for r in rows) / len(rows) / 1e3
        g = sum(r["OP TO OP LATENCY [ns]"] for r in rows) / len(rows) / 1e3
        r0 = rows[0]
        tot_k += k
        tot_g += g
        print(
            f"  {tag} {p:2d} {r0['OP CODE'][:34]:34s} cores {int(r0['CORE COUNT']):3d} {k:6.2f} us gap {g:5.2f}  in0 {r0['INPUT_0_Y_PAD[LOGICAL]']}x{r0['INPUT_0_X_PAD[LOGICAL]']} {r0['INPUT_0_DATATYPE']} in1 {r0['INPUT_1_Y_PAD[LOGICAL]']}x{r0['INPUT_1_X_PAD[LOGICAL]']}"
        )
    print(f"  {tag} TOTAL {n} ops: device {tot_k:.1f} us + gaps {tot_g:.1f} us = {tot_k + tot_g:.1f} us")


label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
print(f"== {label}: {len(iters)} decode iterations")
show("pre ", [a for a, _ in iters])
show("post", [b for _, b in iters])
