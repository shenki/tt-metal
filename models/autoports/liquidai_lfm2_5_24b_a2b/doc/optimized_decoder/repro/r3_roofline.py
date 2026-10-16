# Roofline vs device vs host end-to-end for the final optimized decoder (one decode step per layer kind).
#   python r3_roofline.py <tracy runs.txt> <bench jsonl>
# bytes/step = weight bytes actually read (dtypes taken from DEFAULT_POLICY) + KV bytes read by SDPA;
# roofline = bytes / DRAM bandwidth (Blackhole p150b, 8 GDDR6 banks, 512 GB/s nominal).
import json
import re
import sys

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.optimized_decoder import DEFAULT_POLICY as P

BW = 512e9
B8, B4, B16 = 1088 / 1024, 576 / 1024, 2.0  # bytes per element: BFP8, BFP4, BF16
NB = {"bfp8": B8, "bfp4": B4, "bf16": B16}  # dtype names as used by Policy
H, I_MOE, I_DENSE_PAD, E, K, NKV, HD = 2048, 1536, 12288, 64, 4, 8, 64
CTX = 2048 + 32  # bench decode position ~ PREFILL_LEN (tests/test_perf_determinism.py)

attn_proj = H * 3072 * NB[P.attn_qkv] + H * H * NB[P.attn_out]
conv_proj = H * 6144 * NB[P.conv_in] + H * H * NB[P.conv_out]
router = H * 256 * B16  # DRAM-sharded router, N padded 64 -> 256, bf16
# decode (stacked) expert copy; "" = same dtype as the prefill copy
expert = (
    H * 2 * I_MOE * NB[P.expert_gate_up_decode or P.expert_gate_up]
    + I_MOE * H * NB[getattr(P, "expert_down_decode", "") or P.expert_down]
)
dense = 2 * H * I_DENSE_PAD * NB[P.dense_w13] + I_DENSE_PAD * H * NB[P.dense_w2]
kv_per_user = CTX * 2 * NKV * HD * NB[P.kv_cache]


def experts_read(b):
    if b <= 4:  # compact path reads b*K expert slots (duplicates re-read)
        return b * K
    return E * (1 - (1 - K / E) ** b)  # expected union of selected experts over b users


def step_bytes(kind, b):
    if kind == "conv_dense":
        return conv_proj + dense
    moe = router + experts_read(b) * expert
    if kind == "conv_moe":
        return conv_proj + moe
    return attn_proj + moe + b * kv_per_user


def main(runs, bench):
    dev = {}
    for line in open(runs):
        m = re.match(r"TRACY (\w+)\s+b(\d+)\s+(\S+)\s+prefill\s+([\d.]+) us .*decode\s+([\d.]+) us", line)
        if m:
            dev[(m[1], int(m[2]), "_bucket" in m[3])] = float(m[5])
    host = {}
    for line in open(bench):
        r = json.loads(line)
        for k, v in r.items():
            if k.startswith("decode_us_b"):
                host[(r["kind"], int(k[len("decode_us_b") :]), bool(r.get("bucket")))] = v
    print(
        f"{'kind':10s} {'b':>3s} {'bucket':6s} {'MB/step':>8s} {'roofline us':>11s} {'device us':>9s} {'dev/roof':>8s} {'host e2e us':>11s} {'host-dev':>8s}"
    )
    for (kind, b, bk), d in sorted(dev.items()):
        by = step_bytes(kind, b)
        roof = by / BW * 1e6
        h = host.get((kind, b, bk))
        hs = f"{h:11.1f} {h - d:8.1f}" if h else f"{'-':>11s} {'-':>8s}"
        print(f"{kind:10s} {b:3d} {str(bk):6s} {by / 1e6:8.1f} {roof:11.1f} {d:9.1f} {d / roof:8.2f} {hs}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
