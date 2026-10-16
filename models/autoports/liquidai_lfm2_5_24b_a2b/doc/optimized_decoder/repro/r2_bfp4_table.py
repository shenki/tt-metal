# Tabulate r2_bfp4 bench jsonl: min output/update PCC prefill/decode and traced decode us b1/b32.
import json
import sys

FIELDS = [
    "attn_qkv",
    "attn_out",
    "attn_fidelity",
    "conv_in",
    "conv_out",
    "conv_fidelity",
    "dense_w13",
    "dense_w2",
    "dense_fidelity",
]
print(f"| kind | tag | out P | out D | upd P | upd D | b1 us | b32 us | pass |")
print("|---|---|---|---|---|---|---|---|---|")
for l in open(sys.argv[1]):
    r = json.loads(l)
    ok = (
        r["min_prefill_pcc"] >= 0.995
        and r["min_decode_pcc"] >= 0.995
        and r["min_delta_prefill_pcc"] >= 0.99
        and r["min_delta_decode_pcc"] >= 0.99
    )
    print(
        f"| {r['kind']} | {r['tag']} | {r['min_prefill_pcc']:.5f} | {r['min_decode_pcc']:.5f} | {r['min_delta_prefill_pcc']:.5f} | {r['min_delta_decode_pcc']:.5f} | {r['decode_us_b1']:.1f} | {r['decode_us_b32']:.1f} | {'PASS' if ok else 'FAIL'} |"
    )
