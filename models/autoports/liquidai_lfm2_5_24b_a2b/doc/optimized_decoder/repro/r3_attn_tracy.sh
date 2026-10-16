#!/bin/bash
# usage: r3_attn_tracy.sh <outdir> <kind> <batch> <tag> [policy-json] [bucket 0/1]   (self-locks the shared tracy lock)
[ -z "$FLOCKED" ] && exec env FLOCKED=1 flock /tmp/claude-1000/lfm2_tracy.lock "$0" "$@"
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=2; OUT=$1; KIND=$2; B=$3; TAG=$4; POL=${5:-"{}"}; BUCKET=${6:-1}
LOG=/tmp/claude-1000/r3_attn_tpol.log
rm -rf generated/profiler/reports
LFM2_POLICY="$POL" LFM2_BENCH_BUCKET=$BUCKET LFM2_TAG=$TAG LFM2_BENCH_BATCH=$B LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_MODE=perf_only LFM2_BENCH_OUT=generated/r3_attn_tracy.jsonl timeout 1800 python -m tracy -r -p -v -m pytest -q models/autoports/lfm2_r3_attn/tests/bench_policy.py -k $KIND > $LOG 2>&1
grep -q " passed" $LOG || { echo "TRACY RUN FAILED: $(grep -m1 -E "Error|FAILED" $LOG)"; exit 1; }
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
head -1 "$csv" | grep -q "DEVICE ID" || { echo "csv without DEVICE ID: $csv"; exit 1; }
O=$OUT/$KIND/b$B/$TAG; mkdir -p $O; cp "$csv" $O/ops.csv
printf '%s\n%s\n%s\n' "$TAG bucket=$BUCKET policy=$POL batch=$B" "$csv" "$(md5sum models/autoports/lfm2_r3_attn/tt/optimized_decoder.py)" > $O/ops_source.txt
tt-perf-report $O/ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv $O/decode_perf_report.csv > $O/decode_perf_report.console.log 2>&1
python - "$O" "$TAG" "$KIND" "$B" <<'PY'
import pandas as pd, sys
O, tag, kind, b = sys.argv[1:5]
df = pd.read_csv(f"{O}/decode_perf_report.csv")
print(f"TRACY {kind} b{b} {tag}: decode {df['Device Time'].sum() / 5:.1f} us/iter, ops {len(df)/5:.0f}, divisible {all(c % 5 == 0 for c in df['OP Code'].value_counts())}")
PY
xz -f $O/ops.csv
