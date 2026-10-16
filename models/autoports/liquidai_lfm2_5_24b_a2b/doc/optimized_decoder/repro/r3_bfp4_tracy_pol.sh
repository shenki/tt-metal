#!/bin/bash
# usage: r3_bfp4_tracy_pol.sh <kind> <batch> <tag> [policy-json] [bucket 0|1]
# Copy of session_state/scratchpad_tools/tracy_pol_c0.sh pointed at the private package lfm2_r3_bfp4, chip 3.
# Holds the shared tracy lock for the whole run (generated/profiler/reports is shared) incl. csv pickup.
[ -z "$FLOCKED" ] && exec env FLOCKED=1 flock /tmp/claude-1000/lfm2_tracy.lock "$0" "$@"
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=3
KIND=$1; B=$2; TAG=$3; POL=${4:-"{}"}; BUCKET=${5:-0}
OUT=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/tracy_r3_bfp4
LOG=/home/ttuser/liquid-LFM2.5-24B-A2B/session_state/scratchpad_tools/tpol_r3_bfp4.log
rm -rf generated/profiler/reports
T0=$(date +%s)
LFM2_POLICY="$POL" LFM2_TAG=$TAG LFM2_BENCH_BATCH=$B LFM2_BENCH_BUCKET=$BUCKET LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_MODE=perf_only \
  LFM2_BENCH_OUT=generated/r3_bfp4_tracy_pol.jsonl timeout 1800 python -m tracy -r -p -v -m pytest -q models/autoports/lfm2_r3_bfp4/tests/bench_policy.py -k $KIND > $LOG 2>&1
grep -q " passed" $LOG || { echo "TRACY RUN FAILED $KIND b$B $TAG: $(grep -m1 -E "Error|FAILED" $LOG)"; exit 1; }
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
[ $(stat -c %Y "$csv") -ge $T0 ] || { echo "STALE CSV $csv"; exit 1; }
head -1 "$csv" | grep -q "DEVICE ID" || { echo "NO DEVICE ID column in $csv"; exit 1; }
O=$OUT/$KIND/b$B$([ "$BUCKET" = 1 ] && echo _bucket)/$TAG; mkdir -p $O
AE=""; [ "$BUCKET" = 1 ] && AE="--active-experts $((B * 4))"
printf '%s\n%s mtime=%s\n%s\n' "$TAG policy=$POL batch=$B bucket=$BUCKET" "$csv" "$(stat -c %y "$csv")" "$(cd models/autoports/lfm2_r3_bfp4 && cat tt/*.py tests/bench_policy.py tests/test_perf_determinism.py | sha256sum | cut -c1-16) (sha256 of lfm2_r3_bfp4 tt/*.py + bench files)" > $O/ops_source.txt
for ph in PREFILL DECODE; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report "$csv" $AE --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report "$csv" $AE --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary --no-color > $O/${p}_perf_report.txt 2>&1
done
python - "$O" "$TAG" "$KIND" "$B$([ "$BUCKET" = 1 ] && echo k)" <<'PY'
import pandas as pd, sys
O, tag, kind, b = sys.argv[1:5]
r = {}
for ph, n in [("prefill", 1), ("decode", 5)]:
    df = pd.read_csv(f"{O}/{ph}_perf_report.csv")
    r[ph] = (df["Device Time"].sum() / n, all(c % n == 0 for c in df["OP Code"].value_counts()))
line = f"TRACY {kind:10s} b{b:<4s} {tag:24s} prefill {r['prefill'][0]:10.1f} us ({r['prefill'][1]})  decode {r['decode'][0]:8.1f} us ({r['decode'][1]})"
print(line)
open(f"{O}/../../../summary.txt", "a").write(line + "\n")
PY
