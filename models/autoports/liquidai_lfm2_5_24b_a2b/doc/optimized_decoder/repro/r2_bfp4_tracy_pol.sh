#!/bin/bash
# usage: r2_bfp4_tracy_pol.sh <kind> <batch> <tag> [policy-json]   (copy of session_state/scratchpad_tools/tracy_pol.sh
# pointed at the private snapshot copy lfm2_s3snap_r2bfp4; chip 1; serialized by the shared tracy flock)
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1
KIND=$1; B=$2; TAG=$3; POL=${4:-"{}"}
OUT=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/tracy_r2_bfp4
RUN=generated/r2_bfp4_tracy_$$; rm -rf $RUN
O=$OUT/$KIND/b$B/$TAG; mkdir -p $O
LFM2_POLICY="$POL" LFM2_TAG=$TAG LFM2_BENCH_BATCH=$B LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_MODE=perf_only LFM2_BENCH_OUT=$RUN/b.jsonl \
  timeout 1800 python -m tracy -r -p -v -o $RUN -m pytest -q models/autoports/lfm2_s3snap_r2bfp4/tests/bench_policy.py -k $KIND > $O/tracy_run.log 2>&1
echo "tracy rc=$?" >> $O/tracy_run.log
csv=$(ls -t $RUN/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
[ -z "$csv" ] && csv=$(find $RUN -name 'ops_perf_results_*.csv' | head -1)
printf '%s\n%s\n' "$TAG policy=$POL batch=$B" "$csv" > $O/ops_source.txt
for ph in PREFILL DECODE; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report "$csv" --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report "$csv" --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary > $O/${p}_perf_report.txt 2>&1
done
python - "$O" "$TAG" "$KIND" "$B" <<'PY'
import pandas as pd, sys
O, tag, kind, b = sys.argv[1:5]
r = {}
for ph, n in [("prefill", 1), ("decode", 5)]:
    df = pd.read_csv(f"{O}/{ph}_perf_report.csv")
    r[ph] = (df["Device Time"].sum() / n, all(c % n == 0 for c in df["OP Code"].value_counts()))
print(f"TRACY {kind:12s} b{b:<3s} {tag:28s} prefill {r['prefill'][0]:10.1f} us ({r['prefill'][1]})  decode {r['decode'][0]:8.1f} us ({r['decode'][1]})")
PY
rm -rf $RUN
