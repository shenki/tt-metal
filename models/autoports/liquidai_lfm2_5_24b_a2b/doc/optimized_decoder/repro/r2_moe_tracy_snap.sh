#!/bin/bash
# usage: tracy_pol.sh <outdir> <kind> <batch> <tag> [policy-json] [base]
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2
OUT=$1; KIND=$2; B=$3; TAG=$4; POL=${5:-"{}"}; BASE=${6:-default}
export TT_METAL_PROFILER_DIR=${TT_METAL_PROFILER_DIR:-$PWD/generated/profiler_r2moe}; rm -rf $TT_METAL_PROFILER_DIR/reports
LFM2_BENCH_BUCKET=${BUCKET:-0} LFM2_POLICY="$POL" LFM2_POLICY_BASE=$BASE LFM2_TAG=$TAG LFM2_BENCH_BATCH=$B LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_MODE=perf_only LFM2_BENCH_OUT=generated/tracy_pol.jsonl timeout 1800 python -m tracy -r -p -v -m pytest -q models/autoports/lfm2_s3snap/tests/bench_policy.py -k $KIND > ${TPOL_LOG:-/tmp/r2_moe_tpol.log} 2>&1
csv=$(ls -t $TT_METAL_PROFILER_DIR/reports/*/ops_perf_results_*.csv | head -1)
O=$OUT/$KIND/b$B/$TAG; mkdir -p $O; cp "$csv" $O/ops.csv; printf '%s\n%s\n%s\n' "$TAG base=$BASE policy=$POL batch=$B" "$csv" "$(git rev-parse --short HEAD)" > $O/ops_source.txt
for ph in PREFILL DECODE; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report $O/ops.csv ${ACTIVE:+--active-experts $ACTIVE} --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report $O/ops.csv ${ACTIVE:+--active-experts $ACTIVE} --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary > $O/${p}_perf_report.txt 2>&1
done
python - "$O" "$TAG" "$KIND" "$B" <<'PY'
import pandas as pd, sys
O, tag, kind, b = sys.argv[1:5]
r = {}
for ph, n in [("prefill", 1), ("decode", 5)]:
    df = pd.read_csv(f"{O}/{ph}_perf_report.csv")
    r[ph] = (df["Device Time"].sum() / n, all(c % n == 0 for c in df["OP Code"].value_counts()))
print(f"TRACY {kind:10s} b{b:<3s} {tag:30s} prefill {r['prefill'][0]:10.1f} us ({r['prefill'][1]})  decode {r['decode'][0]:8.1f} us ({r['decode'][1]})")
PY
