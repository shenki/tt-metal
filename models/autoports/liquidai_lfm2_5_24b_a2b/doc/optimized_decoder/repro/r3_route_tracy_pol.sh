#!/bin/bash
# usage: tracy_pol.sh <outdir> <kind> <batch> <tag> [policy-json] [base]
[ -z "$FLOCKED" ] && exec env FLOCKED=1 flock /tmp/claude-1000/lfm2_tracy.lock "$0" "$@"
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1; OUT=$1; KIND=$2; B=$3; TAG=$4; POL=${5:-"{}"}; BASE=${6:-default}
rm -rf generated/profiler/reports
LFM2_POLICY="$POL" LFM2_POLICY_BASE=$BASE LFM2_TAG=$TAG LFM2_BENCH_BATCH=$B LFM2_BENCH_DECODE_ITERS=5 LFM2_BENCH_MODE=perf_only LFM2_BENCH_OUT=generated/r3_route_tracy_pol.jsonl timeout 1800 python -m tracy -r -p -v -m pytest -q models/autoports/lfm2_r3_route/tests/bench_policy.py -k $KIND > /tmp/claude-1000/-home-ttuser-liquid-LFM2-5-24B-A2B/d07f798f-c504-4fdb-a218-1c7b9baf07e5/scratchpad/tpol_r3route.log 2>&1
grep -q " passed" /tmp/claude-1000/-home-ttuser-liquid-LFM2-5-24B-A2B/d07f798f-c504-4fdb-a218-1c7b9baf07e5/scratchpad/tpol_r3route.log || { echo "TRACY RUN FAILED: $(grep -m1 -E "Error|FAILED" /tmp/claude-1000/-home-ttuser-liquid-LFM2-5-24B-A2B/d07f798f-c504-4fdb-a218-1c7b9baf07e5/scratchpad/tpol_r3route.log)"; exit 1; }
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
O=$OUT/$KIND/b$B/$TAG; mkdir -p $O; cp "$csv" $O/ops.csv; printf '%s\n%s\n%s\n' "$TAG base=$BASE policy=$POL batch=$B" "$csv" "$(git rev-parse --short HEAD)" > $O/ops_source.txt
for ph in PREFILL DECODE; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary > $O/${p}_perf_report.txt 2>&1
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
xz -f $O/ops.csv
