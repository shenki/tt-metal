#!/bin/bash
# Tracy + tt-perf-report for one multichip perf case (test_multichip_perf_determinism_no_fallback).
# usage: tracy_mc.sh <outdir> <kind> <active> <bucket|None> <tag> [policy-override-json]
#   writes <outdir>/<kind>/a<active>_b<bucket>/<tag>/{ops.csv.xz,ops_source.txt,{prefill,decode}_perf_report.{txt,csv},...}
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
OUT=$1; KIND=$2; A=$3; BK=$4; TAG=$5; POL=${6:-"{}"}
rm -rf generated/profiler/reports
O=$OUT/$KIND/a${A}_b${BK}/$TAG; mkdir -p $O
LFM2_POLICY_OVERRIDE="$POL" timeout 2400 python -m tracy -r -p -v -m pytest -q \
  "models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_multichip_decoder.py::test_multichip_perf_determinism_no_fallback[blackhole-$A-$BK-$KIND-mesh_device0-device_params0]" > $O/tracy_run.log 2>&1
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
cp "$csv" $O/ops.csv
printf '%s\n%s\n%s\n' "$TAG active=$A bucket=$BK policy_override=$POL" "$csv" "$(git rev-parse --short HEAD) worktree $(git diff HEAD -- models/autoports/liquidai_lfm2_5_24b_a2b/tt | md5sum | cut -c1-12)" > $O/ops_source.txt
for ph in PREFILL DECODE; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary > $O/${p}_perf_report.txt 2>&1
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-merge-devices --no-summary --csv $O/${p}_perf_report_per_device.csv > /dev/null 2>&1
done
python - "$O" "$TAG" "$KIND" "$A" "$BK" <<'PY'
import pandas as pd, sys
O, tag, kind, a, bk = sys.argv[1:6]
r = {}
for ph, n in [("prefill", 1), ("decode", 5)]:
    df = pd.read_csv(f"{O}/{ph}_perf_report.csv")
    r[ph] = df["Device Time"].sum() / n
    try:
        dd = pd.read_csv(f"{O}/{ph}_perf_report_per_device.csv")
        col = "Device"
        per = dd.groupby(col)["Device Time"].sum() / n
        r[ph + "_dev"] = " ".join(f"{v:.1f}" for v in per.values)
    except Exception as e:
        r[ph + "_dev"] = f"? ({e})"
print(f"TRACY {kind:10s} a{a} b{bk:<4s} {tag:24s} prefill {r['prefill']:9.1f} us [{r['prefill_dev']}]  decode {r['decode']:8.1f} us [{r['decode_dev']}]")
PY
xz -T0 -f $O/ops.csv
