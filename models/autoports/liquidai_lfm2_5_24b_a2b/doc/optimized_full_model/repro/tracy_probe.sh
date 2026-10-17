#!/bin/bash
# Tracy + tt-perf-report of the reduced full-model probe (tests/test_full_model_perf.py::test_probe_profile).
# usage: tracy_probe.sh <outdir>   -> <outdir>/{ops.csv.xz,ops_source.txt,{prefill,decode}_perf_report.{txt,csv},...}
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export PYTHONPATH=/home/ttuser/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PYTHONPATH
O=$1; mkdir -p $O
rm -rf generated/profiler/reports
timeout 2400 python -m tracy -r -p -v --op-support-count=20000 --dump-device-data-mid-run -m pytest -q --timeout=2400 \
  "models/autoports/liquidai_lfm2_5_24b_a2b/tests/test_full_model_perf.py::test_probe_profile" > $O/tracy_run.log 2>&1
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
cp "$csv" $O/ops.csv
printf '%s\n%s\n' "$csv" "$(git rev-parse --short HEAD) worktree $(git diff HEAD -- models/autoports/liquidai_lfm2_5_24b_a2b/tt | md5sum | cut -c1-12)" > $O/ops_source.txt
for ph in PREFILL DECODE DECODE_B8; do p=$(echo $ph | tr A-Z a-z)
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $O/${p}_perf_report.csv > $O/${p}_perf_report.console.log 2>&1
  tt-perf-report $O/ops.csv --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary > $O/${p}_perf_report.txt 2>&1
done
python - "$O" <<'PY'
import pandas as pd, sys
O = sys.argv[1]
for ph, n in [("prefill", 1), ("decode", 5), ("decode_b8", 5)]:
    df = pd.read_csv(f"{O}/{ph}_perf_report.csv")
    print(f"TRACY probe {ph}: {df['Device Time'].sum() / n:.1f} us device per {'prefill' if n == 1 else 'step'} ({len(df)} op rows)")
PY
xz -T0 -f $O/ops.csv
