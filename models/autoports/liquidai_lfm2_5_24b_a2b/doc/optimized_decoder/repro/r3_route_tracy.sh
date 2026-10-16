#!/bin/bash
# usage: r3_route_tracy.sh <python script> <out dir> [args...]  -- Tracy run of a signposted microbench,
# holding the shared Tracy lock for the whole run (report dir is shared), then per-signpost device sums.
[ -z "$FLOCKED" ] && exec env FLOCKED=1 flock /tmp/claude-1000/lfm2_tracy.lock "$0" "$@"
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1; S=$1; O=$2; shift 2; mkdir -p $O
rm -rf generated/profiler/reports
timeout 1800 python -m tracy -r -p -v $S "$@" > $O/run.log 2>&1 || { echo "TRACY RUN FAILED"; tail -5 $O/run.log; exit 1; }
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
cp "$csv" $O/ops.csv; echo "$csv" > $O/ops_source.txt
python /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal/models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/repro/r3_route_parse.py $O/ops.csv > $O/summary.txt; cat $O/summary.txt
xz -f $O/ops.csv
