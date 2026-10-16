#!/bin/bash
# usage: r2_bfp4_bench.sh <kind> <tag> <policy-json> [out-jsonl]
# Real-weight PCC (32 users, traced decode) + traced-replay decode us at b1/b32, run from the frozen
# stage-3 snapshot package (private copy models/autoports/lfm2_s3snap_r2bfp4 of lfm2_s3snap) on chip TT_VISIBLE_DEVICES=1.
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1
D=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/sweeps
KIND=$1; TAG=$2; POL=${3:-"{}"}; OUT=${4:-$D/r2_bfp4_bench.jsonl}
mkdir -p $D/r2_bfp4_logs
LFM2_POLICY="$POL" LFM2_TAG=$TAG LFM2_BENCH_BATCH=1,32 LFM2_BENCH_OUT=$OUT timeout 2400 \
  pytest -q -x ${LFM2_PKG:-models/autoports/lfm2_s3snap_r2bfp4}/tests/bench_policy.py -k $KIND > $D/r2_bfp4_logs/${KIND}_${TAG}.log 2>&1
rc=$?
echo "rc=$rc $KIND $TAG" | tee -a $D/r2_bfp4_logs/${KIND}_${TAG}.log
tail -n 3 $D/r2_bfp4_logs/${KIND}_${TAG}.log
