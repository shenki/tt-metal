#!/bin/bash
# Real-weight test matrix for the prefill candidate (private copy lfm2_s3snap_r2pf).
#   r2_prefill_tests.sh <name> '<policy override json>' '<pytest -k expr>'
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
LOGD=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/logs
LFM2_POLICY_OVERRIDE="$2" timeout 3600 pytest -q -s models/autoports/lfm2_s3snap_r2pf/tests/test_optimized_decoder.py -k "$3" > $LOGD/$1.log 2>&1
echo "rc=$? $1"; grep -h "delta PCC" $LOGD/$1.log | sed 's/.*run_case:[0-9]* - //; s/.*run_interleaved:[0-9]* - //' ; grep -E "passed|failed" $LOGD/$1.log | tail -1
