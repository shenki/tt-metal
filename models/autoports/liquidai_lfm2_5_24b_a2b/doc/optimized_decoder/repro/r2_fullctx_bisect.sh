#!/bin/bash
# Review P5 control: attn_moe 131069-token full-context test under policy overrides (snapshot package).
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal
source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
LOGD=models/autoports/liquidai_lfm2_5_24b_a2b/doc/optimized_decoder/logs
run() {  # name json
  echo "=== $1 $2" ; LFM2_POLICY_OVERRIDE="$2" timeout 1500 pytest -q -s \
    models/autoports/lfm2_s3snap/tests/test_optimized_decoder.py -k "full_context and attn_moe" > $LOGD/r2_fullctx_$1.log 2>&1
  echo "rc=$?"; grep -h "delta PCC" $LOGD/r2_fullctx_$1.log | sed 's/.*run_case:[0-9]* - //'
}
for spec in "$@"; do run "${spec%%=*}" "${spec#*=}"; done
