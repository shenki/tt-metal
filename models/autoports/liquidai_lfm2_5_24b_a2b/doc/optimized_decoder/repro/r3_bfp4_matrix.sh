#!/bin/bash
# Real-input PCC matrix (tests/test_r3_bfp4_real.py in the private package lfm2_r3_bfp4, chip 3).
# usage: r3_bfp4_matrix.sh <layers csv> <out jsonl> [tags csv] [extra-tags json]
cd /home/ttuser/liquid-LFM2.5-24B-A2B/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=3
R3_LAYERS=$1 R3_OUT=$2 R3_TAGS=$3 R3_EXTRA=${4:-"{}"} timeout 5400 pytest -q models/autoports/lfm2_r3_bfp4/tests/test_r3_bfp4_real.py
