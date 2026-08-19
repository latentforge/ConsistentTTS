#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
.venv/bin/python new_vd_default.py --gpu 0 --tag vd_default --seed 42 > results/_log/vd_default_gen.log 2>&1
.venv/bin/python new_vd_eval.py vd_default > results/_log/vd_default_eval.log 2>&1
touch results/_log/VD_DEFAULT_DONE
