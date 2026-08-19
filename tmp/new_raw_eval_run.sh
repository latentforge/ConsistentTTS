#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
for i in $(seq 1 200); do [ -e results/_log/RAW_TRAIN_DONE ] && break; sleep 15; done
export LQ_CKPT=ckpt/query_vq_k32raw_s1.pt
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 new_eval_raw.ipynb > results/_log/eval_raw.log 2>&1
touch results/_log/EVAL_RAW_DONE
