#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages LQ_CKPT=ckpt/query_vq_k32.pt
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 new_eval_k32.ipynb > results/_log/fix_eval_k32.log 2>&1
touch results/_log/FIXEVAL_DONE
