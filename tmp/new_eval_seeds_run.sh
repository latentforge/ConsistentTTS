#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages LQ_CKPT=ckpt/query_vq_k32_s1.pt
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 new_eval_seeds.ipynb > results/_log/eval_seeds.log 2>&1
touch results/_log/EVAL_SEEDS_DONE
