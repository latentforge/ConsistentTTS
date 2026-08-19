#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
mkdir -p ckpt results/_log
.venv/bin/python new_lqtrain.py --gpu 0 --k 32 --steps 600 --lr 0.02 --n 400 --out ckpt/query_vq_k32.pt > results/_log/fix_train_k32.log 2>&1
touch results/_log/FIXTRAIN_DONE
