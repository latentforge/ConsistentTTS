#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
mkdir -p ckpt results/_log
.venv/bin/python new_lqtrain.py --gpu 0 --k 32 --steps 600 --lr 0.02 --n 400 --seed 1 --out ckpt/query_vq_k32_s1.pt > results/_log/ft_s1.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --n 400 --seed 2 --out ckpt/query_vq_k32_s2.pt > results/_log/ft_s2.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --n 400 --seed 3 --out ckpt/query_vq_k32_s3.pt > results/_log/ft_s3.log 2>&1 &
wait
touch results/_log/SEEDS_TRAIN_DONE
