#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
CUDA_VISIBLE_DEVICES=1 .venv/bin/python new_curate_wer.py --n_cand 1200 --keep 600 --out results/_summary/curated_wer_idx.json > results/_log/curate_wer.log 2>&1
touch results/_log/CURATE_WER_DONE
mkdir -p ckpt/hunt_werc
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_wer_idx.json --n 500 --save_every 100 --seed 1 --out ckpt/hunt_werc/w_s1.pt > results/_log/werc_s1.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_wer_idx.json --n 500 --save_every 100 --seed 2 --out ckpt/hunt_werc/w_s2.pt > results/_log/werc_s2.log 2>&1 &
wait
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_wer_idx.json --n 500 --save_every 100 --seed 3 --out ckpt/hunt_werc/w_s3.pt > results/_log/werc_s3.log 2>&1 &
wait
touch results/_log/WERC_TRAIN_DONE
