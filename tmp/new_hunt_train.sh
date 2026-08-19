#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
# round 1: seeds 1,2 on GPU1,2 ; round 2: seeds 3,4
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --save_every 100 --n 400 --seed 1 --out ckpt/hunt/p_s1.pt > results/_log/hunt_p_s1.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --save_every 100 --n 400 --seed 2 --out ckpt/hunt/p_s2.pt > results/_log/hunt_p_s2.log 2>&1 &
wait
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --save_every 100 --n 400 --seed 3 --out ckpt/hunt/p_s3.pt > results/_log/hunt_p_s3.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --save_every 100 --n 400 --seed 4 --out ckpt/hunt/p_s4.pt > results/_log/hunt_p_s4.log 2>&1 &
wait
touch results/_log/HUNT_TRAIN1_DONE
