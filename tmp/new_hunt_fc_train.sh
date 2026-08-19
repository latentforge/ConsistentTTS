#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --init fixed_content --vocab_filter english --save_every 50 --n 400 --seed 1 --out ckpt/hunt_fc/fc_s1.pt > results/_log/hunt_fc_s1.log 2>&1 &
# wait a bit so seed2 doesn't collide with the finishing GPU2 eval
sleep 180
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --init fixed_content --vocab_filter english --save_every 50 --n 400 --seed 2 --out ckpt/hunt_fc/fc_s2.pt > results/_log/hunt_fc_s2.log 2>&1 &
wait
touch results/_log/HUNT_FC_TRAIN_DONE
