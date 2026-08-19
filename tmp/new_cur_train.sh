#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
for i in $(seq 1 200); do [ -e results/_log/CURATE_DONE ] && break; sleep 15; done
echo "[cur-train] curation done; training best recipe on curated high-UTMOS data"
# best recipe (proj+cosine) on curated data, N=600, intermediate saves, 3 seeds on GPU1,2
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_idx.json --n 600 --save_every 100 --seed 1 --out ckpt/hunt_cur/c_s1.pt > results/_log/cur_s1.log 2>&1 &
.venv/bin/python new_lqtrain.py --gpu 2 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_idx.json --n 600 --save_every 100 --seed 2 --out ckpt/hunt_cur/c_s2.pt > results/_log/cur_s2.log 2>&1 &
wait
.venv/bin/python new_lqtrain.py --gpu 1 --k 32 --steps 600 --lr 0.02 --sched cosine --vqspace proj --curate_file results/_summary/curated_idx.json --n 600 --save_every 100 --seed 3 --out ckpt/hunt_cur/c_s3.pt > results/_log/cur_s3.log 2>&1 &
wait
touch results/_log/CUR_TRAIN_DONE
