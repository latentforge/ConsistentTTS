#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages LQ_CKPT=ckpt/query_vq_k32.pt
# wait for round-1 training
for i in $(seq 1 400); do [ -e results/_log/HUNT_TRAIN1_DONE ] && break; sleep 15; done
echo "[hunt-eval] training done; evaluating $(ls ckpt/hunt/*.pt | wc -l) ckpts across GPU1,2"
CUDA_VISIBLE_DEVICES=1 HUNT_NSHARD=2 HUNT_SHARD=0 .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=9000 --ExecutePreprocessor.kernel_name=python3 new_hunt_eval_g1.ipynb > results/_log/hunt_eval_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 HUNT_NSHARD=2 HUNT_SHARD=1 .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=9000 --ExecutePreprocessor.kernel_name=python3 new_hunt_eval_g2.ipynb > results/_log/hunt_eval_g2.log 2>&1 &
wait
touch results/_log/HUNT_EVAL_DONE
echo "[hunt-eval] all done"
