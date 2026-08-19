#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
.venv/bin/python new_curate.py --n_cand 1400 --keep 800 --out results/_summary/curated_idx.json > results/_log/curate.log 2>&1
touch results/_log/CURATE_DONE
