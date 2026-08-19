#!/usr/bin/env bash
set -u
cd /home/work/voice_research/speakerinc
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/work/voice_research/speakerinc/.venv/lib/python3.12/site-packages
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 new_run_fixedanchor.ipynb > results/_log/fixedanchor_run.log 2>&1
touch results/_log/FIXEDANCHOR_DONE
