#!/usr/bin/env bash
# after dataset_v2 is generated: verification chain (A) and baselines (B) in parallel, then the residual check
cd /d/USV-landing-prediction
L=data/sim/dataset_v2/logs/launch_gpu_v2.txt
until grep -q "all groups done" $L 2>/dev/null; do sleep 60; done
export DATASET=dataset_v2 PYTHONIOENCODING=utf-8
mkdir -p analysis/sim/V_dataset_v2
( python analysis/sim/verify_dataset_v1.py integrity > analysis/sim/V_dataset_v2/log_integrity.txt 2>&1
  python analysis/sim/verify_dataset_v1.py regime    > analysis/sim/V_dataset_v2/log_regime.txt 2>&1
  python analysis/sim/add_regime.py                  > analysis/sim/V_dataset_v2/log_add_regime.txt 2>&1
  python analysis/sim/verify_dataset_v1.py linear    > analysis/sim/V_dataset_v2/log_linear.txt 2>&1
  python analysis/sim/verify_dataset_v1.py forces    > analysis/sim/V_dataset_v2/log_forces.txt 2>&1
  echo "verification chain done $(date +%H:%M)" >> analysis/sim/V_dataset_v2/log_integrity.txt ) &
( /d/tools/venv-gpu/Scripts/python.exe experiments/sim_common.py > results/sim/log_cache.txt 2>&1
  /d/tools/venv-gpu/Scripts/python.exe experiments/run_sim_baselines.py naive ar lstm > results/sim/log_baselines.txt 2>&1
  python experiments/report_sim_baselines.py > results/sim/log_report.txt 2>&1
  echo "baselines done $(date +%H:%M)" >> results/sim/log_baselines.txt ) &
wait
python "$1/check_residual_v2.py" --fresh > results/sim/residual_v2/log.txt 2>&1
echo "all done $(date +%H:%M)"
