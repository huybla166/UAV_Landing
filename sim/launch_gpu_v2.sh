#!/usr/bin/env bash
# dataset_v2 on the GPU: waits for prep_gpu_inputs.m (3150 JSONs + 4 vessel files), then runs the two large
# groups together (2 processes fit in RAM), then the two viscous-variant groups.
O=D:/USV-landing-prediction/data/sim/dataset_v2
PY=/d/tools/venv-gpu/Scripts/python.exe
cd /d/USV-landing-prediction/sim
until [ "$(ls $O/runs/*_coef.json 2>/dev/null | wc -l)" -ge 3150 ] && [ "$(ls $O/gpu_inputs/vessel_*.mat 2>/dev/null | wc -l)" -ge 4 ] && [ "$(ls $O/gpu_inputs/waves_*.mat 2>/dev/null | wc -l)" -ge 20 ] && ! tasklist //FI "IMAGENAME eq octave-cli.exe" | grep -q octave; do sleep 30; done
echo "inputs ready $(date +%H:%M)"
for pair in "otter_55kg,0 otter_80kg,0" "otter_55kg,0.2 otter_80kg,0.2"; do
  pids=()
  for g in $pair; do
    t=$(echo $g | tr ',.' '__')
    $PY batch_sim_gpu.py --out $O --ids missing --group $g --tag gpu_$t --writers 3 --h 0.025 > $O/logs/gpu_$t.txt 2>&1 & pids+=($!)
    sleep 30
  done
  wait "${pids[@]}"
done
echo "gpu done $(date +%H:%M)"
python batch_report.py $O | head -12
