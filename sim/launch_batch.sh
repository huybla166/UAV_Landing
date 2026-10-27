#!/usr/bin/env bash
# Launch sim/run_batch.m on N parallel Octave workers, then merge the QC files
# (same as launch_batch.ps1, without needing a PowerShell execution-policy change).
#   bash sim/launch_batch.sh FIRST LAST WORKERS [OUTDIR]
#   bash sim/launch_batch.sh 1 20 8
#   bash sim/launch_batch.sh 1 3150 8          (full dataset)
FIRST=${1:-1}
LAST=${2:-20}
WORKERS=${3:-8}
OUTDIR=${4:-D:/USV-landing-prediction/data/sim/dataset_v1}
OCTAVE=${OCTAVE:-/d/tools/octave/mingw64/bin/octave-cli.exe}
SIM=D:/USV-landing-prediction/sim
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"
cp "$SIM/design_matrix.csv" "$OUTDIR/design_matrix.csv"

t0=$(date +%s)
pids=()
for k in $(seq 1 "$WORKERS"); do
    "$OCTAVE" --no-gui --quiet --eval \
        "cd('$SIM'); mss_setup; run_batch('$SIM/design_matrix.csv', '$OUTDIR', $FIRST, $LAST, $k, $WORKERS);" \
        > "$LOGDIR/worker$k.out.txt" 2> "$LOGDIR/worker$k.err.txt" &
    pids+=($!)
done
wait "${pids[@]}"
echo "batch ids $FIRST..$LAST on $WORKERS workers: $(( $(date +%s) - t0 )) s"
python "$SIM/batch_report.py" "$OUTDIR"
