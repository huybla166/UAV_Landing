"""One compact table, in the layout of Table 2 of the IFToMM ISRM 2026 ship-motion paper: models x horizons, MSE and MAE
on z-score normalised signals, averaged over the forecast horizon and over the three signals (heave, roll, pitch).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/table_mse_mae.py

For a horizon of H seconds the errors of leads 0.2 .. H s are averaged (every model forecasts 20 s; the column shows
its first H s). Normalisation: each signal divided by its standard deviation over the training runs of the fold
(z-score with training statistics). Values: mean over the 6 folds of evaluation protocol (a). LSTM [Wang 2026]: mean
of its 3 seeds; PhyLSTM variants: epoch with the lowest validation error. Writes results/sim/table_mse_mae.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
from sim_common import FS, K_FOLDS, RESULTS_SIM, load, motion  # noqa: E402

HORIZONS = [2, 5, 10, 20]
MODELS = [("lstm", "LSTM [Wang 2026]"), ("data", "PhyLSTM w/o physics"), ("phy", "PhyLSTM2")]


def main():
    lstm = dict(np.load(L.METRICS))
    phy = dict(np.load(os.path.join(RESULTS_SIM, "phylstm", "phylstm_metrics.npz")))
    meta, X = load()
    Y = motion(X)
    folds = [f"cv{k}" for k in range(K_FOLDS)]
    sd = {s: Y[L.split_rows(meta, s)["train"]].std(axis=(0, 1)) for s in folds}       # (3,) per fold
    out = {}
    for model, _ in MODELS:
        for H in HORIZONS:
            n = int(H * FS)
            mse, mae = [], []
            for s in folds:
                if model == "lstm":
                    m2 = np.mean([lstm[f"{s}|seed{k}|mse"] for k in L.SEEDS], axis=0)
                    m1 = np.mean([lstm[f"{s}|seed{k}|mae"] for k in L.SEEDS], axis=0)
                else:
                    m2, m1 = phy[f"{s}|{model}|mse"], phy[f"{s}|{model}|mae"]
                mse.append((m2[:, :n] / sd[s] ** 2).mean())
                mae.append((m1[:, :n] / sd[s]).mean())
            out[model, H] = (np.mean(mse), np.mean(mae))
    lines = ["Table. Comparison of MSE and MAE for different models and prediction horizons (dataset_v2, 6-fold CV, "
             "z-score normalised, mean over heave, roll and pitch)", "",
             "| Model | " + " | ".join(f"{H}s Horizon MSE | {H}s Horizon MAE" for H in HORIZONS) + " |",
             "|---|" + "---|" * (2 * len(HORIZONS))]
    for model, name in MODELS:
        lines.append(f"| {name} | " + " | ".join(f"{out[model, H][0]:.4f} | {out[model, H][1]:.4f}" for H in HORIZONS) + " |")
    path = os.path.join(RESULTS_SIM, "table_mse_mae.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
