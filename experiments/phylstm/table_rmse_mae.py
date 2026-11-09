"""The compact results table: models x horizons, RMSE and MAE (layout of Table 2 of the IFToMM ISRM 2026 ship-motion
paper, with RMSE in place of MSE as the project reports).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/table_rmse_mae.py

For a horizon of H seconds the errors of leads 0.2 .. H s are used (every model forecasts 20 s; the column shows its
first H s). Per channel (heave, roll, pitch): RMSE = sqrt(mean e^2 over test runs, origins and leads <= H) and MAE =
mean |e| over the same, both divided by the channel's standard deviation over the training runs of the fold (z-score
with training statistics, accumulated in float64: in float32 numpy's std over ~6 M samples per channel came out
0.40-0.49 % low); then the mean over the three channels, over the seeds a model has, and over the 6 folds of
evaluation protocol (a). LSTM [Wang 2026]: last epoch (as the paper); PhyLSTM variants: epoch with the lowest
validation error (as the PhyLSTM paper). A model is listed only when every fold of every listed seed is present.
Writes results/sim/table_rmse_mae.md.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
from sim_common import FS, K_FOLDS, RESULTS_SIM, load, motion  # noqa: E402

HORIZONS = [2, 5, 10, 20]
FOLDS = [f"cv{k}" for k in range(K_FOLDS)]
PHY_DIR = os.path.join(RESULTS_SIM, "phylstm")
# (row label, [result tag per seed]); tags as written by phylstm_kaggle.py ('_s1' = seed 1, '_e60' = 60 epochs)
MODELS = [
    ("LSTM [Wang 2026]", "lstm"),
    ("PhyLSTM w/o physics (30 epochs)", ["data", "data_s1"]),
    ("PhyLSTM2 (30 epochs)", ["phy", "phy_s1"]),
    ("PhyLSTM w/o physics", ["data_e60", "data_s1_e60"]),
    ("PhyLSTM2", ["phy_e60", "phy_s1_e60"]),
    ("PINN Cummins", ["cummins_e60", "cummins_s1_e60"]),
    ("Sea-state aware", ["seastate_e60", "seastate_s1_e60"]),
    ("PINN Cummins + sea-state", ["cummins_seastate_e60", "cummins_seastate_s1_e60"]),
    ("PatchTST [Nie 2023]", ["patchtst_e60"]),            # tsf_baselines.py (Time-Series-Library models)
    ("iTransformer [Liu 2024]", ["itransformer_e60"]),
    ("TimeXer [Wang 2024]", ["timexer_e60"]),
]


def load_phylstm():
    """Every phylstm_metrics.npz (or part_*.npz) under results/sim/phylstm/, merged; a key twice is an error."""
    res = {}
    for d in sorted({os.path.dirname(f) for f in glob.glob(os.path.join(PHY_DIR, "**", "*.npz"), recursive=True)}):
        if any(os.sep + x in d for x in ("smoke", "rerun", "failed_cpu")):
            continue
        files = glob.glob(os.path.join(d, "phylstm_metrics.npz")) or glob.glob(os.path.join(d, "part_*.npz"))
        for f in files:
            for k, v in np.load(f).items():
                if k.endswith("|ids"):
                    res[k] = v
                elif k in res:
                    raise SystemExit(f"{k} in two result files ({f})")
                else:
                    res[k] = v
    return res


def score(m2, m1, sd, n):
    """per-run mean e^2, |e| per lead (R, 100, 3) -> (normalised RMSE, MAE) averaged over the 3 channels."""
    return ((np.sqrt(m2[:, :n].mean(axis=(0, 1), dtype=np.float64)) / sd).mean(),
            (m1[:, :n].mean(axis=(0, 1), dtype=np.float64) / sd).mean())


def main():
    lstm, phy = dict(np.load(L.METRICS)), load_phylstm()
    for s in FOLDS:
        assert np.array_equal(phy[f"{s}|ids"], lstm[f"{s}|ids"]), f"test runs differ on {s}"
    meta, X = load()
    Y = motion(X)
    sd = {s: Y[L.split_rows(meta, s)["train"]].std(axis=(0, 1), dtype=np.float64) for s in FOLDS}
    lines = ["Table. Comparison of RMSE and MAE for different models and prediction horizons (dataset_v2, 6-fold CV "
             "grouped by wave seed, z-score normalised, mean over heave, roll and pitch; past 60 s of motion only, no "
             "future waves)", "",
             "| Model | " + " | ".join(f"{H}s RMSE | {H}s MAE" for H in HORIZONS) + " |",
             "|---|" + "---|" * (2 * len(HORIZONS))]
    missing = []
    for name, tags in MODELS:
        if tags == "lstm":
            srcs = [(lstm, f"seed{k}") for k in L.SEEDS]
        else:
            srcs = [(phy, t) for t in tags if all(f"{s}|{t}|mse" in phy for s in FOLDS)]
            if not srcs:
                missing.append(name)
                continue
        cells = []
        for H in HORIZONS:
            n = int(H * FS)
            v = np.array([[score(r[f"{s}|{t}|mse"], r[f"{s}|{t}|mae"], sd[s], n) for r, t in srcs] for s in FOLDS])
            rmse, mae = v.mean(axis=(0, 1))
            cells += [f"{rmse:.4f}", f"{mae:.4f}"]
        seeds = "" if tags == "lstm" else f" ({len(srcs)} seed{'s' if len(srcs) > 1 else ''})"
        lines.append(f"| {name}{seeds if tags != 'lstm' else ' (3 seeds)'} | " + " | ".join(cells) + " |")
    if missing:
        lines += ["", "Not yet available: " + ", ".join(missing) + "."]
    path = os.path.join(RESULTS_SIM, "table_rmse_mae.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
