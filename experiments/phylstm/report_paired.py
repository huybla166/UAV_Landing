"""Paired per-split comparisons behind the advisor slides of 08/11/2026: % change of the normalised RMSE (the definition
of table_rmse_mae.py: per channel sqrt(mean e^2 over test runs, origins, leads <= H) / train std, mean over heave, roll,
pitch) of model A against model B on every split where both exist, seed 0.

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/report_paired.py

Writes results/sim/table_paired.md. Negative = A has the lower error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
import table_rmse_mae as T  # noqa: E402
from sim_common import FS, RESULTS_SIM, load, motion  # noqa: E402

SPLITS = [f"cv{k}" for k in range(6)] + ["ood_Hs", "ood_beta"]
PAIRS = [   # (A, B, what it shows)
    ("cummins_e60", "data_e60", "PINN Cummins vs the same network without physics (60 epochs, 100 units)"),
    ("cummins_e60", "phy_e60", "PINN Cummins vs PhyLSTM2 (60 epochs, 100 units)"),
    ("phy_e60", "data_e60", "PhyLSTM2 vs the same network without physics (60 epochs, 100 units)"),
    ("data_e60_u200", "data_e60", "no physics: 200 vs 100 LSTM units (screening, 07/11)"),
    ("cummins_e60_u200", "cummins_e60", "PINN Cummins: 200 vs 100 LSTM units (screening, 07/11)"),
    ("cummins_e60_u200", "data_e60_u200", "PINN Cummins vs no physics, both 200 units (screening, 07/11)"),
    ("data_s1", "data", "run-to-run noise: no physics, seed 1 vs seed 0 (30 epochs, 100 units, same GPU type)"),
    ("phy_s1", "phy", "run-to-run noise: PhyLSTM2, seed 1 vs seed 0 (30 epochs, 100 units, same GPU type)"),
]


def main():
    phy = T.load_phylstm()
    meta, X = load()
    Y = motion(X)
    sd = {s: Y[L.split_rows(meta, s)["train"]].std(axis=(0, 1), dtype=np.float64) for s in SPLITS}
    lines = ["# Paired comparisons per split (seed 0)", "",
             "% change of normalised RMSE, A vs B (negative = A better); source: experiments/phylstm/report_paired.py.", ""]
    for a, b, what in PAIRS:
        common = [s for s in SPLITS if f"{s}|{a}|mse" in phy and f"{s}|{b}|mse" in phy]
        lines += [f"## {what}", "", f"A = `{a}`, B = `{b}`, splits: {', '.join(common) or 'none'}", ""]
        if not common:
            continue
        lines += ["| Horizon | " + " | ".join(common) + " | A mean | B mean |", "|---|" + "---|" * (len(common) + 2)]
        for H in T.HORIZONS:
            n = int(H * FS)
            ra = {s: T.score(phy[f"{s}|{a}|mse"], phy[f"{s}|{a}|mae"], sd[s], n)[0] for s in common}
            rb = {s: T.score(phy[f"{s}|{b}|mse"], phy[f"{s}|{b}|mae"], sd[s], n)[0] for s in common}
            cells = [f"{100 * (ra[s] / rb[s] - 1):+.2f}" for s in common]
            lines.append(f"| {H} s | " + " | ".join(cells) +
                         f" | {np.mean(list(ra.values())):.4f} | {np.mean(list(rb.values())):.4f} |")
        lines.append("")
    path = os.path.join(RESULTS_SIM, "table_paired.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
