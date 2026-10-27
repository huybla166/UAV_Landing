"""One run of data/sim/dataset_v1 as the advisor sees it: wave at the vessel, wave load, and motions.

  python analysis/sim/plot_dataset_sample.py [run name]

Writes analysis/sim/dataset_v1_sample_run.png (60 s window, 300-360 s). Columns as written by
sim/run_one.m: D is heave (positive down), angles in rad, converted to degrees here.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "sim", "dataset_v1", "runs")
NAME = sys.argv[1] if len(sys.argv) > 1 else "otter_55kg_Hs0.30_Tp2.35_b135_U1_k0_r3"
INK, MUTED, BLUE, ORANGE, GRID = "#13293d", "#5a6672", "#2a78d6", "#eb6834", "#ddd8cc"

d = pd.read_csv(os.path.join(RUNS, NAME + ".csv"))
d = d[(d.t >= 300) & (d.t <= 360)]
deg = 180 / 3.141592653589793
rows = [
    ("mặt nước ζ tại tàu [m]", d.zeta, INK),
    ("mômen sóng pitch [N·m]", d.tau_w5, ORANGE),
    ("heave D [m]", d.D, BLUE),
    ("roll φ [độ]", d.phi * deg, BLUE),
    ("pitch θ [độ]", d.theta * deg, BLUE),
]
fig, axes = plt.subplots(len(rows), 1, figsize=(11, 8.5), sharex=True)
for ax, (label, y, c) in zip(axes, rows):
    ax.plot(d.t, y, color=c, lw=1.1)
    ax.set_ylabel(label, fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[-1].set_xlabel("thời gian [s]", color=INK)
fig.suptitle(f"data/sim/dataset_v1/runs/{NAME}.csv (đoạn 300–360 s)", fontsize=10, color=INK)
fig.tight_layout()
out = os.path.join(ROOT, "analysis", "sim", "dataset_v1_sample_run.png")
fig.savefig(out, dpi=130)
print(out)
