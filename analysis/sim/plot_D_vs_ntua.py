"""Verification D (step 1.6): simulated Otter 55 kg vs measured NTUA 10-12 m model.

Both are compared on the same band 0.2-2.5 Hz (the NTUA records are band-passed and
contain no energy below ~0.2 Hz; data/processed/README.md). For heave/roll/pitch:
std from the integrated Welch spectrum, peak frequency, mean frequency m1/m0, spectral
width nu = sqrt(m0 m2 / m1^2 - 1), and for roll a half-power-bandwidth damping estimate
zeta_hp = (w2 - w1) / (2 w_peak) (only meaningful for a resonance-dominated peak).
Sim: data/sim/D_ntua_like/otter55_<run>.csv (first 30 s dropped).
NTUA: data/processed/<run>_20Hz.csv (5 s trimmed at both ends).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = ["noWing_headQ", "noWing_followQ", "wing_headQ", "wing_followQ"]
FS = 20.0
BAND = (0.2, 2.5)  # Hz
CH = ["heave", "roll", "pitch"]
UNIT = {"heave": "m", "roll": "deg", "pitch": "deg"}


def load_ntua(run):
    x = np.loadtxt(os.path.join(ROOT, "data", "processed", f"{run}_20Hz.csv"), delimiter=",", skiprows=1)
    x = x[int(5 * FS):-int(5 * FS)]
    return {"heave": x[:, 1], "roll": x[:, 2], "pitch": x[:, 3]}


def load_sim(run):
    x = np.loadtxt(os.path.join(ROOT, "data", "sim", "D_ntua_like", f"otter55_{run}.csv"), delimiter=",", skiprows=1)
    x = x[x[:, 0] >= 30]
    return {"heave": x[:, 10], "roll": np.degrees(x[:, 11]), "pitch": np.degrees(x[:, 12])}


def stats(y):
    f, p = signal.welch(signal.detrend(y), fs=FS, nperseg=1024)
    m = (f >= BAND[0]) & (f <= BAND[1])
    f, p = f[m], p[m]
    w = 2 * np.pi * f
    m0 = np.trapezoid(p, f); m1 = np.trapezoid(w * p, f); m2 = np.trapezoid(w ** 2 * p, f)
    k = np.argmax(p)
    half = p[k] / 2
    lo = k
    while lo > 0 and p[lo] > half:
        lo -= 1
    hi = k
    while hi < len(p) - 1 and p[hi] > half:
        hi += 1
    zeta_hp = (w[hi] - w[lo]) / (2 * w[k])
    return dict(std=np.sqrt(m0), w_peak=w[k], w_mean=m1 / m0, width=np.sqrt(max(m0 * m2 / m1 ** 2 - 1, 0)),
                zeta_hp=zeta_hp), (w, p)


rows, spec = [], {}
for run in RUNS:
    for src, loader in (("NTUA", load_ntua), ("sim Otter 55 kg", load_sim)):
        d = loader(run)
        for c in CH:
            s, sp = stats(d[c])
            rows.append(dict(run=run, source=src, dof=c, **s))
            spec[(run, src, c)] = sp
df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("Band 0.2-2.5 Hz. w in rad/s. std in m (heave) or deg (roll, pitch).")
print(df.round(3).to_string(index=False))

piv = df.pivot_table(index=["dof", "run"], columns="source", values=["std", "w_peak", "width"]).round(3)
print("\n", piv.to_string())
r = df[df.dof == "roll"].pivot_table(index="run", columns="source", values="zeta_hp").round(3)
print("\nRoll half-power damping estimate zeta_hp:\n", r.to_string())
df.to_csv(os.path.join(ROOT, "analysis", "sim", "D_vs_ntua.csv"), index=False)

fig, axes = plt.subplots(len(RUNS), 3, figsize=(13, 2.8 * len(RUNS)), squeeze=False)
for i, run in enumerate(RUNS):
    for j, c in enumerate(CH):
        ax = axes[i, j]
        for src, col in (("NTUA", "k"), ("sim Otter 55 kg", "C0")):
            w, p = spec[(run, src, c)]
            ax.semilogy(w, p / (2 * np.pi), color=col, lw=1.4, label=src)
        ax.set_title(f"{run}: {c}", fontsize=9)
        ax.set_xlabel("w [rad/s]")
        ax.set_ylabel(f"PSD [{UNIT[c]}^2 s/rad]")
        ax.grid(alpha=0.3, which="both")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Verification D: motion spectra, NTUA 10-12 m model (measured) vs Otter 2 m (MSS-Capytaine, simulated, U = 1 m/s assumed)")
fig.tight_layout()
out = os.path.join(ROOT, "analysis", "sim", "D_vs_ntua.png")
fig.savefig(out, dpi=130)
print("saved", out)
