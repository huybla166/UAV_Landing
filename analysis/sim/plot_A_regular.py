"""Plot verification A (step 1.3): regular-wave steady-state response.

Reads analysis/sim/A_regular_wave*.csv written by sim/verify_A_regular.m and draws, per
wave heading, heave/roll/pitch RAO (amplitude per metre of wave amplitude) from
  - the nonlinear time-domain simulation (markers),
  - A1: the same constant-coefficient model in the frequency domain (solid),
  - A2: frequency-dependent Capytaine coefficients + the same viscous damping (dashed),
  - A3: Capytaine motion RAO, potential flow only (dotted).
Also prints the A1 pass/fail table.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "A_regular_wave*.csv")))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
a = df["a"].iloc[0]
w0 = df["w0_model"].iloc[0]
scale = {"heave": 1.0, "roll": 180 / np.pi, "pitch": 180 / np.pi, "yaw": 180 / np.pi}
unit = {"heave": "m/m", "roll": "deg/m", "pitch": "deg/m", "yaw": "deg/m"}

# ---------------------------------------------------------------- A1 table
df["ratio_const"] = df["sim_amp"] / df["const_amp"]
df["dphase_const_deg"] = np.degrees(np.angle(np.exp(1j * (df["sim_phase"] - df["const_phase"]))))
df["ratio_fd"] = df["sim_amp"] / df["fd_amp"]
main = df[df["dof"].isin(["heave", "roll", "pitch"])]
# only cases where the DOF is actually excited (amplitude > 2 % of its max at that heading)
mx = main.groupby(["beta_deg", "dof"])["const_amp"].transform("max")
excited = main[main["const_amp"] > 0.02 * mx]
print(f"files: {[os.path.basename(f) for f in files]}")
print(f"wave amplitude a = {a} m, constant model built at w0 = {w0:.3f} rad/s, {len(excited)} excited cases")
print("\nA1: simulation vs constant-coefficient frequency response (same M, D, G)")
print(excited.groupby("dof").agg(
    max_abs_amp_err_pct=("ratio_const", lambda r: 100 * np.max(np.abs(r - 1))),
    max_abs_phase_err_deg=("dphase_const_deg", lambda p: np.max(np.abs(p))),
    max_fit_rms_rel=("fit_rms_rel", "max")).round(3).to_string())
print("\nA2: constant model vs frequency-dependent coefficients, |H_const/H_fd - 1| in %")
excited = excited.assign(cf=100 * np.abs(excited["const_amp"] / excited["fd_amp"] - 1))
print(excited.pivot_table(index="w", columns=["dof"], values="cf", aggfunc="max").round(1).to_string())

# ---------------------------------------------------------------- figure
betas = sorted(df["beta_deg"].unique(), reverse=True)
dofs = ["heave", "roll", "pitch"]
fig, axes = plt.subplots(len(betas), 3, figsize=(13, 3.2 * len(betas)), squeeze=False)
for r, b in enumerate(betas):
    for c, d in enumerate(dofs):
        ax = axes[r, c]
        s = df[(df["beta_deg"] == b) & (df["dof"] == d)].sort_values("w")
        k = scale[d] / a
        ax.plot(s["w"], s["const_amp"] * k, "-", color="C0", lw=2, label="A1 constant M, D, G")
        ax.plot(s["w"], s["fd_amp"] * k, "--", color="C1", lw=1.5, label="A2 A(w), B(w) + Bv")
        ax.plot(s["w"], s["capy_amp"] * k, ":", color="C2", lw=1.5, label="A3 Capytaine (no Bv)")
        ax.plot(s["w"], s["sim_amp"] * k, "o", color="k", ms=4, label="time-domain sim")
        ax.axvline(w0, color="grey", lw=0.8, ls="-.")
        ax.set_title(f"{d}, beta = {b:g} deg", fontsize=10)
        ax.set_xlabel("w [rad/s]")
        ax.set_ylabel(f"RAO [{unit[d]}]")
        ax.grid(alpha=0.3)
        ymax = max(s["const_amp"].max(), s["fd_amp"].max(), s["sim_amp"].max()) * k
        ax.set_ylim(0, 1.15 * ymax if ymax > 0 else 1)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Verification A: regular waves, otter_55kg, a = {a} m (grey line: w0 = {w0:.2f} rad/s used for M, D)")
fig.tight_layout()
out = os.path.join(HERE, "A_regular_wave.png")
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
