"""Verification B (step 1.4): linear limit and dynamic nonlinearity.

Reads analysis/sim/B/B_<case>_metrics.csv and B_<case>_spectra.csv written by
sim/verify_B_linear.m. Prints, per case and DOF, the share of variance explained by the
linear constant-coefficient ODE driven by the same logged wave forcing (R2_lin), and
plots B1 (linear limit) spectra: simulation (Welch) vs component-wise prediction.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "B")
m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "B_*_metrics.csv")))], ignore_index=True)
m["kappa"] = [f"{a:g}/{b:g}" for a, b in zip(m["kappa4"], m["kappa5"])]
pd.set_option("display.width", 160)
print("R2 = share of variance explained by a linear model driven by the same logged forcing")
print("L0 = zero-speed constant ODE (M, D, G); LU = linearization of hydroVessel about forward speed U")
for col in ("R2_L0", "R2_LU"):
    print(f"\n{col}:")
    print(m[m.dof != "yaw"].pivot_table(index=["case", "Hs", "U_ref", "control", "kappa"], columns="dof",
                                        values=col).round(4).to_string())
print("\nstd of the full model (heave m, angles rad):")
print(m.pivot_table(index="case", columns="dof", values="std_nl").round(4).to_string())
m["R2_lin"] = m["R2_LU"]
m.to_csv(os.path.join(HERE, "B_summary.csv"), index=False)

cases = [c for c in sorted(m["case"].unique()) if c.startswith("B1")]
fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.0 * len(cases)), squeeze=False)
for i, c in enumerate(cases):
    s = pd.read_csv(os.path.join(HERE, f"B_{c}_spectra.csv"))
    for j, d in enumerate(["heave", "roll", "pitch"]):
        ax = axes[i, j]
        sim = s[s.kind == "sim"]; pr = s[s.kind == "pred_coherent"]; pe = s[s.kind == "pred_expected"]
        ax.semilogy(sim["w"], sim[d], color="k", lw=1.2, label="simulation (Welch)")
        ax.semilogy(pe["w"], pe[d], color="C1", lw=1.2, ls="--", label="expected |H F|^2 S (incoherent)")
        ax.set_xlim(0, 10)
        ax.set_ylim(max(sim[d].max(), pe[d].max()) * 1e-4, max(sim[d].max(), pe[d].max()) * 3)
        r2 = m[(m.case == c) & (m.dof == d)]["R2_LU"].iloc[0]
        ax.set_title(f"{c}: {d}  (R2 linear = {r2:.4f})", fontsize=9)
        ax.set_xlabel("w [rad/s]"); ax.grid(alpha=0.3, which="both")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Verification B1: linear limit (U = 0, no control, kappa = 0, Hs = 0.05 m), Otter 55 kg")
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "B_linear_limit.png")
fig.savefig(out, dpi=130)
print("saved", out)

b2 = m[m.case.str.startswith("B2")]
if len(b2):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, d in zip(axes, ["heave", "roll", "pitch"]):
        for k, mk in (("0/0", "o-"), ("5/5", "s--")):
            s = b2[(b2.dof == d) & (b2.kappa == k)].sort_values("Hs")
            ax.plot(s["Hs"], np.clip(1 - s["R2_LU"], 1e-4, None), mk, label=f"kappa4/kappa5 = {k}")
        ax.set_title(d); ax.set_xlabel("Hs [m]"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("1 - R2 (variance not explained by the\nforward-speed linear model)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Verification B2: nonlinearity vs sea state (U = 1 m/s, autopilot, beta = 135 deg)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "B_nonlinearity.png")
    fig.savefig(out, dpi=130)
    print("saved", out)
