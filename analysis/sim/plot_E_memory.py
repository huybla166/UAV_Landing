"""Plot verification E (sim/verify_E_memory.m): memory-model response in regular waves.

Top row: heave/roll/pitch amplitude per unit wave amplitude vs frequency, for the time-domain
simulation (dots), the BEM frequency-dependent transfer function H_fd (line) and the old MSS
constant model at w0 = 2 pi/1.9 (dashed). Bottom row: ratio sim / H_fd.
Prints the summary table used in the report.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ["sim", "fit", "fd", "const"]:
        df[c + "_rao"] = df[c + "_amp"] / df["a"]
    df["sim_fd"] = df.sim_amp / df.fd_amp
    df["sim_fit"] = df.sim_amp / df.fit_amp
    df["const_fd"] = df.const_amp / df.fd_amp
    df["dphase_fd_deg"] = np.degrees(np.angle(np.exp(1j * (df.sim_phase - df.fd_phase))))

    for vessel, dv in df.groupby("vessel"):
        fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
        for j, dof in enumerate(["heave", "roll", "pitch"]):
            for k, (beta, d) in enumerate(dv[dv.dof == dof].groupby("beta_deg")):
                col = f"C{k}"
                ax[0, j].plot(d.w, d.fd_rao, "-", color=col, lw=1.5, label=f"H_fd (BEM), β={beta:g}°")
                ax[0, j].plot(d.w, d.const_rao, "--", color=col, lw=1, label=f"MSS hằng số, β={beta:g}°")
                ax[0, j].plot(d.w, d.sim_rao, "o", color=col, ms=4, label=f"mô phỏng bộ nhớ, β={beta:g}°")
                ax[1, j].plot(d.w, d.sim_fd, "o-", color=col, ms=4, lw=1)
            ax[0, j].set_title(dof)
            ax[1, j].axhline(1, color="0.5", lw=0.8)
            ax[1, j].set_xlabel("ω (rad/s)")
        ax[0, 0].set_ylabel("biên độ / a  (m/m, rad/m)")
        ax[1, 0].set_ylabel("mô phỏng / H_fd")
        ax[0, 2].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Kiểm tra E — mô hình Cummins có bộ nhớ ({os.environ.get('E_TAG', 'v1_').rstrip('_')}), {vessel}, sóng đều, U = 0")
        fig.tight_layout()
        tag = os.environ.get("E_TAG", "")                      # e.g. E_TAG=v3_ for the v3 memory fit
        out = os.path.join(HERE, f"E_memory_{tag}{vessel}.png")
        fig.savefig(out, dpi=130)
        print("wrote", out)

    pd.set_option("display.width", 160)
    summ = df.groupby(["vessel", "dof"]).agg(
        sim_fd_min=("sim_fd", "min"), sim_fd_max=("sim_fd", "max"),
        sim_fit_min=("sim_fit", "min"), sim_fit_max=("sim_fit", "max"),
        const_fd_min=("const_fd", "min"), const_fd_max=("const_fd", "max"),
        max_abs_dphase_deg=("dphase_fd_deg", lambda x: np.abs(x).max()),
        max_fit_rms=("fit_rms_rel", "max"))
    print(summ.round(3).to_string())
    return df


if __name__ == "__main__":
    files = sys.argv[1:] or [os.path.join(HERE, f) for f in sorted(os.listdir(HERE))
                             if f.startswith("E_memory_") and f.endswith(".csv")]
    main(files)
