"""Prediction figures in the style of Wang et al. 2026 (JFI 363:108381, Figs. 4-5: actual vs predicted motion and the
absolute error) for one test run, from the forecasts saved by predict_runs.py.

  python experiments/phylstm/plot_predictions.py results/sim/figures/pred_cv0_run2711.npz

(any Python with numpy and matplotlib; the GPU venv has no matplotlib). Writes next to the .npz:
  <stem>_data.png       the run: heave, roll, pitch over the whole record and a 60 s zoom
  <stem>_lead0.2s.png   forecast at lead 0.2 s (one step ahead, the setting of Wang 2026 Figs. 4-5) and absolute error
  <stem>_lead2s.png     the same at lead 2 s
  <stem>_leads.png      leads 0.2 / 1 / 2 / 5 s side by side
  <stem>_fan.png        the whole 20 s forecast from origins 25 s apart (the task as defined)
Lead tau = (j + 1) x 0.2 s is column j of the forecast made at origin o (last observed sample o - 1); it predicts
sample o + j. RMSE and MAE in the legends: over every origin of the run (one per sample), in m or degrees.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FS = 5.0
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # categorical slots 1-2 of the reference palette (validated)
CH = [("Heave D", "heave", "m"), ("Roll φ", "roll", "°"), ("Pitch θ", "pitch", "°")]
MODELS = [("pinn", "PINN Cummins", BLUE, "-"), ("timexer", "TimeXer", ORANGE, (0, (4, 2)))]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "regular", "legend.fontsize": 8,
                     "axes.edgecolor": MUTED, "savefig.dpi": 200, "figure.dpi": 100})


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def num(v, unit):
    return f"{v:.4f} m" if unit == "m" else f"{v:.3f}°"


def at_lead(D, name, j, c):
    """(time s, forecast, actual) of the forecasts at lead (j + 1) / FS over every origin."""
    k = D["origins"] + j
    return k / FS, D[name][:, j, c], D["eta"][k, c]


def stats(D, name, j, c):
    _, f, y = at_lead(D, name, j, c)
    e = f.astype(np.float64) - y
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))


def legend(ax, ncol=3):
    """Legend above the axes, right-aligned, so it never covers the curves (the panel title sits on the left)."""
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False, handlelength=2.4,
                    columnspacing=1.2, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)


def header(D):
    sea = json.loads(str(D["sea"]))
    return (f"Test run {int(D['run_id'])} ({str(D['split'])} test fold; {str(D['rank'])})\n"
            f"{sea['vessel']}, Hs {sea['Hs']:.2f} m, Tp {sea['Tp']:.2f} s, wave direction {sea['beta_deg']:g}°, "
            f"U {sea['U']:g} m/s; sampled at 5 Hz")


def fig_data(D, path, zoom=(200, 260)):
    t, eta = D["t"], D["eta"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for c, (name, _, unit) in enumerate(CH):
        for col, (lo, hi) in enumerate([(0, t[-1]), zoom]):
            ax = axes[c, col]
            m = (t >= lo) & (t <= hi)
            ax.plot(t[m], eta[m, c], color=INK, lw=0.7 if col == 0 else 1.2)
            ax.set_xlim(lo, hi)
            style(ax)
            if col == 0:
                ax.set_ylabel(f"{name} ({unit})")
                ax.axvspan(zoom[0], zoom[1], color=BLUE, alpha=0.08, lw=0)
            if c == 2:
                ax.set_xlabel("time (s)")
        axes[c, 1].sharey(axes[c, 0])
    axes[0, 0].set_title("whole record, 5 Hz (shaded: the zoom)", loc="left", color=INK)
    axes[0, 1].set_title(f"zoom {zoom[0]}-{zoom[1]} s", loc="left", color=INK)
    fig.suptitle(header(D), x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_lead(D, j, path, win=(200, 260)):
    tau = (j + 1) / FS
    fig, axes = plt.subplots(6, 1, figsize=(10, 13.5), sharex=True)
    for c, (name, short, unit) in enumerate(CH):
        ax, ax_e = axes[2 * c], axes[2 * c + 1]
        t, _, y = at_lead(D, "pinn", j, c)
        m = (t >= win[0]) & (t <= win[1])
        ax.plot(t[m], y[m], color=INK, lw=2.0, label="actual")
        for key, label, color, ls in MODELS:
            _, f, _ = at_lead(D, key, j, c)
            rmse, mae = stats(D, key, j, c)
            ax.plot(t[m], f[m], color=color, lw=1.1, ls=ls, label=f"{label} (RMSE = {num(rmse, unit)})")
            ax_e.plot(t[m], np.abs(f[m] - y[m]), color=color, lw=0.9, ls=ls, label=f"{label} error (MAE = {num(mae, unit)})")
        ax.set_ylabel(f"{name} ({unit})")
        ax_e.set_ylabel(f"|error| ({unit})")
        ax_e.set_ylim(bottom=0)
        ax.set_title(f"({'abcdef'[2 * c]}) {short.capitalize()}, {tau:g} s ahead", loc="left", color=INK)
        ax_e.set_title(f"({'abcdef'[2 * c + 1]}) Absolute error of {short}", loc="left", color=INK)
        for a in (ax, ax_e):
            style(a)
        legend(ax)
        legend(ax_e, ncol=2)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(*win)
    fig.suptitle(header(D) + f"\nForecast {tau:g} s ahead; {win[0]}-{win[1]} s shown, RMSE / MAE over all "
                 f"{len(D['origins'])} forecast origins of the run", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path)
    plt.close(fig)


def fig_leads(D, path, leads=(0.2, 1, 2, 5), win=(200, 230)):
    fig, axes = plt.subplots(len(leads), 3, figsize=(12, 2.3 * len(leads) + 0.9), sharex=True, sharey="col")
    for r, tau in enumerate(leads):
        j = int(round(tau * FS)) - 1
        for c, (name, short, unit) in enumerate(CH):
            ax = axes[r, c]
            t, _, y = at_lead(D, "pinn", j, c)
            m = (t >= win[0]) & (t <= win[1])
            ax.plot(t[m], y[m], color=INK, lw=1.8, label="actual")
            parts = []
            for key, label, color, ls in MODELS:
                _, f, _ = at_lead(D, key, j, c)
                ax.plot(t[m], f[m], color=color, lw=1.0, ls=ls, label=label)
                parts.append(f"{label.split()[0]} {num(stats(D, key, j, c)[0], unit)}")
            ax.set_title((f"{name} ({unit})\n" if r == 0 else "") + "RMSE " + " · ".join(parts), loc="left",
                         color=INK, fontsize=8.5)
            style(ax)
            if c == 0:
                ax.set_ylabel(f"{tau:g} s ahead")
            if r == len(leads) - 1:
                ax.set_xlabel("time (s)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    axes[0, 0].set_xlim(*win)
    fig.suptitle(header(D) + "\nThe same forecasts read at different leads: how far ahead the forecast still follows "
                 "the motion", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_fan(D, path, starts=(200, 225, 250, 275), span=(195, 300)):
    t, eta, o = D["t"], D["eta"], D["origins"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    for c, (name, short, unit) in enumerate(CH):
        ax = axes[c]
        m = (t >= span[0]) & (t <= span[1])
        ax.plot(t[m], eta[m, c], color=INK, lw=1.8, label="actual")
        for n, ts in enumerate(starts):
            i = int(np.nonzero(o == int(round(ts * FS)))[0][0])
            tt = (o[i] + np.arange(D["pinn"].shape[1])) / FS
            ax.axvline((o[i] - 1) / FS, color=MUTED, lw=0.8, ls=":")
            for key, label, color, ls in MODELS:
                ax.plot(tt, D[key][i, :, c], color=color, lw=1.2, ls=ls, label=f"{label}, 20 s forecast" if n == 0 else None)
        ax.set_ylabel(f"{name} ({unit})")
        style(ax)
        if c == 0:
            legend(ax)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(*span)
    fig.suptitle(header(D) + "\nEach coloured segment is one 20 s forecast, made at the dotted line from the past "
                 "60 s only", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    path = sys.argv[1]
    D = dict(np.load(path, allow_pickle=False))
    stem = os.path.splitext(path)[0]
    fig_data(D, stem + "_data.png")
    fig_lead(D, 0, stem + "_lead0.2s.png")
    fig_lead(D, 9, stem + "_lead2s.png")
    fig_leads(D, stem + "_leads.png")
    fig_fan(D, stem + "_fan.png")
    for j, tau in ((0, 0.2), (4, 1), (9, 2), (24, 5), (49, 10), (99, 20)):
        row = [f"{label} " + "/".join(num(stats(D, key, j, c)[0], unit) for c, (_, _, unit) in enumerate(CH))
               for key, label, _, _ in MODELS]
        print(f"lead {tau:>4g} s  RMSE heave/roll/pitch  " + "   ".join(row))
    print("written:", ", ".join(os.path.basename(stem) + s for s in
                                ("_data.png", "_lead0.2s.png", "_lead2s.png", "_leads.png", "_fan.png")))


if __name__ == "__main__":
    main()
