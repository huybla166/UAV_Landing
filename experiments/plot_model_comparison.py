"""Clean comparison figure of the key forecasters: nRMSE vs lead time and useful horizon."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FS, HORIZON, RESULTS, RUNS, load_all, load_preds, targets

KEY = [("lstm_paper", "LSTM của bài (Wang et al. 2026)", "C3", "-"),
       ("ar_aic", "AR(p) theo AIC", "C1", "-"),
       ("tirex_sp", "TiRex zero-shot (NeurIPS 2025)", "C0", "--"),
       ("tirex_ft", "TiRex tinh chỉnh (đề xuất)", "C0", "-"),
       ("k2vae", "K²VAE gốc (ICML 2025)", "C2", "--"),
       ("k2vae_small_short", "K²VAE thu nhỏ (dừng theo 5 s)", "C2", "-")]
KEY = [k for k in KEY if os.path.exists(os.path.join(RESULTS, f"preds_{k[0]}.npz"))]

data = load_all()
truth = {r: targets(x) for r, x in data.items()}
scale = {r: x.std(0) for r, x in data.items()}
lead = np.arange(1, HORIZON + 1) / FS
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for m, lab, col, ls in KEY:
    p = load_preds(m)[0]
    nr = np.mean([np.sqrt(np.mean((p[r] - truth[r]) ** 2, 0)) / scale[r] for r in RUNS], 0)
    for c, ax in enumerate(axes):
        ax.plot(lead[:50], nr[:50, c], ls, color=col, lw=1.8 if "đề xuất" in lab else 1.2, label=lab)
for c, ax in enumerate(axes):
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axhline(0.9, color="grey", lw=0.6, ls=":")
    ax.set_title(["Heave", "Roll", "Pitch"][c])
    ax.set_xlabel("Tầm dự đoán [s]")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("nRMSE (1 = đoán trung bình)")
axes[0].set_ylim(0, 1.4)
axes[-1].legend(fontsize=8)
fig.suptitle("So sánh mô hình trên dữ liệu thật NTUA (0–10 s, leave-one-run-out)")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "figures", "model_comparison.png"), dpi=130)
print("saved")
