"""Score every model saved in results/preds_*.npz on identical targets; write tables and figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CHANNELS, FS, HORIZON, REPORT_STEPS, RESULTS, RUNS, load_all, load_preds, origins, targets

LABELS = {
    "mean": "Trung bình (không làm gì)",
    "persistence": "Persistence",
    "ar5": "AR(5) – như bài",
    "ar_aic": "AR(p) chọn theo AIC",
    "lstm_paper": "LSTM của bài (10 bước)",
    "lstm_paper_in": "LSTM của bài + chuẩn hóa cửa sổ",
    "tirex_ctx150": "TiRex zero-shot (30 s)",
    "tirex_ctx300": "TiRex zero-shot (60 s)",
    "tirex_sp": "TiRex zero-shot, 1 lượt (mốc cho bản tinh chỉnh)",
    "tirex_ft": "TiRex tinh chỉnh đầu ra",
    "k2vae": "K²VAE (ICML 2025), cấu hình gốc",
    "k2vae_small": "K²VAE thu nhỏ",
    "k2vae_small_short": "K²VAE thu nhỏ, dừng sớm theo 5 s đầu",
    "ridge_direct": "Tuyến tính trực tiếp nhiều tầm (kiểu DLinear)",
    "opt_ft": "TiRex tinh chỉnh (tái lập)",
    "opt_ft_fredf": "TiRex + FreDF",
    "opt_mv": "TiRex + bộ điều hợp đa kênh",
    "opt_mv_fredf": "TiRex + đa kênh + FreDF",
    "final": "MÔ HÌNH TỐI ƯU (đề xuất cuối)",
}
ORDER = [m for m in LABELS if os.path.exists(os.path.join(RESULTS, f"preds_{m}.npz"))]
CH_VI = {"heave_m": "Heave", "roll_deg": "Roll", "pitch_deg": "Pitch"}

data = load_all()
truth = {r: targets(x) for r, x in data.items()}
scale = {r: x.std(0) for r, x in data.items()}  # per-run, per-channel std for normalisation

nrmse = {}     # model -> (HORIZON, 3), averaged over runs
per_run = {}   # model -> {run: (HORIZON, 3)}
coverage = {}  # model -> (HORIZON, 3) empirical coverage of the 10-90 % band
width = {}
for m in ORDER:
    preds, q, levels = load_preds(m)
    per_run[m] = {r: np.sqrt(np.mean((preds[r] - truth[r]) ** 2, 0)) / scale[r] for r in RUNS}
    nrmse[m] = np.mean([per_run[m][r] for r in RUNS], 0)
    if q is not None:
        lo, hi = q[RUNS[0]].shape[-1] and 0, len(levels) - 1
        cov = [np.mean((truth[r] >= q[r][..., lo]) & (truth[r] <= q[r][..., hi]), 0) for r in RUNS]
        wid = [np.mean(q[r][..., hi] - q[r][..., lo], 0) / scale[r] for r in RUNS]
        coverage[m], width[m] = np.mean(cov, 0), np.mean(wid, 0)

# ---------- tables ----------
lines = ["# Kết quả dự đoán trên dữ liệu NTUA (leave-one-run-out, 4 lần chạy)", "",
         "nRMSE = RMSE / độ lệch chuẩn của chuỗi thật (1.0 ≈ không tốt hơn đoán giá trị trung bình). Càng nhỏ càng tốt.", ""]
for c, ch in enumerate(CHANNELS):
    lines += [f"## {CH_VI[ch]}", "", "| Mô hình | " + " | ".join(REPORT_STEPS) + " | TB 0–20 s |",
              "|---|" + "---|" * (len(REPORT_STEPS) + 1)]
    for m in ORDER:
        vals = [nrmse[m][s - 1, c] for s in REPORT_STEPS.values()] + [nrmse[m][:, c].mean()]
        lines.append(f"| {LABELS[m]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines.append("")
if coverage:
    lines += ["## Độ phủ khoảng 10–90 % của TiRex (lý tưởng = 0.80), chưa hiệu chỉnh conformal", "",
              "| Mô hình | Kênh | " + " | ".join(REPORT_STEPS) + " |", "|---|---|" + "---|" * len(REPORT_STEPS)]
    for m in coverage:
        for c, ch in enumerate(CHANNELS):
            lines.append(f"| {LABELS[m]} | {CH_VI[ch]} | " +
                         " | ".join(f"{coverage[m][s - 1, c]:.2f}" for s in REPORT_STEPS.values()) + " |")
    lines.append("")
lines += ["## Tầm dự đoán hữu ích: thời điểm đầu tiên nRMSE ≥ 0.9 (mô hình không còn tốt hơn đoán trung bình 10 %)", "",
          "| Mô hình | Heave | Roll | Pitch |", "|---|---|---|---|"]
for m in ORDER:
    cells = []
    for c in range(3):
        bad = np.nonzero(nrmse[m][:, c] >= 0.9)[0]
        cells.append(f"{bad[0] / FS:.1f} s" if len(bad) else "> 20 s")
    lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
lines.append("")
lines += ["## nRMSE trung bình 0–20 s theo từng lần chạy (trung bình 3 kênh)", "",
          "| Mô hình | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
for m in ORDER:
    lines.append(f"| {LABELS[m]} | " + " | ".join(f"{per_run[m][r].mean():.3f}" for r in RUNS) + " |")
with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- figures ----------
os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
lead = np.arange(1, HORIZON + 1) / FS
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for c, ax in enumerate(axes):
    for m in ORDER:
        ax.plot(lead, nrmse[m][:, c], label=LABELS[m], lw=1.6 if m.startswith("tirex") else 1.1,
                ls="--" if m in ("mean", "persistence") else "-")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_title(CH_VI[CHANNELS[c]])
    ax.set_xlabel("Tầm dự đoán [s]")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("nRMSE")
axes[0].set_ylim(0, 1.6)
axes[-1].legend(fontsize=7.5)
fig.suptitle("Sai số theo tầm dự đoán – dữ liệu thật NTUA, leave-one-run-out")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "figures", "nrmse_vs_horizon.png"), dpi=130)

run = "noWing_headQ"
x = data[run]
o = origins(len(x))
k = len(o) // 3
t_hist = np.arange(-60, 0) / FS
t_fut = np.arange(HORIZON) / FS
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for c, ax in enumerate(axes):
    ax.plot(t_hist, x[o[k] - 60:o[k], c], color="k", lw=1, label="Quá khứ (đầu vào)")
    ax.plot(t_fut, truth[run][k, :, c], color="k", lw=1.6, label="Thật (tương lai)")
    for m, col in [("lstm_paper", "C3"), ("ar5", "C1")]:
        if m in ORDER:
            ax.plot(t_fut, load_preds(m)[0][run][k, :, c], color=col, lw=1.2, label=LABELS[m])
    for m, col in [("tirex_ctx300", "C0"), ("tirex_ctx150", "C0")]:
        if m in ORDER:
            p, q, lv = load_preds(m)
            ax.plot(t_fut, p[run][k, :, c], color=col, lw=1.6, label=LABELS[m])
            ax.fill_between(t_fut, q[run][k, :, c, 0], q[run][k, :, c, -1], color=col, alpha=0.2,
                            label="TiRex 10–90 %")
            break
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_ylabel(CH_VI[CHANNELS[c]])
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7.5, ncol=3)
axes[-1].set_xlabel("Thời gian so với thời điểm dự đoán [s]")
fig.suptitle(f"Ví dụ một lần dự đoán 20 s – {run}")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "figures", "example_forecast.png"), dpi=130)
print("figures saved")
