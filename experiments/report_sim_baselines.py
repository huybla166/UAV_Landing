"""Tables and figure for the simulated-dataset baselines (results/sim/baselines_metrics.npz).

  python experiments/report_sim_baselines.py
Writes results/sim/baselines.md and results/sim/figures/baselines_horizon.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_common import RESULTS_SIM, SIM   # noqa: E402  (DATASET env var selects dataset_v1 / dataset_v2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = RESULTS_SIM
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
FS, HORIZON = 5.0, 100
REPORT_STEPS = {"1 s": 5, "2 s": 10, "5 s": 25, "10 s": 50, "20 s": 100}
LABELS = {"mean": "Trung bình 60 s", "persistence": "Persistence", "ar5": "AR(5) – như bài",
          "ar_aic": "AR(p) theo AIC", "lstm_paper": "LSTM của bài (Wang 2026)"}
SPLITS = {"test_id": "Trong phân phối (rep 3)", "test_ood_Hs": "Ngoài phân phối: Hs = 0.40 m",
          "test_ood_beta": "Ngoài phân phối: β = 45°"}
CH = ["Heave", "Roll", "Pitch"]

d = np.load(os.path.join(RES, "baselines_metrics.npz"))
models = [m for m in LABELS if f"{m}|test_id" in d]
meta = pd.read_csv(os.path.join(SIM, "cache_5Hz", "meta.csv")).set_index("id")


def useful(curve):
    bad = np.nonzero(curve >= 0.9)[0]
    return f"{bad[0] / FS:.1f} s" if len(bad) else "> 20 s"


L = ["# Baseline trên dữ liệu mô phỏng dataset_v1 (Giai đoạn 3.1)", "",
     "Giao thức: `experiments/sim_common.py` (5 Hz, gốc dự đoán mỗi 1 s, ngữ cảnh ≤ 60 s, tầm 20 s, bỏ 30 s đầu). "
     "Huấn luyện trên tập `train` (1440 lần chạy), chấm điểm trên mọi gốc dự đoán của ba tập kiểm tra. "
     "nRMSE = RMSE / độ lệch chuẩn của chính lần chạy đó, rồi lấy trung bình theo lần chạy "
     "(1.0 ≈ không tốt hơn đoán giá trị trung bình). Script: `experiments/run_sim_baselines.py`.", ""]
if "ar_aic|orders" in d:
    L += [f"Bậc AR chọn theo AIC (heave, roll, pitch): {', '.join(str(int(v)) for v in d['ar_aic|orders'])}.", ""]
for s, sname in SPLITS.items():
    n_runs = d[f"{models[0]}|{s}"].shape[0]
    L += [f"## {sname} — {n_runs} lần chạy", ""]
    for c, ch in enumerate(CH):
        L += [f"**{ch}**", "", "| Mô hình | " + " | ".join(REPORT_STEPS) + " | TB 0–20 s | Tầm hữu ích (nRMSE < 0.9) |",
              "|---|" + "---|" * (len(REPORT_STEPS) + 2)]
        for m in models:
            cur = d[f"{m}|{s}"][:, :, c].mean(0)
            L.append(f"| {LABELS[m]} | " + " | ".join(f"{cur[k - 1]:.3f}" for k in REPORT_STEPS.values())
                     + f" | {cur.mean():.3f} | {useful(cur)} |")
        L.append("")

# breakdown by speed and wave direction on test_id
L += ["## Theo điều kiện (tập trong phân phối): nRMSE trung bình 0–20 s, trung bình 3 kênh", ""]
ids = d["meta_test_id_ids"]
m_id = meta.loc[ids]
for m in [x for x in ("ar_aic", "lstm_paper") if x in models]:
    v = d[f"{m}|test_id"].mean(axis=(1, 2))
    t = pd.DataFrame({"U": m_id.U.values, "beta": m_id.beta_deg.values, "v": v}).pivot_table(
        index="beta", columns="U", values="v", aggfunc="mean")
    L += [f"**{LABELS[m]}** (hàng: β, cột: U m/s)", "", "| β \\ U | " + " | ".join(f"{u:g}" for u in t.columns) + " |",
          "|---|" + "---|" * len(t.columns)]
    for b, row in t.iterrows():
        L.append(f"| {b:g}° | " + " | ".join(f"{x:.3f}" for x in row.values) + " |")
    L.append("")
with open(os.path.join(RES, "baselines.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))

# figure: nRMSE vs lead time
fig, ax = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey="row")
lead = np.arange(1, HORIZON + 1) / FS
for j, (s, sname) in enumerate(SPLITS.items()):
    for c, ch in enumerate(CH):
        a = ax[c, j]
        for k, m in enumerate(models):
            a.plot(lead, d[f"{m}|{s}"][:, :, c].mean(0), color=f"C{k}", lw=1.6, label=LABELS[m])
        a.axhline(0.9, color="0.6", lw=0.8, ls="--")
        a.set_ylim(0, 1.6)
        if c == 0:
            a.set_title(sname, fontsize=10)
        if j == 0:
            a.set_ylabel(f"{ch}: nRMSE")
        if c == 2:
            a.set_xlabel("Tầm dự đoán (s)")
ax[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle(f"Baseline trên dữ liệu mô phỏng {os.path.basename(SIM)} (Otter 55/80 kg, 3150 lần chạy)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "baselines_horizon.png"), dpi=130)
print("wrote", os.path.join(FIG, "baselines_horizon.png"))
