"""Compare PhyLSTM2 (and its data-only Network 1) with the paper's LSTM baseline under evaluation protocol (a).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/report_phylstm.py

Reads results/sim/phylstm/phylstm_metrics.npz (downloaded from the Kaggle kernel, phylstm_kaggle.py) and
results/sim/lstm_cv_metrics.npz (run_sim_lstm_cv.py); writes results/sim/phylstm_cv.md. Same metric code for every
model (run_sim_lstm_cv.metrics): RMSE and MAE over every forecast origin and every lead 0-20 s, plus RMSE at single leads.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
from sim_common import CHANNELS, K_FOLDS, RESULTS_SIM, load, motion  # noqa: E402

PHY = os.path.join(RESULTS_SIM, "phylstm", "phylstm_metrics.npz")
OUT = os.path.join(RESULTS_SIM, "phylstm_cv.md")
MODELS = [("lstm", "LSTM của bài gốc (Wang 2026)"), ("data", "Mạng 1 của PhyLSTM, không vật lý"),
          ("phy", "PhyLSTM2"), ("data_last", "Mạng 1 không vật lý, epoch cuối"), ("phy_last", "PhyLSTM2, epoch cuối")]
LEADS = [1, 2, 5, 10, 20]


def per_split(res_lstm, res_phy, model, s):
    """(metrics dict, RMSE per lead) of one model on one split, or None if missing. LSTM: mean over its 3 seeds."""
    if model == "lstm":
        keys = [(f"{s}|seed{sd}|mse", f"{s}|seed{sd}|mae") for sd in L.SEEDS]
        if not all(k in res_lstm for pair in keys for k in pair):
            return None
        ms = [L.metrics(res_lstm[a], res_lstm[b]) for a, b in keys]
        return {k: np.mean([m[0][k] for m in ms], axis=0) for k in ("RMSE", "MAE")}, np.mean([m[1] for m in ms], axis=0)
    a, b = f"{s}|{model}|mse", f"{s}|{model}|mae"
    if a not in res_phy:
        return None
    return L.metrics(res_phy[a], res_phy[b])


def main():
    res_lstm = dict(np.load(L.METRICS))
    res_phy = dict(np.load(PHY))
    for s in L.SPLITS:                                    # the same test runs, in the same order
        if f"{s}|ids" in res_phy and f"{s}|ids" in res_lstm:
            assert np.array_equal(res_phy[f"{s}|ids"], res_lstm[f"{s}|ids"]), f"test runs differ on {s}"
    meta, X = load()
    Y = motion(X)
    sig = {s: Y[L.split_rows(meta, s)["test"]].std(axis=(0, 1)) for s in L.SPLITS}
    cv = [f"cv{k}" for k in range(K_FOLDS)]
    lines = ["# PhyLSTM2 so với LSTM của bài gốc — dataset_v2, giao thức (a)", "",
             "Cùng 6 fold theo seed sóng + 2 bài thử ngoài phân phối, cùng gốc dự đoán (mỗi 1 s, đoán 20 s), cùng cách tính "
             "RMSE và MAE (`run_sim_lstm_cv.metrics`). Không mô hình nào được biết sóng tương lai khi dự đoán; PhyLSTM2 chỉ "
             "dùng lực sóng thật trong hàm phạt vật lý lúc huấn luyện. LSTM của bài: trung bình 3 seed; PhyLSTM: 1 seed. "
             "Dòng \"6 fold\": trung bình ± độ lệch chuẩn giữa các fold. σ = độ lệch chuẩn chuyển động của các lần chạy "
             "kiểm tra (đoán bằng trung bình cho RMSE ≈ σ). Đơn vị: heave m, roll và pitch độ. Số liệu: "
             "`results/sim/phylstm/phylstm_metrics.npz`, `results/sim/lstm_cv_metrics.npz`; script "
             "`experiments/phylstm/report_phylstm.py`.", "",
             "Hai điểm cần nhớ khi đọc: (1) PhyLSTM (cả hai biến thể) giữ epoch có sai số kiểm định thấp nhất như bài báo, còn "
             "LSTM của bài gốc lấy epoch cuối; hàng \"epoch cuối\" cho phép so sòng phẳng. (2) Phạt vật lý J_g trên 20 s tương "
             "lai dùng lực sóng thật, nên một phần chênh lệch PhyLSTM2 so với mạng không vật lý có thể đến từ thông tin thêm "
             "lúc huấn luyện chứ không chỉ từ ràng buộc phương trình.", ""]
    for c, ch in enumerate(CHANNELS):
        lines += [f"## {ch.split('_')[0].capitalize()} ({L.UNITS[c]})", "",
                  "| Tập | Mô hình | σ | RMSE | MAE | " + " | ".join(f"RMSE {t} s" for t in LEADS) + " |",
                  "|---|---|---|---|---|" + "---|" * len(LEADS)]
        for label, splits in [("**6 fold**", cv), ("Ngoài phân phối: Hs = 0,40 m", ["ood_Hs"]),
                              ("Ngoài phân phối: β = 45°", ["ood_beta"])]:
            for model, name in MODELS:
                got = [per_split(res_lstm, res_phy, model, s) for s in splits]
                if any(g is None for g in got):
                    lines.append(f"| {label} | {name} | | chưa có | | " + " | " * (len(LEADS) - 1) + " |")
                    continue
                cell = {}
                for k in ("RMSE", "MAE"):
                    v = [g[0][k][c] for g in got]
                    cell[k] = f"{np.mean(v):.4g} ± {np.std(v, ddof=1):.2g}" if len(v) > 1 else f"{v[0]:.4g}"
                leads = [f"{np.mean([g[1][int(t * 5) - 1, c] for g in got]):.4g}" for t in LEADS]
                sg = np.mean([sig[s][c] for s in splits])
                lines.append(f"| {label} | {name} | {sg:.4g} | {cell['RMSE']} | {cell['MAE']} | " + " | ".join(leads) + " |")
        lines.append("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"report -> {OUT}")


if __name__ == "__main__":
    main()
