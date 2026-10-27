"""Head-to-head comparison: the paper's LSTM predictor vs the proposed predictor, on the real NTUA data.

  A  LSTM (paper)            : LSTM point forecast, thresholds checked on the point forecast (Wang et al. 2026)
  B  LSTM + proposed layer   : same LSTM forecasts, decisions by CRC-by-heading + ACI (ablation)
  C  Proposed                : fine-tuned TiRex + CRC-by-heading + ACI

All trained components use leave-one-run-out; conformal calibration uses nested models (never trained on the
test run). Statistics: Diebold-Mariano test with Newey-West variance for forecast errors; moving-block bootstrap
(blocks of 10 s inside each run) for decision metrics.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch import nn

import run_regime_calibration as rc
import windows_lib as wl
from common import FS, HORIZON, RESULTS, RUNS, load_all, load_preds, origins, save_preds, targets
from run_baselines import PaperLSTM, windows as lstm_windows

torch.set_num_threads(4)
data = load_all()
truth = {r: targets(x) for r, x in data.items()}
SETTINGS = [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]
MAIN = (1.5, 1.0, 5.0)
FIG = os.path.join(RESULTS, "figures")


# ---------------- nested LSTM forecasts (for leakage-free calibration of variant B) ----------------
def train_lstm_predict(train_runs, target, L=10, epochs=60, seeds=(0, 1, 2)):
    """Literal replication of the paper's LSTM (see run_baselines.run_lstm), trained on `train_runs`."""
    train = [data[r] for r in train_runs]
    mu, sd = np.concatenate(train).mean(0), np.concatenate(train).std(0)
    X, Y = lstm_windows([(s - mu) / sd for s in train], L)
    X, Y = torch.tensor(X), torch.tensor(Y)
    x = (data[target] - mu) / sd
    o = origins(len(x))
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = PaperLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.2)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(X))
            for b in range(0, len(X), 64):
                i = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.mse_loss(model(X[i]), Y[i]).backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            hist = torch.tensor(np.stack([x[i - L:i] for i in o]), dtype=torch.float32)
            steps = []
            for _ in range(HORIZON):
                nxt = model(hist)
                steps.append(nxt)
                hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
            preds.append(torch.stack(steps, 1).numpy())
    return np.mean(preds, 0) * sd + mu


nested_path = os.path.join(RESULTS, "preds_lstmq_nested.npz")
if not os.path.exists(nested_path):
    nested = {}
    for t in RUNS:
        for j in RUNS:
            if j != t:
                p = train_lstm_predict([r for r in RUNS if r not in (t, j)], j)
                nested[f"{t}__{j}"] = np.repeat(p[..., None], 9, -1)  # point forecast as degenerate quantiles
                print(f"  nested LSTM test={t} cal={j} done", flush=True)
    np.savez_compressed(nested_path, **nested)
lstm_pred = load_preds("lstm_paper")[0]
save_preds("lstmq", lstm_pred, {r: np.repeat(lstm_pred[r][..., None], 9, -1) for r in RUNS}, np.arange(0.1, 1.0, 0.1))


# ---------------- decisions ----------------
def decisions(variant, rth, pth, T):
    H = int(round(T * FS))
    out = {}
    for r in RUNS:
        if variant == "A":
            D = wl.decide("lstm_point", r, H, rth, pth)[0]
            U = ~wl.safe_truth(r, H, rth, pth)
        else:
            D, U = rc.decide("crc_heading_aci", r, H, rth, pth)
        out[r] = (D, U)
    return out


def use_base(name):
    wl.set_base(name)
    rc.tx_med, rc.tx_w = wl.tx_med, wl.tx_w


def metrics(dec):
    D = np.concatenate([dec[r][0] for r in RUNS]); U = np.concatenate([dec[r][1] for r in RUNS])
    per = {r: (dec[r][0] & dec[r][1]).mean() for r in RUNS}
    return dict(declared=D.mean(), violation=(D & U).mean(), fsr=(D & U).sum() / max(D.sum(), 1),
                recall=(D & ~U).sum() / max((~U).sum(), 1), worst_run=max(per.values()))


VARIANTS = {"A": ("LSTM của bài (dự đoán điểm + ngưỡng)", "lstmq"),
            "B": ("LSTM + lớp quyết định đề xuất (CRC theo hướng + ACI)", "lstmq"),
            "C": ("Đề xuất: TiRex tinh chỉnh + CRC theo hướng + ACI", "tirex_ft")}
dec_all = {}
for v, (_, base) in VARIANTS.items():
    use_base(base)
    for rth, pth in SETTINGS:
        for T in (5.0, 10.0):
            dec_all[(v, rth, T)] = decisions(v, rth, pth, T)


# ---------------- moving-block bootstrap for decision differences ----------------
def block_boot(dec_a, dec_b, n_boot=2000, block=10, seed=0):
    rng = np.random.default_rng(seed)
    diffs = {"fsr": [], "recall": [], "violation": []}
    for _ in range(n_boot):
        Da, Ua, Db, Ub = [], [], [], []
        for r in RUNS:
            n = len(dec_a[r][0])
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            Da.append(dec_a[r][0][idx]); Ua.append(dec_a[r][1][idx]); Db.append(dec_b[r][0][idx]); Ub.append(dec_b[r][1][idx])
        Da, Ua, Db, Ub = map(np.concatenate, (Da, Ua, Db, Ub))
        fa, fb = (Da & Ua).sum() / max(Da.sum(), 1), (Db & Ub).sum() / max(Db.sum(), 1)
        ra, rb = (Da & ~Ua).sum() / max((~Ua).sum(), 1), (Db & ~Ub).sum() / max((~Ub).sum(), 1)
        diffs["fsr"].append(fb - fa); diffs["recall"].append(rb - ra); diffs["violation"].append((Db & Ub).mean() - (Da & Ua).mean())
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in diffs.items()}


# ---------------- forecast accuracy + Diebold-Mariano ----------------
def dm_test(e_a, e_b, lag):
    """Diebold-Mariano on loss differential d = e_a - e_b (positive => b better), Newey-West variance."""
    d = e_a - e_b
    n = len(d); dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


tirex = load_preds("tirex_ft")[0]
scale = {r: data[r].std(0) for r in RUNS}
CH = ["Heave", "Roll", "Pitch"]
fc_rows = []
for c, ch in enumerate(CH):
    row = {"channel": ch}
    for lead_s in (1, 2, 5):
        k = int(lead_s * FS)
        a = np.mean([np.sqrt(np.mean((lstm_pred[r][:, k - 1, c] - truth[r][:, k - 1, c]) ** 2)) / scale[r][c] for r in RUNS])
        b = np.mean([np.sqrt(np.mean((tirex[r][:, k - 1, c] - truth[r][:, k - 1, c]) ** 2)) / scale[r][c] for r in RUNS])
        row[f"{lead_s}s"] = (a, b)
    # DM on per-origin normalised MSE over the first 5 s (25 leads), pooled over runs (HAC lag 5 origins)
    ea = np.concatenate([np.mean(((lstm_pred[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    eb = np.concatenate([np.mean(((tirex[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    row["mse5_lstm"], row["mse5_prop"] = float(ea.mean()), float(eb.mean())
    row["dm_stat"], row["dm_p"] = dm_test(ea, eb, lag=5)
    row["runs_won"] = int(sum(
        np.mean(((tirex[r][:, :25, c] - truth[r][:, :25, c])) ** 2) < np.mean(((lstm_pred[r][:, :25, c] - truth[r][:, :25, c])) ** 2)
        for r in RUNS))
    fc_rows.append(row)


def useful(pred, c):
    nr = np.mean([np.sqrt(np.mean((pred[r][..., c] - truth[r][..., c]) ** 2, 0)) / scale[r][c] for r in RUNS], 0)
    bad = np.nonzero(nr >= 0.9)[0]
    return bad[0] / FS if len(bad) else HORIZON / FS


# ---------------- report ----------------
res = {"forecast": fc_rows, "decisions": {}, "bootstrap_main": {}}
for key, dec in dec_all.items():
    res["decisions"][f"{key[0]}|{key[1]}|{key[2]}"] = metrics(dec)
rth, pth, T = MAIN
res["bootstrap_main"]["C_vs_A"] = block_boot(dec_all[("A", rth, T)], dec_all[("C", rth, T)])
res["bootstrap_main"]["C_vs_B"] = block_boot(dec_all[("B", rth, T)], dec_all[("C", rth, T)])
res["bootstrap_main"]["B_vs_A"] = block_boot(dec_all[("A", rth, T)], dec_all[("B", rth, T)])
with open(os.path.join(RESULTS, "lstm_vs_proposed.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1, default=float)

L = ["# So sánh đối đầu: LSTM của bài vs phương pháp đề xuất (dữ liệu thật NTUA, leave-one-run-out)", "",
     "## 1. Độ chính xác dự đoán (nRMSE, 1.0 = không hơn đoán trung bình; càng nhỏ càng tốt)", "",
     "| Kênh | 1 s: LSTM → Đề xuất | 2 s: LSTM → Đề xuất | 5 s: LSTM → Đề xuất | Tầm hữu ích: LSTM → Đề xuất | "
     "MSE 0–5 s giảm | Diebold–Mariano p | Thắng ở số lần chạy |", "|---|---|---|---|---|---|---|---|"]
for c, row in enumerate(fc_rows):
    red = 100 * (1 - row["mse5_prop"] / row["mse5_lstm"])
    L.append(f"| {row['channel']} | " + " | ".join(f"{row[k][0]:.2f} → **{row[k][1]:.2f}**" for k in ("1s", "2s", "5s")) +
             f" | {useful(lstm_pred, c):.1f} s → **{useful(tirex, c):.1f} s** | {red:.0f} % | "
             f"{row['dm_p']:.1e} | {row['runs_won']}/4 |")
L += ["", f"## 2. Quyết định 'an toàn trong {T:.0f} s tới' (roll ±{rth}°, pitch ±{pth}°)", "",
      "| Phương pháp | Báo an toàn | P(báo an toàn & nguy hiểm) | Tỉ lệ báo nhầm (FSR) | Recall | Rủi ro tệ nhất trong 1 lần chạy |",
      "|---|---|---|---|---|---|"]
for v, (lab, _) in VARIANTS.items():
    m = res["decisions"][f"{v}|{rth}|{T}"]
    L.append(f"| {v}. {lab} | {m['declared']:.2f} | {m['violation']:.3f} | {m['fsr']:.2f} | {m['recall']:.2f} | {m['worst_run']:.3f} |")
L += ["", "### Khoảng tin cậy 95 % (bootstrap theo khối 10 s) cho chênh lệch", "",
      "| So sánh | Δ FSR | Δ Recall | Δ P(báo an toàn & nguy hiểm) |", "|---|---|---|---|"]
for k, lab in (("C_vs_A", "C so với A"), ("B_vs_A", "B so với A (chỉ đổi lớp quyết định)"), ("C_vs_B", "C so với B (chỉ đổi mô hình dự đoán)")):
    b = res["bootstrap_main"][k]
    L.append(f"| {lab} | " + " | ".join(f"{b[m][0]:+.3f} [{b[m][1]:+.3f}, {b[m][2]:+.3f}]" for m in ("fsr", "recall", "violation")) + " |")
L += ["", "## 3. Trên cả 8 tổ hợp ngưỡng × độ dài cửa sổ: FSR / recall / rủi ro tệ nhất 1 lần chạy", "",
      "| Ngưỡng | T | Tỉ lệ an toàn thật | A. LSTM của bài | B. LSTM + lớp đề xuất | C. Đề xuất |", "|---|---|---|---|---|---|"]
for rth_, pth_ in SETTINGS:
    for T_ in (5.0, 10.0):
        H = int(T_ * FS)
        base = np.mean(np.concatenate([wl.safe_truth(r, H, rth_, pth_) for r in RUNS]))
        cells = []
        for v in VARIANTS:
            m = res["decisions"][f"{v}|{rth_}|{T_}"]
            cells.append(f"{m['fsr']:.2f} / {m['recall']:.2f} / {m['worst_run']:.3f}")
        L.append(f"| ±{rth_}°/±{pth_}° | {T_:.0f} s | {base:.2f} | " + " | ".join(cells) + " |")
with open(os.path.join(RESULTS, "lstm_vs_proposed.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------- figures ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
lead = np.arange(1, HORIZON + 1) / FS
for c, ax in enumerate(axes):
    for pred, lab, col in ((lstm_pred, "LSTM của bài", "C3"), (tirex, "Đề xuất (TiRex tinh chỉnh)", "C0")):
        per_run = np.array([np.sqrt(np.mean((pred[r][..., c] - truth[r][..., c]) ** 2, 0)) / scale[r][c] for r in RUNS])
        ax.plot(lead[:50], per_run.mean(0)[:50], color=col, lw=2, label=lab)
        ax.fill_between(lead[:50], per_run.min(0)[:50], per_run.max(0)[:50], color=col, alpha=0.15)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_title(CH[c]); ax.set_xlabel("Tầm dự đoán [s]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("nRMSE (vùng mờ = min–max 4 lần chạy)")
axes[0].set_ylim(0, 1.6); axes[0].legend(fontsize=8)
fig.suptitle("Độ chính xác dự đoán: LSTM của bài vs phương pháp đề xuất")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lstm_vs_proposed_forecast.png"), dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
labels = ["A. LSTM của bài", "B. LSTM + lớp đề xuất", "C. Đề xuất"]
cols = ["C3", "C1", "C0"]
m = [res["decisions"][f"{v}|{rth}|{T}"] for v in VARIANTS]
for ax, key, title in zip(axes, ("fsr", "recall", "worst_run"),
                          ("Tỉ lệ báo nhầm an toàn (FSR) ↓", "Tận dụng cửa sổ an toàn (recall) ↑", "Rủi ro tệ nhất trong 1 lần chạy ↓")):
    vals = [mm[key] for mm in m]
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}" if key != "worst_run" else f"{v:.3f}", ha="center", fontsize=9)
    if key == "worst_run":
        ax.axhline(0.05, color="r", ls="--", lw=1); ax.text(2.4, 0.052, "α = 5 %", color="r", fontsize=8, ha="right")
    ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelsize=8); ax.grid(alpha=0.3, axis="y")
fig.suptitle(f"Quyết định hạ cánh: an toàn trong {T:.0f} s tới, roll ±{rth}°, pitch ±{pth}°")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lstm_vs_proposed_decision.png"), dpi=130)

fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
H = int(T * FS)
for ax, r in zip(axes, ["noWing_headQ", "noWing_followQ"]):
    x = data[r]; o = origins(len(x)); t = np.arange(len(x)) / FS
    m_roll = np.interp(np.arange(len(x)), o, wl.ref[r][:, 1])
    ax.plot(t, x[:, 1] - m_roll, color="k", lw=0.7, label="Roll thật (đã trừ nghiêng tĩnh)")
    for s in (rth, -rth):
        ax.axhline(s, color="grey", ls="--", lw=0.8)
    S = wl.safe_truth(r, H, rth, pth)
    rows = [("Thật sự an toàn 5 s", S, "0.5"), ("A. LSTM của bài báo an toàn", dec_all[("A", rth, T)][r][0], "C3"),
            ("C. Đề xuất báo an toàn", dec_all[("C", rth, T)][r][0], "C0")]
    for k, (lab, flag, col) in enumerate(rows):
        yk = -3.0 - 0.45 * k
        ax.scatter(o[flag] / FS, np.full(flag.sum(), yk), marker="|", s=60, color=col, label=lab)
        bad = flag & ~S
        if k > 0 and bad.any():
            ax.scatter(o[bad] / FS, np.full(bad.sum(), yk), marker="x", s=30, color="red")
    nA = int((dec_all[("A", rth, T)][r][0] & ~S).sum()); nC = int((dec_all[("C", rth, T)][r][0] & ~S).sum())
    ax.set_title(f"{r}: × đỏ = báo an toàn nhưng thực tế nguy hiểm  (LSTM: {nA} lần, Đề xuất: {nC} lần)")
    ax.set_ylabel("Roll [deg]"); ax.set_ylim(-4.4, 4); ax.grid(alpha=0.25)
axes[0].legend(fontsize=7.5, loc="upper right", ncol=2)
axes[-1].set_xlabel("Thời gian [s]")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lstm_vs_proposed_timeline.png"), dpi=130)
print("figures saved")
