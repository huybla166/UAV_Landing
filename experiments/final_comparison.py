"""Final comparison of the optimised predictor against the paper's LSTM and the intermediate models.

Forecasting: nRMSE at 1/2/5 s, useful horizon, normalised pinball loss over 0-5 s (CRPS proxy), raw band
coverage, Diebold-Mariano tests. Decisions: CRC-by-heading + ACI with nested (leakage-free) calibration,
8 threshold x window settings, alpha = 0.05.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import online_elf as oe
import run_regime_calibration as rc
import windows_lib as wl
from common import FS, HORIZON, RESULTS, RUNS, load_all, load_preds, origins, save_preds, targets

data = load_all()
truth = {r: targets(x) for r, x in data.items()}
scale = {r: data[r].std(0) for r in RUNS}
LV = np.round(np.arange(0.1, 1.0, 0.1), 1)
CH = ["Heave", "Roll", "Pitch"]
SETTINGS = [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]

# ridge as a point-forecast base for the decision layer (needs nested forecasts too)
if not os.path.exists(os.path.join(RESULTS, "preds_ridgeq_nested.npz")):
    p, nest = load_preds("ridge_direct")[0], {}
    for t in RUNS:
        for j in RUNS:
            if j != t:
                W = [oe.fit_ridge([r for r in RUNS if r not in (t, j)], c)[0] for c in range(3)]
                nest[f"{t}__{j}"] = np.repeat(np.stack([oe.lin_forecast(W[c], j, c) for c in range(3)], -1)[..., None], 9, -1)
    save_preds("ridgeq", p, {r: np.repeat(p[r][..., None], 9, -1) for r in RUNS}, LV)
    np.savez_compressed(os.path.join(RESULTS, "preds_ridgeq_nested.npz"), **nest)

MODELS = [("lstm_paper", "LSTM của bài (Wang et al. 2026)"), ("ar_aic", "AR(p) theo AIC"),
          ("tirex_ft", "TiRex tinh chỉnh (bản trước)"), ("ridge_direct", "Tuyến tính trực tiếp (kiểu DLinear)"),
          ("final", "MÔ HÌNH TỐI ƯU: TiRex+FreDF ⊕ tuyến tính, kết hợp trực tuyến ELF")]


def fc_metrics(name):
    p, q, lv = load_preds(name)
    nr = np.mean([np.sqrt(np.mean((p[r] - truth[r]) ** 2, 0)) / scale[r] for r in RUNS], 0)   # (H, 3)
    useful = []
    for c in range(3):
        bad = np.nonzero(nr[:, c] >= 0.9)[0]
        useful.append(bad[0] / FS if len(bad) else HORIZON / FS)
    out = dict(nr=nr, useful=useful, pin=None, cov=None)
    if q is not None and not np.allclose(q[RUNS[0]][..., 0], q[RUNS[0]][..., -1]):
        pins, covs = [], []
        for r in RUNS:
            e = (truth[r][..., None] - q[r]) / scale[r][None, None, :, None]
            pins.append(np.mean(np.maximum(lv * e, (lv - 1) * e)[:, :25], axis=(0, 1, 3)))
            covs.append(np.mean((truth[r] >= q[r][..., 0]) & (truth[r] <= q[r][..., -1]), axis=(0, 1)))
        out["pin"], out["cov"] = np.mean(pins, 0), np.mean(covs, 0)
    return out


def dm(a, b, c, lag=5):
    pa, pb = load_preds(a)[0], load_preds(b)[0]
    ea = np.concatenate([np.mean(((pa[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    eb = np.concatenate([np.mean(((pb[r][:, :25, c] - truth[r][:, :25, c]) / scale[r][c]) ** 2, 1) for r in RUNS])
    d = ea - eb; n = len(d); dc = d - d.mean()
    g = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))
    z = d.mean() / np.sqrt(var / n)
    return float(z), float(2 * (1 - norm.cdf(abs(z)))), float(100 * (1 - eb.mean() / ea.mean()))


FM = {m: fc_metrics(m) for m, _ in MODELS}
L = ["# Kết quả tối ưu mô hình dự đoán – so sánh cuối cùng (NTUA, leave-one-run-out)", "",
     "## 1. Dự đoán: nRMSE (heave / roll / pitch), càng nhỏ càng tốt", "",
     "| Mô hình | 1 s | 2 s | 5 s | Tầm hữu ích [s] | Pinball 0–5 s (xác suất) | Độ phủ dải 10–90 % |", "|---|---|---|---|---|---|---|"]
for m, lab in MODELS:
    f = FM[m]
    cell = lambda k: " / ".join(f"{v:.2f}" for v in f["nr"][k - 1])
    pin = " / ".join(f"{v:.3f}" for v in f["pin"]) if f["pin"] is not None else "– (chỉ dự đoán điểm)"
    cov = " / ".join(f"{v:.2f}" for v in f["cov"]) if f["cov"] is not None else "–"
    L.append(f"| {lab} | {cell(5)} | {cell(10)} | {cell(25)} | {' / '.join(f'{u:.1f}' for u in f['useful'])} | {pin} | {cov} |")
L += ["", "## 2. Kiểm định Diebold–Mariano (MSE trung bình 0–5 s; dương = mô hình thứ hai tốt hơn)", "",
      "| So sánh | Heave: giảm MSE, p | Roll | Pitch |", "|---|---|---|---|"]
DM = {}
for a, b, lab in (("lstm_paper", "final", "Tối ưu so với LSTM của bài"), ("tirex_ft", "final", "Tối ưu so với TiRex tinh chỉnh"),
                  ("ridge_direct", "final", "Tối ưu so với tuyến tính đứng riêng")):
    cells = []
    for c in range(3):
        z, p, red = dm(a, b, c)
        DM[f"{a}->{b}|{c}"] = (z, p, red)
        cells.append(f"{red:+.0f} %, p = {p:.1e}")
    L.append(f"| {lab} | " + " | ".join(cells) + " |")


# ---------------- decisions ----------------
def dec_metrics(base, method, rth, pth, T):
    wl.set_base(base); rc.tx_med, rc.tx_w = wl.tx_med, wl.tx_w
    H = int(round(T * FS))
    Ds, Us, per = [], [], []
    for r in RUNS:
        if method == "point":
            D = wl.decide("lstm_point", r, H, rth, pth)[0]; U = ~wl.safe_truth(r, H, rth, pth)
        else:
            D, U = rc.decide("crc_heading_aci", r, H, rth, pth)
        Ds.append(D); Us.append(U); per.append((D & U).mean())
    D, U = np.concatenate(Ds), np.concatenate(Us)
    return dict(fsr=(D & U).sum() / max(D.sum(), 1), recall=(D & ~U).sum() / max((~U).sum(), 1),
                violation=(D & U).mean(), worst=max(per))


DEC = [("A", "lstmq", "point", "A. LSTM của bài (điểm + ngưỡng)"),
       ("R", "ridgeq", "crc", "Tuyến tính + CRC theo hướng + ACI"),
       ("C", "tirex_ft", "crc", "TiRex tinh chỉnh + CRC theo hướng + ACI (bản trước)"),
       ("F", "final", "crc", "MÔ HÌNH TỐI ƯU + CRC theo hướng + ACI")]
res = {}
for key, base, meth, lab in DEC:
    for rth, pth in SETTINGS:
        for T in (5.0, 10.0):
            res[(key, rth, T)] = dec_metrics(base, meth, rth, pth, T)
L += ["", "## 3. Quyết định 'an toàn trong 5 s tới' (roll ±1.5°, pitch ±1.0°), α = 0.05", "",
      "| Phương pháp | FSR | Recall | P(báo an toàn & nguy hiểm) | Rủi ro tệ nhất 1 lần chạy |", "|---|---|---|---|---|"]
for key, _, _, lab in DEC:
    m = res[(key, 1.5, 5.0)]
    L.append(f"| {lab} | {m['fsr']:.2f} | {m['recall']:.2f} | {m['violation']:.3f} | {m['worst']:.3f} |")
L += ["", "## 4. Cả 8 tổ hợp: recall (rủi ro tệ nhất 1 lần chạy)", "",
      "| Ngưỡng | T | " + " | ".join(lab for _, _, _, lab in DEC) + " |", "|---|---|" + "---|" * len(DEC)]
for rth, pth in SETTINGS:
    for T in (5.0, 10.0):
        L.append(f"| ±{rth}°/±{pth}° | {T:.0f} s | " + " | ".join(
            f"{res[(k, rth, T)]['recall']:.2f} ({res[(k, rth, T)]['worst']:.3f})" for k, _, _, _ in DEC) + " |")
avg = {k: (np.mean([res[(k, r_, T)]["recall"] for r_, _ in SETTINGS for T in (5.0, 10.0)]),
           max(res[(k, r_, T)]["worst"] for r_, _ in SETTINGS for T in (5.0, 10.0))) for k, _, _, _ in DEC}
L.append("| **Trung bình recall / rủi ro tệ nhất** | | " + " | ".join(f"**{avg[k][0]:.2f} / {avg[k][1]:.3f}**" for k, _, _, _ in DEC) + " |")
with open(os.path.join(RESULTS, "final_summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(RESULTS, "final_results.json"), "w", encoding="utf-8") as fh:
    json.dump({"dm": DM, "decisions": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}}, fh, indent=1, default=float)
print("\n".join(L))

# ---------------- figures ----------------
lead = np.arange(1, HORIZON + 1) / FS
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
style = {"lstm_paper": ("C3", "-", 1.2), "ar_aic": ("C1", "-", 1.0), "tirex_ft": ("C0", "--", 1.3),
         "ridge_direct": ("C4", "-", 1.2), "final": ("k", "-", 2.2)}
for m, lab in MODELS:
    col, ls, lw = style[m]
    for c, ax in enumerate(axes):
        ax.plot(lead[:50], FM[m]["nr"][:50, c], ls, color=col, lw=lw, label=lab)
for c, ax in enumerate(axes):
    ax.axhline(1, color="grey", lw=0.6); ax.set_title(CH[c]); ax.set_xlabel("Tầm dự đoán [s]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("nRMSE"); axes[0].set_ylim(0, 1.4); axes[-1].legend(fontsize=7.5)
fig.suptitle("Mô hình tối ưu so với các mô hình khác (dữ liệu thật NTUA)")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "figures", "final_forecast.png"), dpi=130)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for (k, _, _, lab), col, mk in zip(DEC, ("C3", "C4", "C0", "k"), ("s", "^", "D", "*")):
    pts = [(res[(k, r_, T)]["recall"], res[(k, r_, T)]["worst"]) for r_, _ in SETTINGS for T in (5.0, 10.0)]
    ax.scatter(*zip(*pts), color=col, marker=mk, s=60 if mk == "*" else 30, label=lab, alpha=0.85)
ax.axhline(0.05, color="r", ls="--", lw=1); ax.text(0.01, 0.06, "α = 5 %", color="r", fontsize=8)
ax.set_xlabel("Recall (tận dụng cửa sổ an toàn) →"); ax.set_ylabel("Rủi ro tệ nhất trong 1 lần chạy ↓")
ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
ax.set_title("8 tổ hợp ngưỡng × cửa sổ: an toàn vs tận dụng")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "figures", "final_decisions.png"), dpi=130)
print("figures saved")
