"""Conformal calibration of TiRex forecasts and risk-aware landing-window decisions (main experiment).
Method definitions live in windows_lib.py."""
from windows_lib import *  # noqa: F401,F403
import json, os
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 1) per-lead marginal coverage: raw TiRex vs split-conformal (CQR-style) ----------------
cov_raw, cov_cp, wid_cp = [], [], []
for r in RUNS:
    others = [j for j in RUNS if j != r]
    y = truth[r]
    lo, hi = tx_q[r][..., i_lo], tx_q[r][..., i_hi]
    cov_raw.append(((y >= lo) & (y <= hi)).mean(0))
    cal = np.concatenate([np.abs(truth[j] - tx_med[j]) / tx_w[j] for j in others])  # (n, H, 3)
    Q = np.apply_along_axis(lambda s: conformal_q(s, ALPHA), 0, cal)            # (H, 3)
    cov_cp.append((np.abs(y - tx_med[r]) <= Q[None] * tx_w[r]).mean(0))
    wid_cp.append((2 * Q[None] * tx_w[r]).mean(0) / data[r].std(0))
cov_raw, cov_cp, wid_cp = np.mean(cov_raw, 0), np.mean(cov_cp, 0), np.mean(wid_cp, 0)

# ---------------- 2) decisions ----------------
results = {"main": MAIN, "alpha": ALPHA, "gamma": GAMMA, "methods": {}}
for meth in METHODS:
    pooled, rows = evaluate(meth, MAIN["roll_th"], MAIN["pitch_th"], MAIN["T"])
    results["methods"][meth] = {"pooled": pooled, "per_run": rows}

grid = []
for rth, pth in [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]:
    for T in (5.0, 10.0):
        for meth in METHODS:
            pooled, _ = evaluate(meth, rth, pth, T)
            grid.append(dict(roll_th=rth, pitch_th=pth, T=T, method=meth, **pooled))
results["grid"] = grid

curves = {}
for meth in ("ar_cp", "tirex_cp", "tirex_cp_aci", "ar_crc", "tirex_crc", "tirex_crc_aci"):
    curves[meth] = []
    alphas = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3) if "crc" in meth else (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    for a in alphas:
        pooled, _ = evaluate(meth, MAIN["roll_th"], MAIN["pitch_th"], MAIN["T"], a)
        curves[meth].append(dict(alpha=a, **pooled))
results["curves"] = curves
with open(os.path.join(RESULTS, "windows_results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1, default=float)

# ---------------- tables ----------------
CH = ["Heave", "Roll", "Pitch"]
L = ["# Hiệu chỉnh conformal và chọn cửa sổ hạ cánh (NTUA, leave-one-run-out)", "",
     f"Mô hình nền: {TIREX}. Mục tiêu độ phủ 1 − α = {1 - ALPHA:.2f}.", "",
     "## 1. Độ phủ theo từng tầm dự đoán", "",
     "| Kênh | Loại dải | " + " | ".join(REPORT_STEPS) + " |", "|---|---|" + "---|" * len(REPORT_STEPS)]
for c in range(3):
    L.append(f"| {CH[c]} | TiRex 10–90 % (danh nghĩa 0.80) | " +
             " | ".join(f"{cov_raw[s - 1, c]:.2f}" for s in REPORT_STEPS.values()) + " |")
    L.append(f"| {CH[c]} | TiRex + conformal (mục tiêu 0.90) | " +
             " | ".join(f"{cov_cp[s - 1, c]:.2f}" for s in REPORT_STEPS.values()) + " |")
    L.append(f"| {CH[c]} | độ rộng dải conformal / std | " +
             " | ".join(f"{wid_cp[s - 1, c]:.2f}" for s in REPORT_STEPS.values()) + " |")
p = results["methods"]
L += ["", f"## 2. Quyết định 'an toàn trong {MAIN['T']:.0f} s tới' (roll ±{MAIN['roll_th']}°, pitch ±{MAIN['pitch_th']}°)", "",
      f"Tỉ lệ thời điểm thật sự an toàn: {p['lstm_point']['pooled']['base']:.2f} "
      f"(trên {p['lstm_point']['pooled']['n']} thời điểm). Dải conformal chung: độ phủ mục tiêu {1 - ALPHA:.2f}. "
      f"Conformal Risk Control: bảo đảm P(báo an toàn VÀ thật ra nguy hiểm) ≤ α = {ALPHA_CRC}.", "",
      "| Phương pháp | Báo an toàn | P(báo an toàn & nguy hiểm) | Tỉ lệ báo nhầm (FSR) | Tận dụng cửa sổ thật (recall) | Độ phủ chung của dải |",
      "|---|---|---|---|---|---|"]
for meth in METHODS:
    q = p[meth]["pooled"]
    jc = f"{q['joint_coverage']:.2f}" if q["joint_coverage"] is not None else "–"
    L.append(f"| {LABELS[meth]} | {q['declared']:.2f} ({q['n_declared']}) | {q['violation']:.3f} ({q['n_false_safe']}) | "
             f"{q['fsr']:.2f} | {q['recall']:.2f} | {jc} |")
L += ["", "### Theo từng lần chạy: FSR / recall", "", "| Phương pháp | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
for meth in METHODS:
    L.append(f"| {LABELS[meth]} | " + " | ".join(
        f"{p[meth]['per_run'][r]['fsr']:.2f} / {p[meth]['per_run'][r]['recall']:.2f}" for r in RUNS) + " |")
L += ["", "## 3. Các mức ngưỡng và độ dài cửa sổ khác: FSR / recall / P(báo an toàn & nguy hiểm)", "",
      "| Ngưỡng roll/pitch | T | Tỉ lệ an toàn thật | " + " | ".join(LABELS[m] for m in METHODS) + " |",
      "|---|---|---|" + "---|" * len(METHODS)]
for rth, pth in [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]:
    for T in (5.0, 10.0):
        g = {d["method"]: d for d in grid if d["roll_th"] == rth and d["T"] == T}
        L.append(f"| ±{rth}° / ±{pth}° | {T:.0f} s | {g['lstm_point']['base']:.2f} | " + " | ".join(
            f"{g[m]['fsr']:.2f} / {g[m]['recall']:.2f} / {g[m]['violation']:.3f}" for m in METHODS) + " |")
with open(os.path.join(RESULTS, "windows_summary.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------- figures ----------------
fig_dir = os.path.join(RESULTS, "figures")
lead = np.arange(1, HORIZON + 1) / FS
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for c, ax in enumerate(axes):
    ax.plot(lead, cov_raw[:, c], label="TiRex 10–90 % (danh nghĩa 0.80)", color="C7")
    ax.plot(lead, cov_cp[:, c], label="TiRex + conformal (mục tiêu 0.90)", color="C2", lw=1.8)
    ax.axhline(0.8, color="C7", ls=":", lw=0.9)
    ax.axhline(0.9, color="C2", ls=":", lw=0.9)
    ax.set_title(CH[c]); ax.set_xlabel("Tầm dự đoán [s]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Độ phủ thực tế"); axes[0].set_ylim(0.6, 1.0); axes[0].legend(fontsize=8)
fig.suptitle("Độ phủ của dải dự đoán theo tầm dự đoán (leave-one-run-out)")
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "coverage_vs_horizon.png"), dpi=130)

fig, ax = plt.subplots(figsize=(8, 5.5))
for meth, col, ls in (("tirex_cp", "C2", ":"), ("ar_crc", "C1", "-"), ("tirex_crc", "C2", "-"), ("tirex_crc_aci", "C0", "-")):
    cv = curves[meth]
    ax.plot([d["recall"] for d in cv], [d["fsr"] for d in cv], "o", ls=ls, color=col, label=LABELS[meth], ms=4)
    for d in cv:
        if meth == "tirex_crc" and d["alpha"] in (0.02, 0.05, 0.1, 0.2):
            ax.annotate(f"α={d['alpha']}", (d["recall"], d["fsr"]), fontsize=7, xytext=(4, -9), textcoords="offset points")
for meth, mk, col in (("lstm_point", "s", "C3"), ("tirex_point", "D", "C4"), ("tirex_raw80", "^", "C7")):
    q = p[meth]["pooled"]
    ax.plot(q["recall"], q["fsr"], mk, color=col, ms=9, label=LABELS[meth])
ax.set_xlabel("Tận dụng cửa sổ an toàn thật (recall) →  tốt hơn")
ax.set_ylabel("Tỉ lệ báo nhầm an toàn (FSR) ↓  tốt hơn")
ax.set_title(f"Đánh đổi an toàn – tận dụng (an toàn {MAIN['T']:.0f} s, roll ±{MAIN['roll_th']}°, pitch ±{MAIN['pitch_th']}°)")
ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "safety_tradeoff.png"), dpi=130)

H = int(MAIN["T"] * FS)
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
for ax, r in zip(axes, ["noWing_headQ", "noWing_followQ"]):
    x = data[r]
    o = origins(len(x))
    t = np.arange(len(x)) / FS
    m_roll = np.interp(np.arange(len(x)), o, ref[r][:, ROLL])
    ax.plot(t, x[:, ROLL] - m_roll, color="k", lw=0.7, label="Roll thật (đã trừ nghiêng tĩnh)")
    ax.axhline(MAIN["roll_th"], color="grey", ls="--", lw=0.8); ax.axhline(-MAIN["roll_th"], color="grey", ls="--", lw=0.8)
    S = safe_truth(r, H, MAIN["roll_th"], MAIN["pitch_th"])
    rows = [("Thật sự an toàn 5 s", S, "0.5"), ("LSTM của bài báo an toàn", decide("lstm_point", r, H, MAIN["roll_th"], MAIN["pitch_th"])[0], "C3"),
            ("TiRex + CRC + ACI báo an toàn", decide("tirex_crc_aci", r, H, MAIN["roll_th"], MAIN["pitch_th"], ALPHA_CRC)[0], "C2")]
    for k, (lab, flag, col) in enumerate(rows):
        yk = -3.0 - 0.45 * k
        ax.scatter(o[flag] / FS, np.full(flag.sum(), yk), marker="|", s=60, color=col, label=lab)
        bad = flag & ~S
        if k > 0 and bad.any():
            ax.scatter(o[bad] / FS, np.full(bad.sum(), yk), marker="x", s=30, color="red")
    ax.set_title(f"{r}: dấu × đỏ = báo an toàn nhưng thực tế vượt ngưỡng trong 5 s tới")
    ax.set_ylabel("Roll [deg]"); ax.set_ylim(-4.4, 4); ax.grid(alpha=0.25)
axes[0].legend(fontsize=7.5, loc="upper right", ncol=2)
axes[-1].set_xlabel("Thời gian [s]")
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "decision_timeline.png"), dpi=130)
print("figures saved")
