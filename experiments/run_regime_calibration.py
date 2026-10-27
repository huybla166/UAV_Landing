"""Regime-aware Conformal Risk Control for landing-window decisions (fixes run-level risk violations).

Pooled CRC only guarantees P(declared safe AND unsafe) <= alpha on average over all sea conditions.
Here the calibration is made conditional on the operating regime:

  crc_pooled      : CRC calibrated on all other runs (previous result)
  crc_heading     : Mondrian CRC - calibrate only on runs with the same wave heading (head vs following,
                    known on board from heading + wave direction)
  crc_local       : localized (weighted) CRC - calibration origins weighted by a Gaussian kernel on causal
                    "sea fingerprint" features of the last 60 s (motion RMS, encounter periods); needs no labels
  *_aci           : the same plus adaptive conformal inference online (delayed feedback)

Decision: declare "safe for the next T s" if the TiRex risk score <= lambda; lambda chosen so that the
(weighted) calibration risk with finite-sample correction (sum w L + w_test) / (sum w + w_test) <= alpha.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

import windows_lib as wl
from windows_lib import (FS, GAMMA, MAIN, RESULTS, RUNS, STRIDE, data, origins, ref, risk_scores, safe_truth,
                         tx_med, tx_w)

ALPHA = 0.05
HEADING = {"noWing_headQ": "head", "wing_headQ": "head", "noWing_followQ": "follow", "wing_followQ": "follow"}
BANDWIDTH = 1.0  # kernel bandwidth in standardised feature units


# ---------------- causal sea-state fingerprint of the last 60 s ----------------
def zero_cross_period(x):
    x = x - x.mean()
    up = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    return len(x) / FS / max(len(up), 1)


def features(r):
    x = data[r]
    rows = []
    for i in origins(len(x)):
        w = x[i - 300:i]
        rows.append([w[:, 0].std(), w[:, 1].std(), w[:, 2].std(),
                     zero_cross_period(w[:, 0]), zero_cross_period(w[:, 2])])
    return np.log(np.array(rows))  # log: scale-type features


FEAT = {r: features(r) for r in RUNS}


# ---------------- weighted CRC ----------------
def crc_lambda_weighted(cal_s, cal_bad, w, alpha):
    """Largest lambda with (sum_{s_i<=lambda} w_i*bad_i + 1) / (sum_i w_i + 1) <= alpha (test weight = 1)."""
    order = np.argsort(cal_s)
    s, b = cal_s[order], np.cumsum((w * cal_bad)[order])
    ok = (b + 1.0) / (w.sum() + 1.0) <= alpha
    ok &= np.append(s[1:] != s[:-1], True)
    return s[np.nonzero(ok)[0][-1]] if ok.any() else -np.inf


def decide(method, test, H, rth, pth, alpha=ALPHA):
    base, aci = method.replace("_aci", ""), method.endswith("_aci")
    if base == "crc_heading":
        cal_runs = [r for r in RUNS if r != test and HEADING[r] == HEADING[test]]
    else:
        cal_runs = [r for r in RUNS if r != test]
    cal_s = np.concatenate([risk_scores(*wl.cal_mw(test, j), ref[j], H, rth, pth) for j in cal_runs])
    cal_bad = np.concatenate([~safe_truth(j, H, rth, pth) for j in cal_runs]).astype(float)
    s = risk_scores(tx_med[test], tx_w[test], ref[test], H, rth, pth)
    unsafe = ~safe_truth(test, H, rth, pth)

    if base == "crc_local":
        F = np.concatenate([FEAT[j] for j in cal_runs])
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Fz, Tz = (F - mu) / sd, (FEAT[test] - mu) / sd
        W = np.exp(-((Tz[:, None, :] - Fz[None]) ** 2).sum(-1) / (2 * BANDWIDTH ** 2))  # (n_test, n_cal)
    else:
        W = None

    def lam(k, a):
        w = np.ones(len(cal_s)) if W is None else W[k]
        return crc_lambda_weighted(cal_s, cal_bad, w, min(max(a, 1e-3), 0.999))

    D = np.zeros(len(s), bool)
    a_t, delay = alpha, int(np.ceil(H / STRIDE))
    for k in range(len(s)):
        if aci and k - delay >= 0:
            a_t += GAMMA * (alpha - float(D[k - delay] and unsafe[k - delay]))
        D[k] = s[k] <= lam(k, a_t if aci else alpha)
    return D, unsafe


METHODS = ["crc_pooled", "crc_pooled_aci", "crc_heading", "crc_heading_aci", "crc_local", "crc_local_aci"]
LABELS = {"crc_pooled": "CRC gộp (kết quả trước)", "crc_pooled_aci": "CRC gộp + ACI",
          "crc_heading": "CRC theo hướng sóng (Mondrian)", "crc_heading_aci": "CRC theo hướng sóng + ACI",
          "crc_local": "CRC cục bộ theo dấu vân tay biển", "crc_local_aci": "CRC cục bộ + ACI"}


def cp_upper(k, n, level=0.95):
    return 1.0 if k == n else beta.ppf(level, k + 1, n - k)  # one-sided Clopper-Pearson upper bound


def evaluate(method, rth, pth, T, alpha=ALPHA):
    H = int(round(T * FS))
    per, Ds, Us = {}, [], []
    for r in RUNS:
        D, U = decide(method, r, H, rth, pth, alpha)
        Ds.append(D); Us.append(U)
        k, n = int((D & U).sum()), len(D)
        per[r] = dict(declared=D.mean(), violation=k / n, viol_ub95=cp_upper(k, n),
                      fsr=k / max(D.sum(), 1), recall=(D & ~U).sum() / max((~U).sum(), 1))
    D, U = np.concatenate(Ds), np.concatenate(Us)
    pooled = dict(declared=D.mean(), violation=(D & U).mean(), fsr=(D & U).sum() / max(D.sum(), 1),
                  recall=(D & ~U).sum() / max((~U).sum(), 1),
                  max_run_violation=max(v["violation"] for v in per.values()),
                  runs_over_alpha=sum(v["violation"] > alpha for v in per.values()))
    return pooled, per


if __name__ == "__main__":
    out = {"alpha": ALPHA, "bandwidth": BANDWIDTH, "main": MAIN, "methods": {}, "grid": []}
    for m in METHODS:
        pooled, per = evaluate(m, MAIN["roll_th"], MAIN["pitch_th"], MAIN["T"])
        out["methods"][m] = {"pooled": pooled, "per_run": per}
    for rth, pth in [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]:
        for T in (5.0, 10.0):
            for m in METHODS:
                pooled, per = evaluate(m, rth, pth, T)
                out["grid"].append(dict(roll_th=rth, pitch_th=pth, T=T, method=m, **pooled))
    sens = {}
    for bw in (0.5, 1.0, 2.0):
        BANDWIDTH = bw
        sens[bw] = evaluate("crc_local", MAIN["roll_th"], MAIN["pitch_th"], MAIN["T"])[0]
    BANDWIDTH = 1.0
    out["bandwidth_sensitivity"] = sens
    with open(os.path.join(RESULTS, "regime_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)

    L = ["# Hiệu chỉnh theo chế độ sóng – Conformal Risk Control (α = %.2f)" % ALPHA, "",
         f"Quyết định 'an toàn trong {MAIN['T']:.0f} s tới', roll ±{MAIN['roll_th']}°, pitch ±{MAIN['pitch_th']}°. "
         "Mục tiêu: P(báo an toàn & nguy hiểm) ≤ α **trong từng lần chạy**, không chỉ trung bình.", "",
         "## 1. Kết quả chính", "",
         "| Phương pháp | P(báo an toàn & nguy hiểm), gộp | Lớn nhất trong 4 lần chạy | Số lần chạy vượt α | FSR | Recall |",
         "|---|---|---|---|---|---|"]
    for m in METHODS:
        q = out["methods"][m]["pooled"]
        L.append(f"| {LABELS[m]} | {q['violation']:.3f} | {q['max_run_violation']:.3f} | {q['runs_over_alpha']}/4 | "
                 f"{q['fsr']:.2f} | {q['recall']:.2f} |")
    L += ["", "## 2. Theo từng lần chạy: P(báo an toàn & nguy hiểm) [cận trên 95 %] / recall", "",
          "| Phương pháp | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
    for m in METHODS:
        per = out["methods"][m]["per_run"]
        L.append(f"| {LABELS[m]} | " + " | ".join(
            f"{per[r]['violation']:.3f} [{per[r]['viol_ub95']:.3f}] / {per[r]['recall']:.2f}" for r in RUNS) + " |")
    L += ["", "## 3. Mọi mức ngưỡng × độ dài cửa sổ: lớn nhất theo lần chạy / recall", "",
          "| Ngưỡng | T | " + " | ".join(LABELS[m] for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for rth, pth in [(1.0, 0.75), (1.5, 1.0), (2.0, 1.25), (2.5, 1.5)]:
        for T in (5.0, 10.0):
            g = {d["method"]: d for d in out["grid"] if d["roll_th"] == rth and d["T"] == T}
            L.append(f"| ±{rth}°/±{pth}° | {T:.0f} s | " + " | ".join(
                f"{g[m]['max_run_violation']:.3f} / {g[m]['recall']:.2f}" for m in METHODS) + " |")
    L += ["", "## 4. Độ nhạy theo bandwidth của CRC cục bộ", "", "| Bandwidth | Gộp | Lớn nhất theo lần chạy | Recall |",
          "|---|---|---|---|"]
    for bw, q in sens.items():
        L.append(f"| {bw} | {q['violation']:.3f} | {q['max_run_violation']:.3f} | {q['recall']:.2f} |")
    with open(os.path.join(RESULTS, "regime_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    # figure: per-run violation per method (bars) with alpha line and CP upper bounds
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(RUNS))
    wbar = 0.13
    for i, m in enumerate(METHODS):
        per = out["methods"][m]["per_run"]
        v = [per[r]["violation"] for r in RUNS]
        ub = [per[r]["viol_ub95"] - per[r]["violation"] for r in RUNS]
        ax.bar(xs + (i - 2.5) * wbar, v, wbar, yerr=[np.zeros(4), ub], capsize=2, label=LABELS[m],
               color=f"C{i}", alpha=0.9 if "aci" in m else 0.55)
    ax.axhline(ALPHA, color="r", ls="--", lw=1.2, label=f"α = {ALPHA}")
    ax.set_xticks(xs, RUNS)
    ax.set_ylabel("P(báo an toàn & nguy hiểm)")
    ax.set_title("Rủi ro theo từng lần chạy (thanh lỗi = cận trên 95 % Clopper-Pearson)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "regime_per_run_risk.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r, mk in zip(RUNS, "os^D"):
        ax.scatter(FEAT[r][:, 4], FEAT[r][:, 2], s=8, marker=mk, alpha=0.6, label=r)
    ax.set_xlabel("log chu kỳ cắt không của pitch (60 s gần nhất)")
    ax.set_ylabel("log độ lệch chuẩn pitch (60 s gần nhất)")
    ax.set_title("Dấu vân tay biển tính từ dữ liệu quá khứ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "sea_fingerprint.png"), dpi=130)
    print("figures saved")
