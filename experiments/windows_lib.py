"""Shared library: conformal calibration of TiRex forecasts and risk-aware landing-window decisions on the NTUA data.

Decision task (at every forecast origin, once per second):
    "Is the deck safe for the next T seconds?"  i.e. for every lead h <= T
        |roll(t+h)  - roll_ref|  <= ROLL_TH   and   |pitch(t+h) - pitch_ref| <= PITCH_TH
    where *_ref is the causal mean of the last 60 s (removes each run's static heel).

Methods compared
    lstm_point    : paper's LSTM point forecast, thresholds checked on the point forecast (Wang et al. 2026)
    tirex_point   : TiRex zero-shot mean forecast, same check
    tirex_raw80   : TiRex 10-90 % quantile band must stay inside the thresholds (no calibration)
    ar_cp         : AR(p, AIC) + joint conformal band (nested leave-one-run-out calibration)
    tirex_cp      : TiRex + joint conformal band  (proposed)
    tirex_cp_aci  : TiRex + joint conformal band + adaptive conformal inference with delayed feedback (proposed)

Joint conformal band: score s = max over leads h<=T and channels {roll, pitch} of |y - centre| / scale,
calibrated on the other three runs. With probability >= 1-alpha the whole roll AND pitch trajectory over
the next T seconds lies inside the band, hence P(declared safe AND actually unsafe) <= alpha.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FS, HORIZON, REPORT_STEPS, RESULTS, RUNS, STRIDE, load_all, load_preds, origins, targets
from run_baselines import ar_forecast, fit_ar

ROLL, PITCH = 1, 2
MAIN = dict(roll_th=1.5, pitch_th=1.0, T=5.0)
ALPHA = 0.10
GAMMA = 0.02          # ACI step size
TIREX = "tirex_ctx300"

data = load_all()
truth = {r: targets(x) for r, x in data.items()}
ref = {r: np.stack([x[i - 300:i].mean(0) for i in origins(len(x))]) for r, x in data.items()}
lstm_pred = load_preds("lstm_paper")[0]
NESTED = None


def set_base(name):
    """(Re)load the probabilistic base forecaster used by the conformal decision rules.
    For trained models a file preds_<name>_nested.npz holds calibration forecasts for run j made by a model
    trained without the test run t (key 't__j'); it is used automatically to avoid calibration leakage."""
    global TIREX, tx_mean, tx_q, levels, i_lo, i_med, i_hi, tx_med, tx_w, NESTED
    TIREX = name
    tx_mean, tx_q, levels = load_preds(name)
    i_lo, i_med, i_hi = 0, int(np.argmin(np.abs(levels - 0.5))), len(levels) - 1
    tx_med = {r: tx_q[r][..., i_med] for r in RUNS}
    tx_w = {r: np.maximum((tx_q[r][..., i_hi] - tx_q[r][..., i_lo]) / 2, 1e-6) for r in RUNS}
    path = os.path.join(RESULTS, f"preds_{name}_nested.npz")
    NESTED = None
    if os.path.exists(path):
        d = np.load(path)
        NESTED = {tuple(k.split("__")): (d[k][..., i_med], np.maximum((d[k][..., i_hi] - d[k][..., i_lo]) / 2, 1e-6))
                  for k in d.files}


def cal_mw(test, j):
    """(centre, half-width) forecasts on calibration run j that are valid for test run `test`."""
    return NESTED[(test, j)] if NESTED is not None else (tx_med[j], tx_w[j])


set_base(TIREX)


def conformal_q(scores, alpha):
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.inf if level > 1 else np.quantile(scores, level, method="higher")


# ---------------- nested AR(p, AIC) predictions for clean calibration ----------------
def ar_aic_preds(train_runs, target_run):
    x = data[target_run]
    o = origins(len(x))
    pred = np.empty((len(o), HORIZON, 3))
    for c in range(3):
        series = [data[r][:, c] for r in train_runs]
        n = sum(len(s) for s in series)
        p = min(range(1, 61), key=lambda k: n * np.log(fit_ar(series, k)[1]) + 2 * k)
        a, _ = fit_ar(series, p)
        pred[:, :, c] = ar_forecast(x[:, c], o, a)
    return pred


cache = os.path.join(RESULTS, "preds_ar_aic_nested.npz")
if os.path.exists(cache):
    d = np.load(cache)
    ar_nested = {(t, j): d[f"{t}__{j}"] for t in RUNS for j in RUNS if t != j}
else:
    ar_nested = {}
    for t in RUNS:
        for j in RUNS:
            if j != t:
                ar_nested[(t, j)] = ar_aic_preds([r for r in RUNS if r not in (t, j)], j)
        print("  nested AR done for test", t, flush=True)
    np.savez_compressed(cache, **{f"{t}__{j}": v for (t, j), v in ar_nested.items()})
ar_test = load_preds("ar_aic")[0]


# ---------------- helpers ----------------
def safe_truth(r, H, rth, pth):
    y, m = truth[r][:, :H], ref[r]
    return (np.all(np.abs(y[..., ROLL] - m[:, None, ROLL]) <= rth, 1)
            & np.all(np.abs(y[..., PITCH] - m[:, None, PITCH]) <= pth, 1))


def inside(lo, hi, m, H, rth, pth):
    """Band [lo, hi] over leads <= H stays inside the threshold box for roll and pitch."""
    ok = np.ones(lo.shape[0], bool)
    for c, th in ((ROLL, rth), (PITCH, pth)):
        ok &= np.all(lo[:, :H, c] >= m[:, None, c] - th, 1) & np.all(hi[:, :H, c] <= m[:, None, c] + th, 1)
    return ok


def joint_scores(y, centre, scale, H):
    s = np.abs(y[:, :H, [ROLL, PITCH]] - centre[:, :H, [ROLL, PITCH]]) / scale[:, :H, [ROLL, PITCH]]
    return s.reshape(len(y), -1).max(1)


def risk_scores(centre, half_width, m, H, rth, pth):
    """How close the forecast band edge gets to the threshold box over the next H leads (<= 1: inside)."""
    dev = np.abs(centre[:, :H] - m[:, None, :]) + half_width[:, :H]
    return np.maximum(dev[..., ROLL] / rth, dev[..., PITCH] / pth).max(1)


def crc_lambda(cal_s, cal_unsafe, alpha):
    """Conformal Risk Control (Angelopoulos et al., ICLR 2024) for the loss 1{declared AND unsafe},
    declared = score <= lambda (loss is monotone in lambda). Largest lambda with (n*R + 1)/(n+1) <= alpha."""
    n = len(cal_s)
    order = np.argsort(cal_s)
    s_sorted, bad = cal_s[order], np.cumsum(cal_unsafe[order])
    ok = (bad + 1) / (n + 1) <= alpha
    ok &= np.append(s_sorted[1:] != s_sorted[:-1], True)  # only cut between distinct scores
    return s_sorted[np.nonzero(ok)[0][-1]] if ok.any() else -np.inf


def decide(method, test, H, rth, pth, alpha=ALPHA):
    """Return (declared_safe bool array, band_covers bool array or None)."""
    m = ref[test]
    y = truth[test]
    others = [r for r in RUNS if r != test]
    if method in ("ar_crc", "tirex_crc", "tirex_crc_aci"):
        if method == "ar_crc":
            zero = lambda p: np.zeros_like(p)
            cal_s = np.concatenate([risk_scores(ar_nested[(test, j)], zero(ar_nested[(test, j)]), ref[j], H, rth, pth)
                                    for j in others])
            s = risk_scores(ar_test[test], zero(ar_test[test]), m, H, rth, pth)
        else:
            cal_s = np.concatenate([risk_scores(*cal_mw(test, j), ref[j], H, rth, pth) for j in others])
            s = risk_scores(tx_med[test], tx_w[test], m, H, rth, pth)
        cal_unsafe = np.concatenate([~safe_truth(j, H, rth, pth) for j in others])
        if method != "tirex_crc_aci":
            return s <= crc_lambda(cal_s, cal_unsafe, alpha), None
        unsafe = ~safe_truth(test, H, rth, pth)
        delay = int(np.ceil(H / STRIDE))
        a_t, D = alpha, np.zeros(len(s), bool)
        for k in range(len(s)):
            j = k - delay
            if j >= 0:
                a_t += GAMMA * (alpha - float(D[j] and unsafe[j]))
            D[k] = s[k] <= crc_lambda(cal_s, cal_unsafe, min(max(a_t, 1e-3), 0.999))
        return D, None
    if method == "lstm_point":
        p = lstm_pred[test]
        return inside(p, p, m, H, rth, pth), None
    if method == "tirex_point":
        p = tx_mean[test]
        return inside(p, p, m, H, rth, pth), None
    if method == "tirex_raw80":
        lo, hi = tx_q[test][..., i_lo], tx_q[test][..., i_hi]
        cov = np.all((y[:, :H, [ROLL, PITCH]] >= lo[:, :H, [ROLL, PITCH]])
                     & (y[:, :H, [ROLL, PITCH]] <= hi[:, :H, [ROLL, PITCH]]), axis=(1, 2))
        return inside(lo, hi, m, H, rth, pth), cov
    if method == "ar_cp":
        cal_res = np.concatenate([truth[j] - ar_nested[(test, j)] for j in others])
        sigma = np.maximum(cal_res.std(0), 1e-6)[None]
        cal = np.concatenate([joint_scores(truth[j], ar_nested[(test, j)],
                                           np.broadcast_to(sigma, truth[j].shape), H) for j in others])
        Q = conformal_q(cal, alpha)
        c, s = ar_test[test], np.broadcast_to(sigma, y.shape)
        cov = joint_scores(y, c, s, H) <= Q
        return inside(c - Q * s, c + Q * s, m, H, rth, pth), cov
    if method in ("tirex_cp", "tirex_cp_aci"):
        cal = np.concatenate([joint_scores(truth[j], *cal_mw(test, j), H) for j in others])
        c, s = tx_med[test], tx_w[test]
        test_scores = joint_scores(y, c, s, H)
        if method == "tirex_cp":
            Q = np.full(len(y), conformal_q(cal, alpha))
        else:
            delay = int(np.ceil(H / STRIDE))  # outcome of origin k is known `delay` origins later
            a_t, Q = alpha, np.empty(len(y))
            for k in range(len(y)):
                j = k - delay
                if j >= 0:
                    a_t += GAMMA * (alpha - float(test_scores[j] > Q[j]))
                Q[k] = conformal_q(cal, min(max(a_t, 1e-3), 0.999))
        Qb = Q[:, None, None]
        return inside(c - Qb * s, c + Qb * s, m, H, rth, pth), test_scores <= Q
    raise ValueError(method)


METHODS = ["lstm_point", "tirex_point", "tirex_raw80", "ar_cp", "tirex_cp", "tirex_cp_aci",
           "ar_crc", "tirex_crc", "tirex_crc_aci"]
LABELS = {"lstm_point": "LSTM của bài (điểm)", "tirex_point": "TiRex (điểm)",
          "tirex_raw80": "TiRex dải 10–90 % (chưa hiệu chỉnh)", "ar_cp": "AR(AIC) + dải conformal chung",
          "tirex_cp": "TiRex + dải conformal chung", "tirex_cp_aci": "TiRex + dải conformal chung + ACI",
          "ar_crc": "AR(AIC) + Conformal Risk Control",
          "tirex_crc": "TiRex + Conformal Risk Control (đề xuất)",
          "tirex_crc_aci": "TiRex + CRC + ACI trực tuyến (đề xuất)"}
ALPHA_CRC = 0.05  # bound on P(declared safe AND actually unsafe)


def alpha_for(method):
    return ALPHA_CRC if "crc" in method else ALPHA


def evaluate(method, rth, pth, T, alpha=None):
    alpha = alpha_for(method) if alpha is None else alpha
    H = int(round(T * FS))
    rows = {}
    D_all, S_all, C_all = [], [], []
    for r in RUNS:
        D, cov = decide(method, r, H, rth, pth, alpha)
        S = safe_truth(r, H, rth, pth)
        D_all.append(D); S_all.append(S)
        if cov is not None:
            C_all.append(cov)
        rows[r] = dict(declared=D.mean(), violation=(D & ~S).mean(),
                       fsr=(D & ~S).sum() / max(D.sum(), 1), recall=(D & S).sum() / max(S.sum(), 1))
    D, S = np.concatenate(D_all), np.concatenate(S_all)
    pooled = dict(declared=D.mean(), violation=(D & ~S).mean(), fsr=(D & ~S).sum() / max(D.sum(), 1),
                  recall=(D & S).sum() / max(S.sum(), 1), base=S.mean(), n=len(D),
                  n_declared=int(D.sum()), n_false_safe=int((D & ~S).sum()),
                  joint_coverage=float(np.concatenate(C_all).mean()) if C_all else None)
    return pooled, rows


