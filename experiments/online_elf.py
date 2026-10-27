"""Online adaptation of foundation-model forecasts (in the spirit of ELF, ICML 2025: a lightweight
forecaster learned from data + an online weighter that combines it with the foundation model).

  * lightweight forecaster: direct multi-horizon ridge regression per channel (DLinear-style) from the last
    12 s (60 samples, de-meaned by the causal 60 s context mean) to the next 20 s; ridge strength chosen on a
    temporal validation split of the training runs
  * online weighter: exponentially weighted (Hedge) combination per lead and channel, updated only with
    outcomes that are already observed (delayed feedback), so it is fully causal
  * the combined median shifts the base quantile band (band widths are left to the conformal layer)

Produces preds_<out>.npz (leave-one-run-out) and preds_<out>_nested.npz (calibration forecasts that never use
the test run) from a base forecaster that has both files.
Usage: python online_elf.py <base_name> <out_name> [eta]
"""
import os
import sys

import numpy as np

from common import FS, HORIZON, MAX_CONTEXT, RESULTS, RUNS, STRIDE, load_all, load_preds, origins, save_preds

CTXS = (30, 60, 120, 200)        # history length in samples (6-40 s), chosen on validation
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
data = load_all()


def lin_features(x, idx, ctx):
    """(n, ctx) de-meaned recent history and (n,) causal context means for one channel."""
    m = np.array([x[i - MAX_CONTEXT:i].mean() for i in idx])
    X = np.stack([x[i - ctx:i] for i in idx]) - m[:, None]
    return X, m


def fit_ridge(train_runs, c):
    """Direct multi-horizon ridge for channel c; history length and alpha by temporal validation
    (last 20 % of each training run). Returns ((W, ctx), (ctx, alpha))."""
    def build(runs, part, ctx):
        Xs, Ys = [], []
        for r in runs:
            x = data[r][:, c]
            idx = np.arange(MAX_CONTEXT, len(x) - HORIZON + 1)
            cut = int(0.8 * len(idx))
            sel = idx[: cut - HORIZON] if part == "tr" else idx[cut:] if part == "va" else idx
            X, m = lin_features(x, sel, ctx)
            Xs.append(X); Ys.append(np.stack([x[i:i + HORIZON] for i in sel]) - m[:, None])
        return np.concatenate(Xs), np.concatenate(Ys)

    def solve(X, Y, a):
        s = np.mean(np.sum(X ** 2, 1))
        return np.linalg.solve(X.T @ X + a * s * np.eye(X.shape[1]), X.T @ Y)

    scores = {}
    for ctx in CTXS:
        Xtr, Ytr = build(train_runs, "tr", ctx)
        Xva, Yva = build(train_runs, "va", ctx)
        for a in ALPHAS:
            scores[(ctx, a)] = np.mean((Xva @ solve(Xtr, Ytr, a) - Yva)[:, :25] ** 2)
    ctx, a = min(scores, key=scores.get)
    Xall, Yall = build(train_runs, "all", ctx)
    return (solve(Xall, Yall, a), ctx), (ctx, a)


def lin_forecast(Wc, run, c):
    W, ctx = Wc
    x = data[run][:, c]
    X, m = lin_features(x, origins(len(x)), ctx)
    return X @ W + m[:, None]                                                 # (n_o, H)


def hedge_combine(base_med, lin, y, sd, eta):
    """Per (lead, channel) online Hedge between base median and linear forecast, delayed feedback."""
    n, H, C = base_med.shape
    logw = np.zeros((H, C, 2))
    out = np.empty_like(base_med)
    lead_delay = np.ceil(np.arange(1, H + 1) / STRIDE).astype(int)            # origins until lead h is observed
    for k in range(n):
        for h in range(H):
            j = k - lead_delay[h]
            if j >= 0:  # outcome of origin j at lead h has just become available
                e = np.stack([(base_med[j, h] - y[j, h]) ** 2, (lin[j, h] - y[j, h]) ** 2], -1) / sd[:, None] ** 2
                logw[h] -= eta * e
                logw[h] -= logw[h].max(-1, keepdims=True)
        w = np.exp(logw); w /= w.sum(-1, keepdims=True)
        out[k] = w[..., 0] * base_med[k] + w[..., 1] * lin[k]
    return out


def apply(base_q, lin, run, eta):
    x = data[run]
    o = origins(len(x))
    y = np.stack([x[i:i + HORIZON] for i in o])
    sd = np.array([x[:MAX_CONTEXT, c].std() for c in range(3)])  # scale from the first 60 s only (causal)
    med = base_q[..., base_q.shape[-1] // 2]
    comb = hedge_combine(med, lin, y, sd, eta)
    return base_q + (comb - med)[..., None]


if __name__ == "__main__":
    base, out = sys.argv[1], sys.argv[2]
    eta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    _, bq, levels = load_preds(base)
    nested = np.load(os.path.join(RESULTS, f"preds_{base}_nested.npz"))
    preds, quants, nest_out, alphas = {}, {}, {}, []
    for t in RUNS:
        W = [fit_ridge([r for r in RUNS if r != t], c) for c in range(3)]
        alphas += [a for _, a in W]
        lin = np.stack([lin_forecast(W[c][0], t, c) for c in range(3)], -1)
        quants[t] = apply(bq[t], lin, t, eta)
        preds[t] = quants[t][..., quants[t].shape[-1] // 2]
        for j in RUNS:
            if j != t:
                Wn = [fit_ridge([r for r in RUNS if r not in (t, j)], c)[0] for c in range(3)]
                lin_j = np.stack([lin_forecast(Wn[c], j, c) for c in range(3)], -1)
                nest_out[f"{t}__{j}"] = apply(nested[f"{t}__{j}"], lin_j, j, eta)
        print(f"  {t} done", flush=True)
    save_preds(out, preds, quants, levels)
    np.savez_compressed(os.path.join(RESULTS, f"preds_{out}_nested.npz"), **nest_out)
    print("ridge (history, alpha) chosen on validation:", sorted(set(alphas)))
