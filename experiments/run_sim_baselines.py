"""Stage 3.1 baselines on the simulated dataset (experiments/sim_common.py protocol).

Models (same definitions as run_baselines.py on NTUA):
  mean        - mean of the last 60 s held constant
  persistence - last observed value held constant
  ar5         - AR(5) by Yule-Walker (comparison model in Wang et al. 2026), recursive, around the
                causal 60 s context mean
  ar_aic      - AR(p), p <= 60 by AIC
  lstm_paper  - the paper's LSTM: 1 layer, 50 units, 10-step input, dropout 0.1, Adam lr 0.005 with
                piecewise (step) decay, one-step training, recursive 100-step rollout, global train
                mean/std, mean of 3 seeds.
                Scaled for ~4 M training windows instead of a few thousand: batch 1024 (paper: 64),
                12 epochs with lr x0.2 every 4 (paper: 60 epochs). Everything else as the paper.
Fit on split 'train' (1440 runs). Scored on test_id, test_ood_Hs, test_ood_beta (all origins).

  D:/tools/venv-gpu/Scripts/python.exe experiments/run_sim_baselines.py [naive] [ar] [lstm]
Writes results/sim/baselines_metrics.npz (per-run nRMSE per lead step), results/sim/baselines.md.
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

from sim_common import CHANNELS, FS, HORIZON, MAX_CONTEXT, REPORT_STEPS, RESULTS_SIM, load, motion, origins

TEST_SPLITS = ["test_id", "test_ood_Hs", "test_ood_beta"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = os.path.join(RESULTS_SIM, "baselines_metrics.npz")


def load_motion(split):
    meta, X = load(split)
    return meta, motion(np.asarray(X))                 # (n_runs, T, 3)


def score(pred_fn, Y):
    """pred_fn(Y, o) -> (n_runs, n_origins, HORIZON, 3) for origin indices o. Returns per-run nRMSE
    (n_runs, HORIZON, 3), scale = std of each run's own series (as evaluate.py)."""
    o = origins(Y.shape[1])
    tgt = np.stack([Y[:, i:i + HORIZON] for i in o], axis=1)          # (R, O, H, 3)
    pred = pred_fn(Y, o)
    rmse = np.sqrt(np.mean((pred - tgt) ** 2, axis=1))                # (R, H, 3)
    return rmse / Y.std(axis=1)[:, None, :]


# ---------------------------------------------------------------- naive
def pred_mean(Y, o):
    v = np.stack([Y[:, i - MAX_CONTEXT:i].mean(1) for i in o], axis=1)
    return np.repeat(v[:, :, None, :], HORIZON, axis=2)


def pred_persistence(Y, o):
    v = np.stack([Y[:, i - 1] for i in o], axis=1)
    return np.repeat(v[:, :, None, :], HORIZON, axis=2)


# ---------------------------------------------------------------- AR
def autocov(series, pmax):
    """Pooled autocovariance r[0..pmax] of a list of runs, each centred on its own mean (as fit_ar)."""
    xs = series - series.mean(axis=1, keepdims=True)                  # (R, T)
    n = xs.size
    return np.array([np.sum(xs[:, :xs.shape[1] - k] * xs[:, k:]) / n for k in range(pmax + 1)])


def yw(r, p):
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    a = np.linalg.solve(R, r[1:p + 1])
    return a, r[0] - a @ r[1:p + 1]


def ar_rollout(Y, o, coefs):
    """Recursive AR per channel around the causal 60 s context mean, vectorised over runs and origins."""
    R, O = Y.shape[0], len(o)
    out = np.empty((R, O, HORIZON, 3), np.float32)
    for c in range(3):
        a = coefs[c]; p = len(a)
        mu = np.stack([Y[:, i - MAX_CONTEXT:i, c].mean(1) for i in o], axis=1)          # (R, O)
        hist = np.stack([Y[:, i - p:i, c] for i in o], axis=1) - mu[..., None]        # (R, O, p) oldest..newest
        ar = a[::-1].astype(np.float64)                                                # weights oldest..newest
        for h in range(HORIZON):
            nxt = hist[..., -p:] @ ar
            out[:, :, h, c] = nxt + mu
            hist = np.concatenate([hist[..., 1:], nxt[..., None]], axis=-1)
    return out


def fit_ar(Ytr, order):
    coefs, chosen = [], []
    for c in range(3):
        r = autocov(Ytr[:, :, c].astype(np.float64), 60)
        n = Ytr.shape[0] * Ytr.shape[1]
        p = min(range(1, 61), key=lambda q: n * np.log(yw(r, q)[1]) + 2 * q) if order == "aic" else order
        coefs.append(yw(r, p)[0]); chosen.append(p)
    return coefs, chosen


# ---------------------------------------------------------------- paper LSTM
class PaperLSTM(nn.Module):
    def __init__(self, n_in=3, hidden=50, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_in)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(self.drop(h[:, -1]))


def train_lstm(Ytr, seed, L=10, epochs=12, batch=1024, step=4):
    mu, sd = Ytr.reshape(-1, 3).mean(0), Ytr.reshape(-1, 3).std(0)
    Z = torch.tensor((Ytr - mu) / sd, dtype=torch.float32, device=DEV)            # (R, T, 3)
    R, T, _ = Z.shape
    starts = torch.arange(T - L, device=DEV)
    n = R * len(starts)
    torch.manual_seed(seed); np.random.seed(seed)
    model = PaperLSTM().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.2)
    ar = torch.arange(L, device=DEV)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, batch):
            k = perm[b:b + batch]
            ri, si = k // len(starts), starts[k % len(starts)]
            x = Z[ri[:, None], si[:, None] + ar]                                    # (B, L, 3)
            y = Z[ri, si + L]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward(); opt.step(); tot += loss.item() * len(k)
        sched.step()
        print(f"    seed {seed} epoch {ep + 1}/{epochs}: train mse {tot / n:.5f} ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    return model, mu, sd


def lstm_predictor(models, L=10, chunk=60000):
    def pred(Y, o):
        R, O = Y.shape[0], len(o)
        ctx = np.stack([Y[:, i - L:i] for i in o], axis=1).reshape(R * O, L, 3)
        out = np.zeros((R * O, HORIZON, 3), np.float32)
        for model, mu, sd in models:
            for b in range(0, len(ctx), chunk):
                hist = torch.tensor((ctx[b:b + chunk] - mu) / sd, dtype=torch.float32, device=DEV)
                steps = []
                with torch.no_grad():
                    for _ in range(HORIZON):
                        nxt = model(hist)
                        steps.append(nxt)
                        hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
                out[b:b + chunk] += (torch.stack(steps, 1).cpu().numpy() * sd + mu).astype(np.float32)
        return (out / len(models)).reshape(R, O, HORIZON, 3)
    return pred


# ---------------------------------------------------------------- main
def main(which):
    res = dict(np.load(METRICS)) if os.path.exists(METRICS) else {}
    meta_tr, Ytr = load_motion("train")
    tests = {s: load_motion(s) for s in TEST_SPLITS}
    for s, (m, Y) in tests.items():
        res[f"meta_{s}_ids"] = m.id.to_numpy()

    def run(name, fn):
        for s, (m, Y) in tests.items():
            t0 = time.time()
            res[f"{name}|{s}"] = score(fn, Y).astype(np.float32)
            print(f"  {name} on {s}: nRMSE mean 0-20 s = {res[f'{name}|{s}'].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        np.savez_compressed(METRICS, **res)

    if "naive" in which:
        run("mean", pred_mean)
        run("persistence", pred_persistence)
    if "ar" in which:
        for order in (5, "aic"):
            coefs, chosen = fit_ar(Ytr, order)
            name = "ar5" if order == 5 else "ar_aic"
            print(f"  {name}: orders {chosen}")
            res[f"{name}|orders"] = np.array(chosen)
            run(name, lambda Y, o, coefs=coefs: ar_rollout(Y, o, coefs))
    if "lstm" in which:
        models = [train_lstm(Ytr, seed) for seed in (0, 1, 2)]
        run("lstm_paper", lstm_predictor(models))


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"naive", "ar", "lstm"}
    print(f"device {DEV}", flush=True)
    main(which)
    print("done")
