"""Baselines on the NTUA data under leave-one-run-out:
  mean        - mean of the last 60 s (what a "do nothing" predictor gives for zero-mean waves)
  persistence - last observed value held constant
  ar5         - AR(5) by Yule-Walker, as the comparison model in Wang et al. (2026)
  ar_aic      - AR(p), p <= 60 chosen by AIC (a stronger linear baseline)
  lstm_paper  - replication of the paper's LSTM: 1 layer, 50 units, 10-step input, dropout 0.1,
                Adam lr 0.005 with piecewise decay; one-step training, recursive multi-step rollout
"""
import numpy as np
import torch
from torch import nn

from common import HORIZON, RUNS, load_all, origins, save_preds

data = load_all()


# ---------------- naive ----------------
def naive(kind):
    out = {}
    for r, x in data.items():
        o = origins(len(x))
        if kind == "mean":
            v = np.stack([x[i - 300:i].mean(0) for i in o])
        else:
            v = np.stack([x[i - 1] for i in o])
        out[r] = np.repeat(v[:, None, :], HORIZON, axis=1)
    return out


# ---------------- AR ----------------
def yule_walker(x, p):
    x = x - x.mean()
    n = len(x)
    r = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(p + 1)])
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    a = np.linalg.solve(R, r[1:])
    sigma2 = r[0] - a @ r[1:]
    return a, sigma2


def fit_ar(train_series, p):
    """Pool several runs: concatenate autocovariances weighted by length."""
    xs = [s - s.mean() for s in train_series]
    n = sum(len(s) for s in xs)
    r = np.array([sum(np.dot(s[: len(s) - k], s[k:]) for s in xs) / n for k in range(p + 1)])
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    a = np.linalg.solve(R, r[1:])
    return a, r[0] - a @ r[1:]


def ar_forecast(x, o_idx, a):
    """Recursive AR forecast around the causal context mean (last 60 s): each run has its own static
    heel, so a pooled training mean would inject a constant bias."""
    p = len(a)
    out = np.empty((len(o_idx), HORIZON))
    for k, o in enumerate(o_idx):
        mu = x[o - 300:o].mean()
        hist = list(x[o - p:o] - mu)
        for h in range(HORIZON):
            nxt = float(np.dot(a, hist[::-1][:p]))
            out[k, h] = nxt + mu
            hist.append(nxt)
    return out


def run_ar(order):
    out = {}
    chosen = {}
    for test in RUNS:
        train = [data[r] for r in RUNS if r != test]
        x = data[test]
        o = origins(len(x))
        pred = np.empty((len(o), HORIZON, 3))
        for c in range(3):
            series = [s[:, c] for s in train]
            if order == "aic":
                n = sum(len(s) for s in series)
                best = min(range(1, 61), key=lambda p: n * np.log(fit_ar(series, p)[1]) + 2 * p)
                p = best
            else:
                p = order
            a, _ = fit_ar(series, p)
            chosen[(test, c)] = p
            pred[:, :, c] = ar_forecast(x[:, c], o, a)
        out[test] = pred
    return out, chosen


# ---------------- paper LSTM ----------------
class PaperLSTM(nn.Module):
    def __init__(self, n_in=3, hidden=50, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_in)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(self.drop(h[:, -1]))


def windows(series, L):
    X, Y = [], []
    for s in series:
        for i in range(L, len(s)):
            X.append(s[i - L:i])
            Y.append(s[i])
    return np.array(X, np.float32), np.array(Y, np.float32)


def run_lstm(L=10, epochs=60, seeds=(0, 1, 2), instance_norm=False):
    """instance_norm=False: global train mean/std, as a literal replication of the paper.
    instance_norm=True: subtract each run's own static offset (test: causal 60 s context mean)."""
    out = {}
    for test in RUNS:
        train = [data[r] for r in RUNS if r != test]
        sd = np.concatenate(train).std(0)
        if instance_norm:
            norm = [(s - s.mean(0)) / sd for s in train]
        else:
            mu = np.concatenate(train).mean(0)
            norm = [(s - mu) / sd for s in train]
        X, Y = windows(norm, L)
        X, Y = torch.tensor(X), torch.tensor(Y)
        o = origins(len(data[test]))
        if instance_norm:
            off = np.stack([data[test][i - 300:i].mean(0) for i in o])  # (n_origins, 3)
            x = data[test] / sd
        else:
            off = np.broadcast_to(mu, (len(o), 3))
            x = (data[test] - mu) / sd
        preds = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PaperLSTM()
            opt = torch.optim.Adam(model.parameters(), lr=0.005)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.2)  # piecewise decay
            for _ in range(epochs):
                model.train()
                perm = torch.randperm(len(X))
                for b in range(0, len(X), 64):
                    idx = perm[b:b + 64]
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
                    loss.backward()
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                if instance_norm:
                    ctx = np.stack([x[i - L:i] - off[k] / sd for k, i in enumerate(o)])
                else:
                    ctx = np.stack([x[i - L:i] for i in o])
                hist = torch.tensor(ctx, dtype=torch.float32)
                steps = []
                for _ in range(HORIZON):
                    nxt = model(hist)
                    steps.append(nxt)
                    hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
                preds.append(torch.stack(steps, 1).numpy())
        p = np.mean(preds, axis=0) * sd
        out[test] = p + off[:, None, :] if instance_norm else p + mu
        print(f"  lstm fold {test} done (instance_norm={instance_norm})")
    return out


if __name__ == "__main__":
    import sys
    torch.set_num_threads(4)
    only = set(sys.argv[1:])
    if not only or "naive" in only:
        save_preds("mean", naive("mean"))
        save_preds("persistence", naive("persistence"))
    if not only or "ar" in only:
        p5, _ = run_ar(5)
        save_preds("ar5", p5)
        paic, chosen = run_ar("aic")
        save_preds("ar_aic", paic)
        print("AR orders chosen by AIC:", sorted(set(chosen.values())))
    if not only or "lstm" in only:
        save_preds("lstm_paper", run_lstm())
    if not only or "lstm_in" in only:
        save_preds("lstm_paper_in", run_lstm(instance_norm=True))
    print("baselines done")
