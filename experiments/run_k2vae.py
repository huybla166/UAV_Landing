"""K²VAE (ICML 2025 Spotlight; official model code vendored in ./k2vae) on the NTUA data, leave-one-run-out.

Multivariate model over (heave, roll, pitch). Training follows the official k2VAEModel.loss():
    loss = MSE(rec, input) + MSE(posterior mean, target) + 1{NLL<100} * NLL(target) + beta * KL(prior)
Default short-term hyper-parameters from the official config (config/stsf/*/k2vae.yaml); inputs are
standardised per channel with training-run statistics (as the official DataManager does), the model also
applies RevIN internally. 100 samples per forecast -> 9 empirical quantiles + median.
"""
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import HORIZON, MAX_CONTEXT, RESULTS, RUNS, load_all, origins, save_preds
from k2vae.k2vae import k2VAE

torch.set_num_threads(4)
LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 1)


class Cfg:
    d_model, d_ff, e_layers, dropout, activation = 128, 256, 2, 0.1, "relu"
    n_heads, factor, patch_len, multistep = 8, 3, 8, True
    dynamic_dim, hidden_layers, hidden_dim = 128, 2, 256
    sample_schedule, init_kalman, init_koopman = 20, "randn", "dynamic"
    output_attention = False
    # official code needs seq_len divisible by patch_len: 296 samples = 59.2 s (<= the 60 s used elsewhere)
    # ... and pred_len divisible by patch_len: train on 104 steps, keep the first 100 (20 s)
    n_vars, seq_len, pred_len = 3, 296, 104


WEIGHT_BETA, LR, CLIP, BATCH = 0.01, 1e-3, 0.5, 32


def kld_loss(dist):
    mu, cov = dist.loc, dist.covariance_matrix
    log_det = torch.linalg.slogdet(cov)[1]
    trace = torch.einsum("btii->bt", cov)
    return (0.5 * (trace + (mu ** 2).sum(-1) - mu.size(-1) - log_det)).mean()


def official_loss(model, x, y, threshold=1e2):
    rec, prior, post = model(x)
    rec_loss = F.mse_loss(rec, x) + F.mse_loss(post.loc, y)
    post_loss = -post.log_prob(y).mean()
    alpha = 1.0 if post_loss.item() < threshold else 0.0
    return rec_loss + alpha * post_loss + WEIGHT_BETA * kld_loss(prior)


def make_windows(x, idx):
    X = np.stack([x[i - Cfg.seq_len:i] for i in idx]).astype(np.float32)
    Y = np.stack([x[i:i + Cfg.pred_len] for i in idx]).astype(np.float32) if idx[-1] + Cfg.pred_len <= len(x) else None
    return torch.tensor(X), (torch.tensor(Y) if Y is not None else None)


def run_fold(test, data, max_epochs=50, patience=6, steps_per_epoch=100, seed=1, log=print, val_metric="official"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_runs = [r for r in RUNS if r != test]
    mu = np.concatenate([data[r] for r in train_runs]).mean(0)
    sd = np.concatenate([data[r] for r in train_runs]).std(0)
    norm = {r: (data[r] - mu) / sd for r in RUNS}
    Xtr, Ytr, Xva, Yva = [], [], [], []
    for r in train_runs:
        idx = np.arange(MAX_CONTEXT, len(norm[r]) - Cfg.pred_len + 1)
        cut = int(0.8 * len(idx))
        a, b = make_windows(norm[r], idx[: cut - HORIZON]); Xtr.append(a); Ytr.append(b)
        a, b = make_windows(norm[r], idx[cut:]); Xva.append(a); Yva.append(b)
    Xtr, Ytr, Xva, Yva = map(torch.cat, (Xtr, Ytr, Xva, Yva))

    model = k2VAE(Cfg())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        for _ in range(steps_per_epoch):  # official config: limit_train_batches = 100
            i = torch.randint(0, len(Xtr), (BATCH,))
            opt.zero_grad()
            loss = official_loss(model, Xtr[i], Ytr[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():  # validation: CRPS-like proxy = MSE of the posterior mean + NLL
            vs = []
            for b in range(0, len(Xva), 128):
                _, _, post = model(Xva[b:b + 128])
                if val_metric == "short":  # decision-relevant: MSE of the forecast over the first 5 s (25 steps)
                    vs.append(F.mse_loss(post.loc[:, :25], Yva[b:b + 128, :25]).item())
                else:
                    vs.append((F.mse_loss(post.loc, Yva[b:b + 128]) - post.log_prob(Yva[b:b + 128]).mean()).item())
            v = float(np.mean(vs))
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        log(f"    {test} epoch {ep + 1}: val {v:.4f} (best {best:.4f}) [{time.time() - t0:.0f}s]")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()

    o = origins(len(data[test]))
    Xte, _ = make_windows(norm[test], o)
    qs = []
    with torch.no_grad():
        for b in range(0, len(Xte), 64):
            s = model.sample(Xte[b:b + 64], 100)[:, :, :HORIZON]  # (100, B, H, 3) in normalised units
            qs.append(torch.quantile(s, torch.tensor(LEVELS, dtype=torch.float32), dim=0).permute(1, 2, 3, 0))
    q = torch.cat(qs).numpy() * sd[None, None, :, None] + mu[None, None, :, None]   # (n_o, H, 3, 9)
    return q[..., 4], q


if __name__ == "__main__":
    data = load_all()
    if len(sys.argv) > 1 and sys.argv[1] == "timing":
        m = k2VAE(Cfg())
        print("params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
        x, y = torch.randn(BATCH, Cfg.seq_len, 3), torch.randn(BATCH, Cfg.pred_len, 3)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        t0 = time.time()
        for _ in range(5):
            opt.zero_grad(); official_loss(m, x, y).backward(); opt.step()
        print("sec per training step: %.2f" % ((time.time() - t0) / 5))
        sys.exit()
    variant = sys.argv[1] if len(sys.argv) > 1 else "official"
    if variant in ("small", "small_short"):
        # same Koopman-Kalman VAE architecture, ~30x fewer parameters for a 20-minute dataset
        Cfg.d_model, Cfg.d_ff, Cfg.n_heads, Cfg.e_layers = 32, 64, 4, 1
        Cfg.dynamic_dim, Cfg.hidden_dim, Cfg.hidden_layers = 32, 64, 2
        LR = 5e-4
    name = "k2vae" if variant == "official" else f"k2vae_{variant}"
    logf = open(os.path.join(RESULTS, f"log_{name}.txt"), "w", encoding="utf-8")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    preds, quants = {}, {}
    for test in RUNS:
        # reduced CPU budget vs official (50 epochs x 100 steps): 24 x 50 steps, early stopping on validation
        if variant == "small_short":
            preds[test], quants[test] = run_fold(test, data, max_epochs=60, patience=10, steps_per_epoch=50,
                                                 log=log, val_metric="short")
        else:
            preds[test], quants[test] = run_fold(test, data, max_epochs=30 if variant == "small" else 24,
                                                 patience=6 if variant == "small" else 5, steps_per_epoch=50, log=log)
        log(f"  fold {test} done")
    save_preds(name, preds, quants, LEVELS)
    log(f"{name} done")
