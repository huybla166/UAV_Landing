"""Sanity check: K2VAE (small config) on synthetic sum-of-sines data in the style of the paper's USV model (5).
Shows the pipeline works when the signal is a few sinusoids, unlike the real NTUA data.
Writes results/k2vae_synthetic.json with nRMSE per lead (3 channels)."""
import json
import os

import numpy as np
import torch

import run_k2vae as rk
from common import RESULTS

torch.manual_seed(0)
rng = np.random.default_rng(0)
rk.Cfg.d_model, rk.Cfg.d_ff, rk.Cfg.n_heads, rk.Cfg.e_layers = 32, 64, 4, 1
rk.Cfg.dynamic_dim, rk.Cfg.hidden_dim, rk.Cfg.hidden_layers = 32, 64, 2
fs, T = 5.0, 6000
t = np.arange(T) / fs


def series():
    out = []
    for c in range(3):
        per = rng.uniform(6, 12, 4); amp = rng.uniform(0.3, 1, 4); ph = rng.uniform(0, 2 * np.pi, 4)
        out.append(sum(a * np.sin(2 * np.pi * t / p + f) for a, p, f in zip(amp, per, ph)) + 0.05 * rng.standard_normal(T))
    return np.stack(out, 1)


x = series(); x = (x - x.mean(0)) / x.std(0)
idx = np.arange(296, T - 104)
tr, te = idx[: int(0.7 * len(idx))], idx[int(0.7 * len(idx)) + 104:][::5]
X, Y = rk.make_windows(x, tr); Xt, Yt = rk.make_windows(x, te)
m = rk.k2VAE(rk.Cfg()); opt = torch.optim.Adam(m.parameters(), lr=5e-4)
for ep in range(15):
    m.train()
    for _ in range(50):
        i = torch.randint(0, len(X), (32,)); opt.zero_grad(); loss = rk.official_loss(m, X[i], Y[i]); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5); opt.step()
m.eval()
with torch.no_grad():
    s = m.sample(Xt, 20).median(0).values[:, :100]
nr = torch.sqrt(((s - Yt[:, :100]) ** 2).mean(0)) / Yt[:, :100].std((0, 1))
out = {f"{lead / fs:.0f}s" if lead >= 5 else f"{lead / fs:.1f}s": [float(v) for v in nr[lead - 1]] for lead in (5, 10, 25, 50, 100)}
for k, v in out.items():
    print(f"synthetic sum-of-sines: lead {k:>4}  nRMSE (3 ch) = " + ", ".join(f"{u:.2f}" for u in v))
with open(os.path.join(RESULTS, "k2vae_synthetic.json"), "w", encoding="utf-8") as fh:
    json.dump({"n_params": sum(p.numel() for p in m.parameters()), "nrmse_by_lead": out}, fh, indent=1)
