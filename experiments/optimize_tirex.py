"""Optimising the TiRex-based forecaster with techniques from recent A* papers (frozen backbone, cached features).

  ft          : head-only fine-tuning, pinball loss (current model, reproduced here)
  ft_fredf    : + FreDF frequency-domain loss on the median forecast (ICLR 2025, "Label Correlation Biases
                Direct Time Series Forecast")
  mv          : + cross-channel adapter: per-channel zero-initialised linear map from the concatenated
                heave/roll/pitch embeddings (multivariate adaptation of a univariate foundation model, in the
                spirit of AdaPTS, ICML 2025)
  mv_fredf    : both
Selection uses ONLY validation data (last 20 % of every training run). Test runs are never used for choices.

Usage: python optimize_tirex.py select   # train all variants on every fold, report validation + test
       python optimize_tirex.py nested   # nested calibration forecasts for the selected variant
"""
import copy
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from tirex import load_model

from common import HORIZON, MAX_CONTEXT, RESULTS, RUNS, STRIDE, load_all, origins, save_preds
from finetune_tirex import CACHE, Head, all_origins

torch.set_num_threads(4)
data = load_all()
model = load_model("NX-AI/TiRex")
LEVELS = torch.tensor(model.config.quantiles)
cache = np.load(CACHE)
VARIANTS = {"ft": (False, 0.0), "ft_fredf": (False, 0.5), "mv": (True, 0.0), "mv_fredf": (True, 0.5)}


def run_arrays(r):
    """Multichannel arrays over all stride-1 origins of run r."""
    x = data[r]
    o = all_origins(len(x))
    H = np.stack([cache[f"h_{r}_{c}"] for c in range(3)], 1)            # (n, 3, 4, D)
    L = np.stack([cache[f"loc_{r}_{c}"] for c in range(3)], 1)          # (n, 3)
    S = np.stack([cache[f"sc_{r}_{c}"] for c in range(3)], 1)
    Y = np.stack([x[i:i + HORIZON] for i in o]).transpose(0, 2, 1)       # (n, 3, H)
    return o, H, L, S, Y


ARR = {r: run_arrays(r) for r in RUNS}


class MVHead(nn.Module):
    def __init__(self, multivariate):
        super().__init__()
        self.head = Head(model)
        D = model.output_patch_embedding.hidden_layer.in_features if hasattr(model.output_patch_embedding, "hidden_layer") else 512
        self.mv = multivariate
        if multivariate:
            self.adapt = nn.ModuleList([nn.Linear(3 * D, D) for _ in range(3)])
            for lin in self.adapt:
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, h):  # h: (B, 3, 4, D) -> normalised quantiles (B, 3, H, 9)
        if self.mv:
            joint = h.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[2], -1)   # (B, 4, 3D)
            h = torch.stack([h[:, c] + self.adapt[c](joint) for c in range(3)], 1)
        return torch.stack([self.head(h[:, c]) for c in range(3)], 1)


def losses(q, yn, lam):
    e = yn[..., None] - q
    pin = torch.mean(torch.maximum(LEVELS * e, (LEVELS - 1) * e))
    if lam == 0:
        return pin, pin
    med = q[..., q.shape[-1] // 2]
    freq = torch.mean(torch.abs(torch.fft.rfft(med - yn, dim=-1))) / np.sqrt(yn.shape[-1])  # FreDF
    return (1 - lam) * pin + lam * freq, pin


def split(train_runs):
    tr, va = [], []
    for r in train_runs:
        o, H, L, S, Y = ARR[r]
        cut = int(0.8 * len(o))
        tr.append(tuple(a[: cut - HORIZON] for a in (H, L, S, Y)))
        va.append(tuple(a[cut:] for a in (H, L, S, Y)))
    cat = lambda parts: [torch.tensor(np.concatenate([p[k] for p in parts])) for k in range(4)]
    return cat(tr), cat(va)


def fit(train_runs, multivariate, lam, epochs=200, lr=2e-4, wd=1e-4, patience=10, seed=0):
    torch.manual_seed(seed)
    (Htr, Ltr, Str, Ytr), (Hva, Lva, Sva, Yva) = split(train_runs)
    ytr = (Ytr - Ltr[..., None]) / Str[..., None]
    yva = (Yva - Lva[..., None]) / Sva[..., None]
    net = MVHead(multivariate)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    def val():
        net.eval()
        with torch.no_grad():
            q = net(Hva)
            pin = losses(q, yva, 0)[1].item()
            med = q[..., q.shape[-1] // 2]
            mse5 = torch.mean((med[..., :25] - yva[..., :25]) ** 2).item()
        return pin, mse5

    best, state, bad = val()[0], copy.deepcopy(net.state_dict()), 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Htr))
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]
            opt.zero_grad()
            losses(net(Htr[i]), ytr[i], lam)[0].backward()
            opt.step()
        v = val()[0]
        if v < best - 1e-5:
            best, state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, val()


def predict(net, run):
    o, H, L, S, _ = ARR[run]
    keep = (o - MAX_CONTEXT) % STRIDE == 0
    with torch.no_grad():
        q = net(torch.tensor(H[keep]))                                         # (n, 3, H, 9)
        q = q * torch.tensor(S[keep])[..., None, None] + torch.tensor(L[keep])[..., None, None]
        q = torch.sort(q, -1).values
    return q.numpy().transpose(0, 2, 1, 3)                                     # (n, H, 3, 9)


def test_scores(qd):
    """Mean over runs of: nRMSE at 1/2/5 s per channel, normalised pinball (CRPS proxy) over 0-5 s."""
    out = {}
    for r in RUNS:
        x = data[r]; o = origins(len(x)); y = np.stack([x[i:i + HORIZON] for i in o]); sd = x.std(0)
        med = qd[r][..., qd[r].shape[-1] // 2]
        e = (y[..., None] - qd[r]) / sd[None, None, :, None]
        lv = LEVELS.numpy()
        pin = np.mean(np.maximum(lv * e, (lv - 1) * e)[:, :25], axis=(0, 1, 3))
        out[r] = {"nrmse": {s: np.sqrt(np.mean((med[:, k - 1] - y[:, k - 1]) ** 2, 0)) / sd for s, k in (("1s", 5), ("2s", 10), ("5s", 25))},
                  "pinball5": pin}
    agg = {s: np.mean([out[r]["nrmse"][s] for r in RUNS], 0) for s in ("1s", "2s", "5s")}
    agg["pinball5"] = np.mean([out[r]["pinball5"] for r in RUNS], 0)
    return agg


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "select"
    if step == "select":
        report = {}
        for name, (mv, lam) in VARIANTS.items():
            preds, quants, vals = {}, {}, []
            for test in RUNS:
                net, (vpin, vmse5) = fit([r for r in RUNS if r != test], mv, lam)
                vals.append((vpin, vmse5))
                quants[test] = predict(net, test)
                preds[test] = quants[test][..., 4]
            save_preds(f"opt_{name}", preds, quants, LEVELS.numpy())
            ts = test_scores(quants)
            report[name] = {"val_pinball": float(np.mean([v[0] for v in vals])), "val_mse5": float(np.mean([v[1] for v in vals])),
                            "test": {k: v.tolist() for k, v in ts.items()}}
            print(f"{name:9s} | VAL pinball {report[name]['val_pinball']:.4f} mse5 {report[name]['val_mse5']:.4f} | "
                  f"TEST nRMSE 1s {np.round(ts['1s'], 3)} 2s {np.round(ts['2s'], 3)} 5s {np.round(ts['5s'], 3)} "
                  f"pinball0-5s {np.round(ts['pinball5'], 4)}", flush=True)
        best = min(report, key=lambda k: report[k]["val_pinball"])
        report["selected_by_validation"] = best
        print("selected by validation pinball:", best)
        with open(os.path.join(RESULTS, "optimize_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    if step == "nested":
        best = json.load(open(os.path.join(RESULTS, "optimize_report.json")))["selected_by_validation"]
        mv, lam = VARIANTS[best]
        out = {}
        for t in RUNS:
            for j in RUNS:
                if j != t:
                    net, _ = fit([r for r in RUNS if r not in (t, j)], mv, lam)
                    out[f"{t}__{j}"] = predict(net, j)
                    print(f"  nested {best} test={t} cal={j}", flush=True)
        np.savez_compressed(os.path.join(RESULTS, f"preds_opt_{best}_nested.npz"), **out)
