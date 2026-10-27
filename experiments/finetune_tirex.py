"""Fine-tuning TiRex (NeurIPS 2025) on the NTUA data, leave-one-run-out.

Strategy (small data, 20 min of motion): keep the 12 pretrained sLSTM blocks frozen and fine-tune only the
output head (RMSNorm + residual output block, ~1.8 M of 35 M parameters) with the pinball (quantile) loss.
Because the backbone is frozen, its hidden states are computed once per forecast window and cached.

Forecasts use one forward pass with 4 masked rollout patches (4 x 32 = 128 steps >= 100), the same for the
zero-shot reference ("tirex_sp", original head) and the fine-tuned model ("tirex_ft"), so the comparison
isolates the effect of fine-tuning.

Usage:  python finetune_tirex.py cache     # compute frozen features (slow, once)
        python finetune_tirex.py train     # fine-tune heads per fold + save predictions (fast)
"""
import copy
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from tirex import load_model

from common import HORIZON, MAX_CONTEXT, RESULTS, RUNS, STRIDE, load_all, origins, save_preds

N_PATCH = 4
CACHE = os.path.join(RESULTS, "tirex_feature_cache.npz")
torch.set_num_threads(4)


def all_origins(n):
    """Every sample from MAX_CONTEXT to n-HORIZON (stride 1); evaluation uses the stride-5 subset."""
    return np.arange(MAX_CONTEXT, n - HORIZON + 1)


@torch.no_grad()
def backbone_features(model, ctx):
    """ctx: (B, L) raw context. Returns hidden states of the last N_PATCH token positions and scaler state."""
    ctx, _ = model._adjust_context_length(ctx, model.config.train_ctx_len, model.config.train_ctx_len)
    tok, state = model.tokenizer.input_transform(ctx)
    bs, _, dim = tok.shape
    pad = torch.full((bs, N_PATCH - 1, dim), torch.nan)
    tok = torch.cat([tok, pad], 1)
    mask = torch.isnan(tok).logical_not().float()
    h = model.input_patch_embedding(torch.cat([torch.nan_to_num(tok, nan=0.0), mask], 2))
    for blk in model.blocks:
        h = blk(h)
    return h[:, -N_PATCH:], state.loc.squeeze(-1), state.scale.squeeze(-1)


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = copy.deepcopy(model.out_norm)
        self.out = copy.deepcopy(model.output_patch_embedding)
        self.nq, self.ps = len(model.config.quantiles), model.config.output_patch_size

    def forward(self, h):  # h: (B, N_PATCH, D) -> normalised quantiles (B, HORIZON, nq)
        q = self.out(self.norm(h)).unflatten(-1, (self.nq, self.ps))   # (B, N_PATCH, nq, ps)
        q = q.permute(0, 2, 1, 3).reshape(h.shape[0], self.nq, -1)      # (B, nq, N_PATCH*ps)
        return q[:, :, :HORIZON].transpose(1, 2)


def build_cache():
    data = load_all()
    model = load_model("NX-AI/TiRex")
    model.eval()
    out = {}
    t0 = time.time()
    for r, x in data.items():
        o = all_origins(len(x))
        for c in range(3):
            H, LOC, SC = [], [], []
            for b in range(0, len(o), 64):
                ctx = torch.tensor(np.stack([x[i - MAX_CONTEXT:i, c] for i in o[b:b + 64]]), dtype=torch.float32)
                h, loc, sc = backbone_features(model, ctx)
                H.append(h.numpy().astype(np.float32)); LOC.append(loc.numpy()); SC.append(sc.numpy())
            out[f"h_{r}_{c}"] = np.concatenate(H)
            out[f"loc_{r}_{c}"] = np.concatenate(LOC)
            out[f"sc_{r}_{c}"] = np.concatenate(SC)
            print(f"  cached {r} ch{c}: {len(o)} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(CACHE, **out)


def pinball(qpred, y, levels):
    e = y[..., None] - qpred
    return torch.mean(torch.maximum(levels * e, (levels - 1) * e))


def _windows(data, cache, r, idx_filter=None):
    x = data[r]
    o = all_origins(len(x))
    keep = np.ones(len(o), bool) if idx_filter is None else idx_filter(o)
    H, Y, LOC, SC = [], [], [], []
    for c in range(3):
        H.append(cache[f"h_{r}_{c}"][keep]); LOC.append(cache[f"loc_{r}_{c}"][keep]); SC.append(cache[f"sc_{r}_{c}"][keep])
        Y.append(np.stack([x[i:i + HORIZON, c] for i in o[keep]]))
    return [np.concatenate(v) for v in (H, Y, LOC, SC)]


def fit_head(model, levels, data, cache, train_runs, epochs=200, lr=2e-4, wd=1e-4, patience=10, seed=0):
    """Fine-tune a copy of the output head on `train_runs`; early stopping on the last 20 % of each run."""
    torch.manual_seed(seed)
    tr, va = [], []
    for r in train_runs:
        n = len(all_origins(len(data[r])))
        cut = int(0.8 * n)
        # leave a HORIZON gap so validation targets never overlap training targets
        tr.append(_windows(data, cache, r, lambda o, cut=cut: np.arange(len(o)) < cut - HORIZON))
        va.append(_windows(data, cache, r, lambda o, cut=cut: np.arange(len(o)) >= cut))
    cat = lambda parts: [torch.tensor(np.concatenate([p[k] for p in parts])) for k in range(4)]
    Htr, Ytr, Ltr, Str = cat(tr)
    Hva, Yva, Lva, Sva = cat(va)
    Ytr_n = (Ytr - Ltr[:, None]) / Str[:, None]
    Yva_n = (Yva - Lva[:, None]) / Sva[:, None]
    head = Head(model)
    with torch.no_grad():
        zs_val = pinball(head(Hva), Yva_n, levels).item()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = zs_val, copy.deepcopy(head.state_dict()), 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Htr))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            opt.zero_grad()
            loss = pinball(head(Htr[i]), Ytr_n[i], levels)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = pinball(head(Hva), Yva_n, levels).item()
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, zs_val, best, ep + 1


def predict_q(head, data, cache, run):
    """Quantile forecasts (n_origins, HORIZON, 3, 9) at the standard evaluation origins of `run`."""
    eval_filter = lambda o: (o - MAX_CONTEXT) % STRIDE == 0
    H, _, L, S = map(torch.tensor, _windows(data, cache, run, eval_filter))
    n_o = len(origins(len(data[run])))
    with torch.no_grad():
        qq = head(H) * S[:, None, None] + L[:, None, None]
        qq = torch.sort(qq, dim=-1).values  # guard against quantile crossing
    return qq.numpy().reshape(3, n_o, HORIZON, -1).transpose(1, 2, 0, 3)


def train_and_predict():
    data = load_all()
    model = load_model("NX-AI/TiRex")
    levels = torch.tensor(model.config.quantiles)
    cache = np.load(CACHE)
    preds_sp, q_sp, preds_ft, q_ft = {}, {}, {}, {}
    log = []
    for test in RUNS:
        head, zs_val, best, n_ep = fit_head(model, levels, data, cache, [r for r in RUNS if r != test])
        log.append(f"{test}: val pinball zero-shot {zs_val:.4f} -> fine-tuned {best:.4f} (epochs run {n_ep})")
        print("  " + log[-1], flush=True)
        for h, P, Q in ((Head(model), preds_sp, q_sp), (head, preds_ft, q_ft)):
            Q[test] = predict_q(h, data, cache, test)
            P[test] = Q[test][..., Q[test].shape[-1] // 2]
    levels_np = np.array(model.config.quantiles)
    save_preds("tirex_sp", preds_sp, q_sp, levels_np)
    save_preds("tirex_ft", preds_ft, q_ft, levels_np)
    with open(os.path.join(RESULTS, "log_finetune_tirex.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


def nested_calibration_predictions():
    """For every (test t, calibration run j): head fine-tuned on the two remaining runs, predictions on j.
    Lets conformal calibration on j use a model that never saw the test run (no leakage)."""
    data = load_all()
    model = load_model("NX-AI/TiRex")
    levels = torch.tensor(model.config.quantiles)
    cache = np.load(CACHE)
    out = {}
    for t in RUNS:
        for j in RUNS:
            if j == t:
                continue
            head, _, best, n_ep = fit_head(model, levels, data, cache, [r for r in RUNS if r not in (t, j)])
            out[f"{t}__{j}"] = predict_q(head, data, cache, j)
            print(f"  nested head test={t} cal={j}: val {best:.4f} ({n_ep} epochs)", flush=True)
    np.savez_compressed(os.path.join(RESULTS, "preds_tirex_ft_nested.npz"), **out)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("cache", "all"):
        build_cache()
    if step in ("train", "all"):
        train_and_predict()
    if step in ("nested", "all"):
        nested_calibration_predictions()
