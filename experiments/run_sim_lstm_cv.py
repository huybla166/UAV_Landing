"""Stage 3 baseline under evaluation protocol (a): the paper's LSTM (Wang et al. 2026, JFI 363:108381) on dataset_v2,
6-fold cross-validation grouped by wave seed + the two out-of-distribution tests, scored at every forecast origin of
every test run (rolling origin, one origin per second, 60 s context, 20 s horizon as sim_common.py).

  D:/tools/venv-gpu/Scripts/python.exe experiments/run_sim_lstm_cv.py            # all 8 splits, then the report
  D:/tools/venv-gpu/Scripts/python.exe experiments/run_sim_lstm_cv.py cv0 ood_Hs # some splits
  D:/tools/venv-gpu/Scripts/python.exe experiments/run_sim_lstm_cv.py report     # tables only
  D:/tools/venv-gpu/Scripts/python.exe experiments/run_sim_lstm_cv.py smoke      # 1 epoch, few runs, separate file

LSTM as in the paper (p. 12-13): 1 layer, 50 units, 10-step input, dropout 0.1, Adam lr 0.005 with piecewise decay,
trained one step ahead and rolled out recursively for 100 steps (20 s); global train mean/std. Scaled for ~6 M training
windows as in run_sim_baselines.py: batch 1024 (paper: 64), 12 epochs with lr x0.2 every 4. Three seeds per split,
each scored on its own (no ensemble). Trained on the split's 'train' runs; the 'val' fold is only logged (one-step
MSE per epoch), since the paper's LSTM has no hyper-parameter to choose. Runs whose manifest status is not 'ok'
(24, all failing the Hs check) are left out everywhere.

Metrics per channel in physical units (heave m, roll and pitch deg). For the forecast from one origin,
e_k = prediction - truth at lead tau_k = k dt, k = 1..100, dt = 0.2 s:
  RMSE = sqrt(mean of e_k^2 over windows and leads)      RMSE(tau) = the same at one lead
  MAE  = mean of |e_k| over windows and leads            MAE(tau) = the same at one lead
the predictor metrics of the paper (p. 13).
results/sim/lstm_cv_metrics.npz: '<split>|seed<s>|mse' and '|mae' = per-run mean e^2 and |e| per lead (R, 100, 3),
from which both metrics follow exactly (all runs have the same number of origins); '<split>|seed<s>|mse_o' =
per-origin MSE over the 20 s (R, O, 3) for Diebold-Mariano tests against later models; '<split>|ids' = run ids.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from sim_common import CHANNELS, FS, HORIZON, K_FOLDS, RESULTS_SIM, cv_split, load, motion, ood_split, origins

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS = [f"cv{k}" for k in range(K_FOLDS)] + ["ood_Hs", "ood_beta"]
SEEDS = (0, 1, 2)
DT = 1.0 / FS
UNITS = ["m", "độ", "độ"]
SMOKE = "smoke" in sys.argv[1:]
TAG = "_smoke" if SMOKE else ""
METRICS = os.path.join(RESULTS_SIM, f"lstm_cv_metrics{TAG}.npz")
REPORT = os.path.join(RESULTS_SIM, f"lstm_cv{TAG}.md")


def split_rows(meta, name):
    """Row indices of meta for train/val/test of one split, QC failures removed."""
    sp = cv_split(meta, int(name[2:])) if name.startswith("cv") else ood_split(meta, name[4:])
    ok = (meta.status == "ok").to_numpy()
    return {p: idx[ok[idx]] for p, idx in sp.items()}


# ---------------------------------------------------------------- paper LSTM (as run_sim_baselines.py)
class PaperLSTM(nn.Module):
    def __init__(self, n_in=3, hidden=50, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_in)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(self.drop(h[:, -1]))


def one_step_mse(model, Z, L, batch=65536):
    """Mean one-step MSE of a model over every window of the normalised runs Z (R, T, 3) on the device."""
    R, T, _ = Z.shape
    starts = torch.arange(T - L, device=DEV)
    ar = torch.arange(L, device=DEV)
    n, tot = R * len(starts), 0.0
    with torch.no_grad():
        for b in range(0, n, batch):
            k = torch.arange(b, min(b + batch, n), device=DEV)
            ri, si = k // len(starts), starts[k % len(starts)]
            tot += nn.functional.mse_loss(model(Z[ri[:, None], si[:, None] + ar]), Z[ri, si + L], reduction="sum").item()
    return tot / (n * 3)


def train_lstm(Ytr, Yva, seed, L=10, epochs=12, batch=1024, step=4):
    mu, sd = Ytr.reshape(-1, 3).mean(0), Ytr.reshape(-1, 3).std(0)
    Z = torch.tensor((Ytr - mu) / sd, dtype=torch.float32, device=DEV)            # (R, T, 3)
    Zva = torch.tensor((Yva - mu) / sd, dtype=torch.float32, device=DEV)
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
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Z[ri[:, None], si[:, None] + ar]), Z[ri, si + L])
            loss.backward(); opt.step(); tot += loss.item() * len(k)
        sched.step()
        model.eval()
        print(f"    seed {seed} epoch {ep + 1}/{epochs}: train mse {tot / n:.5f}, val mse {one_step_mse(model, Zva, L):.5f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return model, mu, sd


def score(model, mu, sd, Y, L=10, runs_per_chunk=40):
    """Recursive 100-step forecasts from every origin of the runs Y (R, T, 3). Returns per-run mean e^2 and |e| per
    lead (R, H, 3) and the per-origin MSE over the horizon (R, O, 3)."""
    o = origins(Y.shape[1])
    R, O = Y.shape[0], len(o)
    mse = np.empty((R, HORIZON, 3), np.float32)
    mae = np.empty((R, HORIZON, 3), np.float32)
    mse_o = np.empty((R, O, 3), np.float32)
    for r0 in range(0, R, runs_per_chunk):
        y = Y[r0:r0 + runs_per_chunk]
        n = len(y)
        ctx = np.stack([y[:, i - L:i] for i in o], axis=1).reshape(n * O, L, 3)
        hist = torch.tensor((ctx - mu) / sd, dtype=torch.float32, device=DEV)
        steps = []
        with torch.no_grad():
            for _ in range(HORIZON):
                nxt = model(hist)
                steps.append(nxt)
                hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
        pred = torch.stack(steps, 1).cpu().numpy().reshape(n, O, HORIZON, 3) * sd + mu
        e = pred - np.stack([y[:, i:i + HORIZON] for i in o], axis=1)
        mse[r0:r0 + n] = (e ** 2).mean(axis=1)
        mae[r0:r0 + n] = np.abs(e).mean(axis=1)
        mse_o[r0:r0 + n] = (e ** 2).mean(axis=2)
    return mse, mae, mse_o


# ---------------------------------------------------------------- metrics and report
def metrics(mse, mae):
    """Per-run mean e^2, |e| per lead (R, H, 3) -> {RMSE, MAE} of (3,) arrays and RMSE per lead (H, 3)."""
    m2, m1 = mse.mean(axis=0), mae.mean(axis=0)                                     # (H, 3), mean over windows
    return dict(RMSE=np.sqrt(m2.mean(axis=0)), MAE=m1.mean(axis=0)), np.sqrt(m2)


def report(res, meta, Y):
    names = {**{f"cv{k}": f"Fold {k + 1}" for k in range(K_FOLDS)}, "ood_Hs": "Ngoài phân phối: Hs = 0,40 m",
             "ood_beta": "Ngoài phân phối: β = 45°"}
    keys = ["RMSE", "MAE"]
    leads = [1, 2, 5, 10, 20]
    done = [s for s in SPLITS if all(f"{s}|seed{sd}|mse" in res for sd in SEEDS)]
    per = {}
    for s in done:
        ms = [metrics(res[f"{s}|seed{sd}|mse"], res[f"{s}|seed{sd}|mae"]) for sd in SEEDS]
        per[s] = ({k: np.mean([m[0][k] for m in ms], axis=0) for k in keys}, np.mean([m[1] for m in ms], axis=0))
    cv = [s for s in done if s.startswith("cv")]
    sig = {s: Y[split_rows(meta, s)["test"]].std(axis=(0, 1)) for s in done}       # spread of the test runs' motion
    L = ["# LSTM của bài gốc trên dataset_v2 — giao thức (a): 6 fold theo seed sóng + 2 bài thử ngoài phân phối", "",
         "Script: `experiments/run_sim_lstm_cv.py` (định nghĩa chỉ số ở đầu file), số liệu thô: `lstm_cv_metrics.npz`, "
         "nhật ký: `log_lstm_cv.txt`. Mỗi fold: huấn luyện trên 4 fold, fold kế tiếp là tập kiểm định (chỉ ghi lại), "
         "chấm điểm trên fold kiểm tra ở mọi gốc dự đoán (mỗi 1 s, đoán 20 s). Mỗi ô là trung bình của 3 seed huấn luyện; "
         "dòng \"6 fold\" là trung bình ± độ lệch chuẩn giữa các fold. Bỏ 24 lần chạy `qc_fail`.", "",
         "Đơn vị: heave m, roll và pitch độ. RMSE và MAE (hai chỉ số của bài gốc, tr. 13) tính trên mọi gốc dự đoán và mọi "
         "tầm 0–20 s; cột \"RMSE 1 s\"… là RMSE riêng ở tầm đó. Số càng nhỏ càng tốt. σ = độ lệch chuẩn chuyển động của chính "
         "các lần chạy kiểm tra: đoán bằng giá trị trung bình cho RMSE ≈ σ.", ""]
    if len(cv) < K_FOLDS:
        L += [f"**Chưa đủ:** mới xong {len(done)}/{len(SPLITS)} lần chia ({', '.join(done) or 'chưa có'}).", ""]
    for c, ch in enumerate(CHANNELS):
        L += [f"## {ch.split('_')[0].capitalize()} ({UNITS[c]})", "",
              "| Tập | σ | " + " | ".join(keys) + " | " + " | ".join(f"RMSE {t} s" for t in leads) + " |",
              "|---|" + "---|" * (1 + len(keys) + len(leads))]
        def row(label, vals, lead_vals):
            return f"| {label} | " + " | ".join(vals) + " | " + " | ".join(lead_vals) + " |"
        for s in cv:
            m, r = per[s]
            L.append(row(names[s], [f"{sig[s][c]:.4g}"] + [f"{m[k][c]:.4g}" for k in keys], [f"{r[int(t / DT) - 1, c]:.4g}" for t in leads]))
        if len(cv) > 1:
            L.append(row(f"**6 fold**" if len(cv) == K_FOLDS else f"**{len(cv)} fold**",
                         [f"**{np.mean([sig[s][c] for s in cv]):.4g}**"] + [f"**{np.mean([per[s][0][k][c] for s in cv]):.4g} ± {np.std([per[s][0][k][c] for s in cv], ddof=1):.2g}**" for k in keys],
                         [f"**{np.mean([per[s][1][int(t / DT) - 1, c] for s in cv]):.4g}**" for t in leads]))
        for s in ("ood_Hs", "ood_beta"):
            if s in per:
                m, r = per[s]
                L.append(row(names[s], [f"{sig[s][c]:.4g}"] + [f"{m[k][c]:.4g}" for k in keys], [f"{r[int(t / DT) - 1, c]:.4g}" for t in leads]))
        L.append("")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"report -> {REPORT}")


# ---------------------------------------------------------------- main
def main(which):
    meta, X = load()
    Y = motion(X)                                                                   # (3150, T, 3) heave m, roll/pitch deg
    res = dict(np.load(METRICS)) if os.path.exists(METRICS) else {}
    for s in which:
        rows = split_rows(meta, s)
        if SMOKE:
            rows = dict(train=rows["train"][:40], val=rows["val"][:10], test=rows["test"][:8])
        Ytr, Yva, Yte = Y[rows["train"]], Y[rows["val"]], Y[rows["test"]]
        print(f"  {s}: train {len(Ytr)}, val {len(Yva)}, test {len(Yte)} runs", flush=True)
        res[f"{s}|ids"] = meta.id.to_numpy()[rows["test"]]
        for sd in SEEDS:
            if f"{s}|seed{sd}|mse" in res:
                print(f"    seed {sd} already scored", flush=True)
                continue
            model, mu, sdv = train_lstm(Ytr, Yva, sd, epochs=1 if SMOKE else 12)
            t0 = time.time()
            mse, mae, mse_o = score(model, mu, sdv, Yte)
            res[f"{s}|seed{sd}|mse"], res[f"{s}|seed{sd}|mae"], res[f"{s}|seed{sd}|mse_o"] = mse, mae, mse_o
            np.savez_compressed(METRICS, **res)
            m, _ = metrics(mse, mae)
            print(f"    seed {sd} scored ({time.time() - t0:.0f} s): RMSE heave {m['RMSE'][0]:.4f} m, roll {m['RMSE'][1]:.3f} deg, "
                  f"pitch {m['RMSE'][2]:.3f} deg", flush=True)
    report(res, meta, Y)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "smoke"]
    print(f"device {DEV}", flush=True)
    if args == ["report"]:
        meta, X = load()
        report(dict(np.load(METRICS)), meta, motion(X))
    else:
        main(args or (["cv0"] if SMOKE else SPLITS))
    print("done")
