"""PhyLSTM2 (Zhang, Liu, Sun 2020, CMAME 369:113226, arXiv:2002.10253) adapted to deck-motion forecasting on dataset_v2,
under evaluation protocol (a): the splits, forecast origins, horizon and metrics of experiments/run_sim_lstm_cv.py.

Self-contained (numpy + torch), for a Kaggle GPU T4 x2 kernel (one worker process per GPU) or a local GPU:
  python phylstm_kaggle.py                          # every job; data under /kaggle/input or data/sim/dataset_v2/kaggle_phylstm
  python phylstm_kaggle.py --smoke                  # tiny run, separate output file
  python phylstm_kaggle.py --jobs cv0:phy cv0:data  # some jobs
Input: usv_phylstm.npz from experiments/phylstm/pack_kaggle.py. Output (/kaggle/working, else results/sim/phylstm/):
phylstm_metrics.npz with '<split>|<variant>|mse', '|mae' = per-run mean e^2 and |e| per lead (R, 100, 3) and '|mse_o' =
per-origin MSE over the 20 s (R, O, 3), errors in m (heave) and degrees (roll, pitch), the layout of
lstm_cv_metrics.npz; '<split>|ids' = run ids; phylstm_history.json = losses per epoch.

Forecast task, as the LSTM baseline: at an origin o the predictor sees only the past 60 s of eta = (D, phi, theta) and
outputs the next 20 s. Nothing about future waves enters the predictor.
  Network 1 (paper Fig. 2, Box I): a 400-step window = the 300 past samples of eta, then 100 future steps where eta is
    0 and a flag channel is 1. 3 x [LSTM(100), ReLU], Dense(100), Dense(9) (layout of the official PhyLSTM2.py), mapped
    pairwise in time to Z = (z1 = eta, z2 = eta_dot, z3 = r, a latent state as the paper's hysteretic r). Unidirectional,
    so the output at t only sees inputs up to t. The forecast is z1 on the last 100 steps.
  Network 2 (Box II): g = LSTM2([Z, P]), same layout with Dense(3): the mass-normalised restoring force, P = run
    parameters (vessel, U, kappa, visc). Used only inside the training loss: neither P nor f reaches the predictor.
  Differentiator (Box III): 4th-order finite differences (official code: 2nd order, which misses eta_dot by 18-27 % at
    5 Hz on this data).
  Losses (eq. 9-11, 13; alpha = beta = gamma = 1, every term in units of its train std), on the whole window:
    J_d = MSE(z1 - eta) + MSE(z2 - eta_dot)      eta_dot from the kinematics (pack_kaggle.py)
    J_e = MSE(D z1 - z2)
    J_g = MSE(D z2 + g - f)                      f = the true wave excitation tau_w / M_ii: a training-only signal
  variant 'data' = Network 1 with J_d only (the paper's LSTM comparison); variant 'phy' = PhyLSTM2.
Training: each epoch draws K random origins per training run; Adam lr 1e-3, 1e-4 for the last third of the epochs
(paper, example 2), gradient norm clipped at 1; the epoch with the lowest validation MSE of the 20 s forecast is kept
(paper: minimum validation loss). One seed per job.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

FS, DT = 5.0, 0.2
CTX, HORIZON, STRIDE = 300, 100, 5          # 60 s context, 20 s horizon, one origin per second (sim_common.py)
WIN = CTX + HORIZON
SPLITS = [f"cv{k}" for k in range(6)] + ["ood_Hs", "ood_beta"]
VARIANTS = ["phy", "data"]
T0 = time.time()                            # process start: Kaggle stops a session after 12 h
DEFAULT_ARGS = []                           # used when run without "--" arguments (a Kaggle script kernel)
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARGV = sys.argv[1:] if any(x.startswith("--") for x in sys.argv[1:]) else DEFAULT_ARGS


def find_data(path):
    if path:
        return path
    hits = glob.glob("/kaggle/input/**/usv_phylstm.npz", recursive=True)
    if hits:
        return hits[0]
    return os.path.join(HERE, "..", "..", "data", "sim", "dataset_v2", "kaggle_phylstm", "usv_phylstm.npz")


SMOKE = "--smoke" in ARGV


def out_dir():
    d = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "..", "..", "results", "sim", "phylstm")
    d = os.path.join(d, "smoke") if SMOKE else d
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- PhyLSTM pieces
class Stack(nn.Module):
    """3 x [LSTM(units), ReLU], Dense(units), Dense(n_out), as LSTM_model in the official PhyLSTM2.py."""

    def __init__(self, n_in, n_out, units=100):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(n_in if i == 0 else units, units, batch_first=True) for i in range(3)])
        self.d1 = nn.Linear(units, units)
        self.d2 = nn.Linear(units, n_out)

    def forward(self, x):
        for lstm in self.lstms:
            x = torch.relu(lstm(x)[0])
        return self.d2(self.d1(x))


def ddt(x, dt=DT):
    """4th-order finite-difference time derivative of x (B, T, C), one-sided 4th order at both ends."""
    c = 1.0 / (12 * dt)
    first = (-25 * x[:, 0] + 48 * x[:, 1] - 36 * x[:, 2] + 16 * x[:, 3] - 3 * x[:, 4]) * c
    second = (-3 * x[:, 0] - 10 * x[:, 1] + 18 * x[:, 2] - 6 * x[:, 3] + x[:, 4]) * c
    mid = (-x[:, 4:] + 8 * x[:, 3:-1] - 8 * x[:, 1:-3] + x[:, :-4]) * c
    penult = (3 * x[:, -1] + 10 * x[:, -2] - 18 * x[:, -3] + 6 * x[:, -4] - x[:, -5]) * c
    last = (25 * x[:, -1] - 48 * x[:, -2] + 36 * x[:, -3] - 16 * x[:, -4] + 3 * x[:, -5]) * c
    return torch.cat([first[:, None], second[:, None], mid, penult[:, None], last[:, None]], dim=1)


class Job:
    def __init__(self, S, P, rows, dev):
        self.dev = dev
        self.S = torch.tensor(S, device=dev)                                  # (N, T, 9) eta | eta_dot | f, SI units
        self.P = torch.tensor(P, device=dev)
        tr = self.S[torch.tensor(rows["train"], device=dev)]
        flat = tr.reshape(-1, 9)
        self.mu_eta = flat[:, :3].mean(0)
        self.sd = flat.std(0)                                                  # eta, eta_dot, f scales
        self.pscale = torch.tensor([1.0, 2.0, 5.0, 1.0], device=dev)           # is_80kg, U, kappa, visc -> ~[0, 1]
        self.ar = torch.arange(WIN, device=dev)
        self.flag = torch.zeros(WIN, device=dev)
        self.flag[CTX:] = 1.0

    def windows(self, runs, origins):
        """runs, origins: (B,) tensors -> W (B, WIN, 9) for times o - CTX .. o + HORIZON - 1."""
        return self.S[runs[:, None], (origins - CTX)[:, None] + self.ar]

    def net1_input(self, W):
        eta = (W[..., :3] - self.mu_eta) / self.sd[:3]
        eta = eta * (1.0 - self.flag)[None, :, None]                           # future eta hidden
        return torch.cat([eta, self.flag[None, :, None].expand(len(W), WIN, 1)], dim=-1)

    def unpack(self, Z):
        """Network-1 output (B, WIN, 9) in train-std units -> eta, eta_dot in SI units, r."""
        return Z[..., :3] * self.sd[:3] + self.mu_eta, Z[..., 3:6] * self.sd[3:6], Z[..., 6:]


def losses(job, net1, net2, W, runs, variant):
    Z = net1(job.net1_input(W))
    eta, etad, _ = job.unpack(Z)
    sd = job.sd
    Jd = (((eta - W[..., :3]) / sd[:3]) ** 2).mean() + (((etad - W[..., 3:6]) / sd[3:6]) ** 2).mean()
    if variant == "data":
        return Jd, {"Jd": Jd.item()}
    Je = (((ddt(eta) - etad) / sd[3:6]) ** 2).mean()
    p = (job.P[runs] / job.pscale)[:, None, :].expand(len(W), WIN, 4)
    g = net2(torch.cat([Z, p], dim=-1)) * sd[6:9]
    Jg = (((ddt(etad) + g - W[..., 6:9]) / sd[6:9]) ** 2).mean()
    return Jd + Je + Jg, {"Jd": Jd.item(), "Je": Je.item(), "Jg": Jg.item()}


@torch.no_grad()
def forecast(job, net1, runs, origins, batch=512):
    """eta forecast (B, HORIZON, 3) in SI units for the given runs/origins."""
    out = []
    for b in range(0, len(runs), batch):
        W = job.windows(runs[b:b + batch], origins[b:b + batch])
        out.append(job.unpack(net1(job.net1_input(W)))[0][:, CTX:])
    return torch.cat(out)


def to_report_units(x):
    """(..., 3) heave m, roll/pitch rad -> heave m, roll/pitch deg (the units of the LSTM baseline)."""
    return torch.cat([x[..., :1], torch.rad2deg(x[..., 1:])], dim=-1)


def run_job(S, P, rows, split, variant, a, dev, log):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    job = Job(S, P, rows, dev)
    net1 = Stack(4, 9).to(dev)
    net2 = Stack(9 + 4, 3).to(dev) if variant == "phy" else None
    params = list(net1.parameters()) + (list(net2.parameters()) if net2 else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    T = S.shape[1]
    tr = torch.tensor(rows["train"], device=dev)
    rng = np.random.default_rng(a.seed)
    va_runs = np.repeat(rows["val"], a.val_origins)
    va_orig = rng.integers(CTX, T - HORIZON + 1, size=len(va_runs))
    va_runs, va_orig = torch.tensor(va_runs, device=dev), torch.tensor(va_orig, device=dev)
    best, best_ep, hist = np.inf, 0, []
    state = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
    for ep in range(a.epochs):
        if (time.time() - T0) / 3600 > a.max_hours - 0.25:
            log(f"  {split}:{variant} stopped after {ep} epochs: {a.max_hours} h limit")
            break
        for g_ in opt.param_groups:
            g_["lr"] = 1e-3 if ep < int(np.ceil(2 * a.epochs / 3)) else 1e-4
        net1.train(); net2 and net2.train()
        t0, tot, n = time.time(), {}, 0
        runs = tr.repeat_interleave(a.k)
        orig = torch.randint(CTX, T - HORIZON + 1, (len(runs),), device=dev)
        perm = torch.randperm(len(runs), device=dev)
        for b in range(0, len(perm), a.batch):
            k = perm[b:b + a.batch]
            W = job.windows(runs[k], orig[k])
            opt.zero_grad()
            loss, parts = losses(job, net1, net2, W, runs[k], variant)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for key, val in parts.items():
                tot[key] = tot.get(key, 0.0) + val * len(k)
            n += len(k)
        net1.eval()
        pred = forecast(job, net1, va_runs, va_orig)
        truth = job.windows(va_runs, va_orig)[:, CTX:, :3]
        val = (((pred - truth) / job.sd[:3]) ** 2).mean().item()
        rec = {"epoch": ep + 1, **{k_: v / n for k_, v in tot.items()}, "val_mse": val, "s": round(time.time() - t0, 1)}
        hist.append(rec)
        if val < best:
            best, best_ep = val, ep + 1
            state = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
        log(f"  {split}:{variant} epoch {ep + 1}/{a.epochs} " + " ".join(f"{k_} {v:.4f}" for k_, v in rec.items()
                                                                         if k_ not in ("epoch", "s")) + f" ({rec['s']} s)")
    if best_ep == 0 or not np.isfinite(best):
        raise RuntimeError(f"{split}:{variant}: validation never improved (best {best}), not scoring an untrained net")
    last = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
    res = {}
    for tag, st in (("", state), ("_last", last)):                       # best-on-val (paper) and last epoch (baseline)
        net1.load_state_dict(st); net1.eval()
        mse, mae, mse_o = test_metrics(job, net1, rows["test"], S.shape[1])
        res.update({f"{split}|{variant}{tag}|mse": mse, f"{split}|{variant}{tag}|mae": mae,
                    f"{split}|{variant}{tag}|mse_o": mse_o})
        rmse = np.sqrt(mse.mean(axis=(0, 1)))
        log(f"  {split}:{variant}{tag} ({'best epoch ' + str(best_ep) if not tag else 'epoch ' + str(len(hist))}, val "
            f"{best if not tag else hist[-1]['val_mse']:.4f}): test RMSE heave {rmse[0]:.4f} m, roll {rmse[1]:.3f} deg, "
            f"pitch {rmse[2]:.3f} deg")
    return res, {"best_epoch": best_ep, "best_val": best, "history": hist}


def test_metrics(job, net1, te, T):
    """Forecast from every origin of the test runs; per-run mean e^2, |e| per lead and per-origin MSE (as
    run_sim_lstm_cv.score), errors in m and degrees."""
    o = np.arange(CTX, T - HORIZON + 1, STRIDE)
    R, O = len(te), len(o)
    mse = np.empty((R, HORIZON, 3), np.float32); mae = np.empty_like(mse); mse_o = np.empty((R, O, 3), np.float32)
    for r0 in range(0, R, 40):
        rr = te[r0:r0 + 40]
        runs = torch.tensor(np.repeat(rr, O), device=job.dev)
        orig = torch.tensor(np.tile(o, len(rr)), device=job.dev)
        e = to_report_units(forecast(job, net1, runs, orig)) - to_report_units(job.windows(runs, orig)[:, CTX:, :3])
        e = e.reshape(len(rr), O, HORIZON, 3).double().cpu().numpy()
        mse[r0:r0 + len(rr)] = (e ** 2).mean(axis=1)
        mae[r0:r0 + len(rr)] = np.abs(e).mean(axis=1)
        mse_o[r0:r0 + len(rr)] = (e ** 2).mean(axis=2)
    return mse, mae, mse_o


def worker(a, jobs, tag):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(find_data(a.data))
    S, P, ids = d["S"], d["P"], d["ids"]
    out, hist = {}, {}
    logf = open(os.path.join(out_dir(), f"log_{tag}.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"worker {tag}: device {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}, torch {torch.__version__}, "
        f"jobs {jobs}")
    t_start, took, failed = time.time(), {}, []
    for spec in jobs:
        split, variant = spec.split(":")
        est = took.get(variant, 0.0) * 1.1                                    # seconds of the last job of this variant
        if (time.time() - T0 + est) / 3600 > a.max_hours:
            log(f"  {spec} skipped: would end after the {a.max_hours} h limit")
            failed.append(spec)
            continue
        rows = {p: d[f"{split}_{p}"] for p in ("train", "val", "test")}
        if a.smoke:
            rows = {"train": rows["train"][:30], "val": rows["val"][:10], "test": rows["test"][:6]}
        t_job = time.time()
        try:
            res, h = run_job(S, P, rows, split, variant, a, dev, log)
        except Exception as exc:                                              # keep the other jobs of this worker
            import traceback
            log(f"  {spec} FAILED: {exc!r}\n{traceback.format_exc()}")
            failed.append(spec)
            continue
        took[variant] = time.time() - t_job
        res[f"{split}|ids"] = ids[rows["test"]]
        out.update(res); hist[spec] = h
        np.savez_compressed(os.path.join(out_dir(), f"part_{tag}.npz"), **out)
        json.dump(hist, open(os.path.join(out_dir(), f"history_{tag}.json"), "w"), indent=1)
    log(f"worker {tag} done in {(time.time() - t_start) / 3600:.2f} h; failed or skipped: {failed or 'none'}")
    if failed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k", type=int, default=64, help="random origins per training run per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-origins", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", default=None)
    a = ap.parse_known_args(ARGV)[0]
    if a.smoke:
        a.epochs, a.k, a.val_origins = 2, 4, 4
    jobs = a.jobs or [f"{s}:{v}" for v in VARIANTS for s in SPLITS]
    if a.worker is not None:
        worker(a, jobs, a.worker)
        return
    for f in glob.glob(os.path.join(out_dir(), "part_*.npz")) + glob.glob(os.path.join(out_dir(), "history_*.json"))             + glob.glob(os.path.join(out_dir(), "log_*.txt")):
        os.remove(f)                                                          # no stale results in the merge
    n_gpu = torch.cuda.device_count()
    print(f"data {find_data(a.data)}; {n_gpu} GPU(s); {len(jobs)} jobs; args {ARGV}", flush=True)
    codes = []
    if n_gpu > 1 and "__file__" not in globals():
        print("WARNING: no __file__, cannot start one process per GPU; running everything on GPU 0", flush=True)
    if n_gpu > 1 and "__file__" in globals():
        procs = []
        for i in range(n_gpu):
            mine = jobs[i::n_gpu]
            if not mine:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", str(i), "--jobs", *mine,
                   "--epochs", str(a.epochs), "--k", str(a.k), "--batch", str(a.batch), "--val-origins", str(a.val_origins),
                   "--seed", str(a.seed), "--max-hours", str(a.max_hours)] + (["--data", a.data] if a.data else []) \
                + (["--smoke"] if a.smoke else [])
            procs.append(subprocess.Popen(cmd, env=env))
        codes = [p.wait() for p in procs]
        print(f"workers exited with {codes}", flush=True)
    else:
        try:
            worker(a, jobs, "0")
        except SystemExit as exc:
            codes = [exc.code]
    merged, hist = {}, {}
    for f in sorted(glob.glob(os.path.join(out_dir(), "part_*.npz"))):
        merged.update(dict(np.load(f)))
    for f in sorted(glob.glob(os.path.join(out_dir(), "history_*.json"))):
        hist.update(json.load(open(f)))
    name = "phylstm_metrics_smoke.npz" if a.smoke else "phylstm_metrics.npz"
    np.savez_compressed(os.path.join(out_dir(), name), **merged)
    json.dump(hist, open(os.path.join(out_dir(), name.replace("metrics", "history").replace(".npz", ".json")), "w"), indent=1)
    missing = [j for j in jobs if "{}|{}|mse".format(*j.split(":")) not in merged]
    print(f"merged {len(jobs) - len(missing)}/{len(jobs)} jobs -> {os.path.join(out_dir(), name)}; missing: {missing or 'none'}",
          flush=True)
    if missing or any(codes):
        sys.exit(1)                                                           # the kernel then ends as ERROR, not COMPLETE


if __name__ == "__main__":
    main()
