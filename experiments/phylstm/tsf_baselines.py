"""Time-series-forecasting baselines (PatchTST, iTransformer, TimeXer: experiments/phylstm/tsf_models.py, Time-Series-
Library code) under evaluation protocol (a), trained and scored by the harness of phylstm_kaggle.py so that only the
network and its loss differ from 'PhyLSTM w/o physics' and 'PINN Cummins':
  same pack (usv_phylstm.npz), splits (6 folds grouped by wave seed + ood_Hs + ood_beta), normalisation (train mean and
  std of eta), task (past 300 samples = 60 s of eta = (heave, roll, pitch) in, next 100 samples = 20 s out; nothing about
  future waves), sampling (K = 64 random origins per training run per epoch, i.e. ~133 000 windows and ~520 steps of
  batch 256 per epoch, ~31 000 steps in 60 epochs), optimiser (Adam, lr x0.1 for the last third of the epochs, gradient
  norm clipped at 1), selection (epoch with the lowest validation MSE of the 20 s forecast, 16 fixed random origins per
  validation run) and scoring (phylstm_kaggle.forecast / test_metrics through Adapter: every origin, one per second, of
  every test run; errors in m and degrees).
Loss: MSE of the normalised 20 s forecast, the models' own objective (no physics, no past-reconstruction term). Each
model keeps its own instance normalisation. Learning rate: chosen per model on the cv0 validation fold (--tune), then
fixed for every split. (cv0's validation fold is cv1's test fold and holds some of the OOD test runs, so the lr choice
has seen data that other splits test on; this can only flatter the baselines.) Training in fp16 mixed precision
(autocast + gradient scaling, TSLib's --use_amp; ~2x faster for PatchTST on the RTX 3060); forecasts for validation
and test in float32; --fp32 trains in float32. A batch whose fp16 loss or gradients are not finite is skipped by the
gradient scaler and left out of the epoch's mean training loss (counted in 'bad_windows'); training stops early only if
the weights themselves become non-finite.

  python tsf_baselines.py --tune --jobs cv0:patchtst:1e-4 cv0:patchtst:3e-4 ... --out results/sim/tsf_tune
  python tsf_baselines.py --score-from results/sim/tsf_tune --jobs cv0:patchtst:3e-4 --out results/sim/phylstm/tsf
  python tsf_baselines.py --jobs cv1:patchtst:3e-4 ood_Hs:timexer:1e-4 --out results/sim/phylstm/tsf --worker cv1
  python tsf_baselines.py --smoke --jobs cv0:timexer:1e-4 --out <scratch dir>
Job spec 'split:model:lr'. Result keys as phylstm_kaggle.py: '<split>|<tag>|mse', '|mae', '|mse_o', '|n_epochs'
([trained, planned]) and '<tag>_last' (last epoch), tag = '<model>_e<epochs>' ('_s<seed>' if seed != 0, '_fp32',
'_smoke'), so table_rmse_mae.py reads them from results/sim/phylstm/**/part_*.npz. --tune: tag
'<model>_lr<lr>_e<epochs>', weights and history only, the test runs are never forecast. --score-from DIR: score the
saved weights of tuning job 'split:model:lr' under the final tag.
Restarts: a job whose results exist in any part_*.npz (or, with --tune, whose weights and history exist) in --out is
skipped, after checking that it was run with the same lr and model configuration (a mismatch fails the job instead of
mixing settings). Result files are written atomically (temporary file, then rename).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phylstm_kaggle as pk  # noqa: E402  (data, splits, windows, forecast, test_metrics)
import tsf_models as tm  # noqa: E402

CTX, HORIZON, WIN = pk.CTX, pk.HORIZON, pk.WIN


class Adapter(nn.Module):
    """phylstm_kaggle's Network-1 interface: x = Job.net1_input(W) (B, WIN, 4) = normalised eta with the future
    100 steps zeroed + the future flag. Only the past 300 steps of eta reach the model. Returns Z (B, WIN, 9) with the
    normalised forecast in Z[:, CTX:, :3], which Job.unpack turns into SI units; the rest is zero and unused."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x[:, :CTX, :3])
        Z = x.new_zeros(x.shape[0], WIN, 9)
        Z[:, CTX:, :3] = y
        return Z


def parse_spec(spec):
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] not in pk.SPLITS or parts[1] not in tm.MODELS:
        raise ValueError(f"bad job spec {spec!r}: want split:model:lr, split in {pk.SPLITS}, model in {list(tm.MODELS)}")
    return parts[0], parts[1], float(parts[2])


def tags(model, lr, a):
    suf = (f"_s{a.seed}" if a.seed else "") + ("_fp32" if a.fp32 else "") + ("_smoke" if a.smoke else "")
    return f"{model}_e{a.epochs}" + suf, f"{model}_lr{lr:g}_e{a.epochs}" + suf


def state_copy(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def atomic(path, write):
    """write(tmp) then rename onto path, so a crash never leaves a truncated result file. The temporary name does not
    match part_*.npz / history_*.json (table_rmse_mae.load_phylstm globs part_*.npz)."""
    d, f = os.path.split(path)
    tmp = os.path.join(d, "tmp_" + f)
    write(tmp)
    os.replace(tmp, path)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def train(job, model, lr, rows, T, a, log, name):
    """phylstm_kaggle.run_job's loop with the forecast loss. Returns best state, last state, best epoch, best val,
    history."""
    dev = job.dev
    n_ep = 2 if a.smoke else a.epochs
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = torch.tensor(rows["train"], device=dev)
    rng = np.random.default_rng(a.seed)
    va_runs = np.repeat(rows["val"], a.val_origins)
    va_orig = rng.integers(CTX, T - HORIZON + 1, size=len(va_runs))
    va_runs, va_orig = torch.tensor(va_runs, device=dev), torch.tensor(va_orig, device=dev)
    net = Adapter(model)
    amp = not a.fp32 and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_ep, hist, state = np.inf, 0, [], state_copy(model)
    for ep in range(n_ep):
        for g in opt.param_groups:
            g["lr"] = lr if ep < int(np.ceil(2 * n_ep / 3)) else lr / 10
        model.train()
        t0 = time.time()
        runs = tr.repeat_interleave(a.k)
        orig = torch.randint(CTX, T - HORIZON + 1, (len(runs),), device=dev)
        perm = torch.randperm(len(runs), device=dev)
        tot = torch.zeros((), device=dev)
        n_ok = torch.zeros((), device=dev)
        for b in range(0, len(perm), a.batch):
            k = perm[b:b + a.batch]
            W = job.windows(runs[k], orig[k])
            x = job.net1_input(W)[:, :CTX, :3]                               # normalised past eta
            y = (W[:, CTX:, :3] - job.mu_eta) / job.sd[:3]                    # normalised future eta
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):  # TSLib --use_amp: fp16 autocast
                out = model(x)
            loss = ((out.float() - y) ** 2).mean()
            scaler.scale(loss).backward()                                     # + gradient scaling; the scaler
            scaler.unscale_(opt)                                              # skips the step when the gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)                 # are not finite
            scaler.step(opt)
            scaler.update()
            fin = torch.isfinite(loss.detach())
            tot += torch.where(fin, loss.detach(), torch.zeros_like(loss.detach())) * len(k)
            n_ok += fin * len(k)
        model.eval()
        n_ok = int(n_ok.item())
        weights_ok = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        rec = {"epoch": ep + 1, "train_mse": tot.item() / max(n_ok, 1), "bad_windows": len(perm) - n_ok}
        if not weights_ok:
            rec["s"] = round(time.time() - t0, 1)
            hist.append(rec)
            log(f"  {name} epoch {ep + 1}: weights not finite ({rec['bad_windows']} bad windows), stopping")
            break
        pred = pk.forecast(job, net, va_runs, va_orig)
        truth = job.windows(va_runs, va_orig)[:, CTX:, :3]
        val = (((pred - truth) / job.sd[:3]) ** 2).mean().item()
        rec.update({"val_mse": val, "s": round(time.time() - t0, 1)})
        hist.append(rec)
        if val < best:
            best, best_ep, state = val, ep + 1, state_copy(model)
        log(f"  {name} epoch {ep + 1}/{n_ep} train {rec['train_mse']:.4f} val {val:.4f} ({rec['s']} s)"
            + (f", {rec['bad_windows']} windows in non-finite batches" if rec["bad_windows"] else ""))
    if best_ep == 0 or not np.isfinite(best):
        raise RuntimeError(f"{name}: validation never improved (best {best}), not keeping an untrained net")
    return state, state_copy(model), best_ep, best, hist


def score(job, model, rows, T, split, tag, best_state, last_state, log, name):
    res = {}
    net = Adapter(model)
    for suf, st in (("", best_state), ("_last", last_state)):
        model.load_state_dict(st)
        model.eval()
        mse, mae, mse_o = pk.test_metrics(job, net, rows["test"], T)
        res.update({f"{split}|{tag}{suf}|mse": mse, f"{split}|{tag}{suf}|mae": mae, f"{split}|{tag}{suf}|mse_o": mse_o})
        rmse = np.sqrt(mse.mean(axis=(0, 1)))
        log(f"  {name}{suf}: test RMSE heave {rmse[0]:.4f} m, roll {rmse[1]:.3f} deg, pitch {rmse[2]:.3f} deg")
    return res


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-origins", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="train in float32 (default: fp16 mixed precision; tag '_fp32')")
    ap.add_argument("--tune", action="store_true", help="validation only: weights + history, no test forecasts")
    ap.add_argument("--score-from", default=None, help="score the tuning weights saved in this directory")
    ap.add_argument("--worker", default="0", help="suffix of part_/history_/log_ files")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a few runs; tags end in '_smoke'")
    a = ap.parse_args()
    if a.smoke:
        a.k, a.val_origins = 4, 4
    specs = [parse_spec(s) for s in a.jobs]
    keys = [(s, m, lr) if a.tune else (s, m) for s, m, lr in specs]
    if len(set(keys)) < len(keys):
        raise ValueError(f"two jobs write the same result keys: {a.jobs}")
    os.makedirs(os.path.join(a.out, "weights"), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = pk.find_data(a.data)
    d = np.load(path)
    S, P, ids = d["S"], d["P"], d["ids"]
    T = S.shape[1]
    part_f = os.path.join(a.out, f"part_{a.worker}.npz")
    hist_f = os.path.join(a.out, f"history_{a.worker}.json")
    out = dict(np.load(part_f)) if os.path.exists(part_f) else {}
    hists = json.load(open(hist_f)) if os.path.exists(hist_f) else {}
    done_keys = set(out)                                                     # results of every worker in --out
    all_hists = {}
    for f in glob.glob(os.path.join(a.out, "part_*.npz")):
        if f != part_f:
            done_keys |= set(np.load(f).files)
    for f in glob.glob(os.path.join(a.out, "history_*.json")):
        all_hists.update(json.load(open(f)))
    all_hists.update(hists)
    logf = open(os.path.join(a.out, f"log_{a.worker}.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    mode = "tune" if a.tune else f"score-from {a.score_from}" if a.score_from else "train+test"
    log(f"tsf_baselines worker {a.worker}: device {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}, "
        f"torch {torch.__version__}, data {path}, mode {mode}, epochs {a.epochs}, k {a.k}, batch {a.batch}, seed {a.seed}, "
        f"precision {'fp32' if a.fp32 else 'fp16 AMP'}, sdpa {tm.USE_SDPA}, jobs {a.jobs}")
    failed = []
    for (split, model_name, lr), spec in zip(specs, a.jobs):
        final_tag, tune_tag = tags(model_name, lr, a)
        tag = tune_tag if a.tune else final_tag
        name = f"{split}:{tag}"
        wfile = os.path.join(a.out, "weights", f"{split}__{tag}.pt")
        cfg_now = tm.config(model_name, CTX, HORIZON, 3)
        prev = all_hists.get(name)
        done = (os.path.exists(wfile) and prev is not None) if a.tune else f"{split}|{tag}|mse" in done_keys
        if done:
            if prev is None or float(prev.get("lr", -1)) != lr or prev.get("cfg") != cfg_now:
                log(f"  {spec} FAILED: results for {name} exist with lr {prev and prev.get('lr')} / cfg "
                    f"{prev and prev.get('cfg')}, not lr {lr:g} / cfg {cfg_now}; remove them to rerun")
                failed.append(spec)
            else:
                log(f"  {name} already done (lr {lr:g}), skipped")
            continue
        rows = {p: d[f"{split}_{p}"] for p in ("train", "val", "test")}
        if a.smoke:
            rows = {"train": rows["train"][:30], "val": rows["val"][:10], "test": rows["test"][:6]}
        try:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            job = pk.Job(S, P, rows, dev)
            model, cfg = tm.build(model_name, CTX, HORIZON, 3)
            model = model.to(dev)
            npar = tm.n_params(model)
            t_job = time.time()
            if a.score_from:
                src = torch.load(os.path.join(a.score_from, "weights", f"{split}__{tune_tag}.pt"), map_location=dev)
                if src["cfg"] != cfg or src["lr"] != lr:
                    raise RuntimeError(f"{name}: saved config/lr differ from the current ones")
                best_state, last_state, best_ep, best = src["best"], src["last"], src["best_epoch"], src["best_val"]
                n_trained = src.get("n_trained", src["epochs"])
                h = {"best_epoch": best_ep, "best_val": best, "scored_from": a.score_from, "tune_tag": tune_tag}
                log(f"  {name}: {model_name} {npar} params, lr {lr:g}, weights of {tune_tag} (best epoch {best_ep}, val {best:.4f})")
            else:
                log(f"  {name}: {model_name} {npar} params, lr {lr:g}, cfg {cfg}")
                best_state, last_state, best_ep, best, hist = train(job, model, lr, rows, T, a, log, name)
                n_trained = len(hist)
                atomic(wfile, lambda p: torch.save(
                    {"best": {k: v.cpu() for k, v in best_state.items()},
                     "last": {k: v.cpu() for k, v in last_state.items()}, "cfg": cfg, "lr": lr, "model": model_name,
                     "split": split, "tag": tag, "seed": a.seed, "epochs": a.epochs, "n_trained": n_trained,
                     "best_epoch": best_ep, "best_val": best, "n_params": npar, "mu_eta": job.mu_eta.cpu(),
                     "sd": job.sd.cpu()}, p))
                h = {"best_epoch": best_ep, "best_val": best, "history": hist}
            h.update({"model": model_name, "lr": lr, "cfg": cfg, "n_params": npar, "seed": a.seed, "epochs": a.epochs,
                      "n_trained": n_trained, "fp32": a.fp32})
            if not a.tune:
                res = score(job, model, rows, T, split, tag, best_state, last_state, log, name)
                res[f"{split}|ids"] = ids[rows["test"]]
                for suf in ("", "_last"):
                    res[f"{split}|{tag}{suf}|n_epochs"] = np.array([n_trained, 2 if a.smoke else a.epochs])
                out.update(res)
                atomic(part_f, lambda p: np.savez_compressed(p, **out))
            hists[name] = h
            all_hists[name] = h
            atomic(hist_f, lambda p: dump_json(hists, p))
            log(f"  {name} done in {(time.time() - t_job) / 60:.1f} min (best epoch {best_ep}, val {best:.4f})")
        except Exception as exc:                                             # keep the other jobs
            import traceback
            log(f"  {spec} FAILED: {exc!r}\n{traceback.format_exc()}")
            failed.append(spec)
        finally:
            torch.cuda.empty_cache() if dev == "cuda" else None
    log(f"worker {a.worker} finished; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
