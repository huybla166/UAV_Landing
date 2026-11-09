"""Forecasts of PINN Cummins and TimeXer on one test run from EVERY origin (one per sample, 0.2 s apart), for the
prediction figures of plot_predictions.py. Same prediction path as the result tables: phylstm_kaggle.Job / forecast
(normalisation with the split's training runs, the 100 future samples of eta masked) and the saved best-validation
weights (PINN Cummins: results/sim/phylstm/local_p3*/weights; TimeXer: results/sim/phylstm/tsf/weights, cv0 from the
lr tuning in results/sim/tsf_tune/weights).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/predict_runs.py [--split cv0] [--run ID]

Without --run: the test run whose PINN Cummins 2 s RMSE (normalised, mean over heave, roll and pitch, from the stored
per-run metrics) is the median of the split's test runs, i.e. a typical run, not a chosen one.
Check: on the origins of the tables (one per second) the per-lead MSE of these forecasts must reproduce the stored
per-run metrics of the run (part_*.npz), so lead and time are aligned exactly as in the tables.
Writes results/sim/figures/pred_<split>_run<id>.npz: t (T,) s, eta (T, 3) heave m / roll deg / pitch deg, origins (O,)
sample index of the first forecast step (the last observed sample is origin - 1), pinn / timexer (O, 100, 3) forecasts
in the same units (column j = lead (j + 1) x 0.2 s, time origin + j), the run's sea state and the check.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import phylstm_kaggle as pk  # noqa: E402
import table_rmse_mae as T  # noqa: E402
import tsf_baselines as tb  # noqa: E402
import tsf_models as tm  # noqa: E402
from sim_common import FS, RESULTS_SIM, load  # noqa: E402

PHY_DIR = os.path.join(RESULTS_SIM, "phylstm")
OUT_DIR = os.path.join(RESULTS_SIM, "figures")


def weights_file(pattern):
    hits = [f for f in glob.glob(os.path.join(PHY_DIR, "**", "weights", pattern), recursive=True)
            if not any(x in f for x in ("smoke", "rerun", "failed_cpu"))]
    hits += glob.glob(os.path.join(RESULTS_SIM, "tsf_tune", "weights", pattern))
    if len(hits) != 1:
        raise SystemExit(f"{pattern}: expected one weights file, found {hits}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="cv0")
    ap.add_argument("--run", type=int, default=None, help="run id (default: the median-error test run)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(pk.find_data(None))
    S, P, ids = d["S"], d["P"], d["ids"]
    rows = {p: d[f"{a.split}_{p}"] for p in ("train", "val", "test")}
    phy = T.load_phylstm()
    assert np.array_equal(phy[f"{a.split}|ids"], ids[rows["test"]]), "stored test runs differ from the pack"

    # the run: median of the stored PINN Cummins 2 s RMSE (normalised by the train std of each channel, float64)
    rep = pk.to_report_units(torch.tensor(S[rows["train"], :, :3])).double().numpy()
    sd = rep.reshape(-1, 3).std(0)
    m2 = phy[f"{a.split}|cummins_e60|mse"].astype(np.float64)                 # (R, 100, 3)
    r2 = (np.sqrt(m2[:, :int(2 * FS)].mean(axis=1)) / sd).mean(axis=1)       # (R,)
    if a.run is None:
        r = int(np.argsort(r2)[len(r2) // 2])
    else:
        r = int(np.nonzero(ids[rows["test"]] == a.run)[0][0])
    row = int(rows["test"][r])
    run_id = int(ids[row])

    job = pk.Job(S, P, rows, dev)
    T_len = S.shape[1]
    origins = np.arange(pk.CTX, T_len - pk.HORIZON + 1)                      # every sample
    runs_t = torch.full((len(origins),), row, device=dev)
    orig_t = torch.tensor(origins, device=dev)

    # PINN Cummins (Network 1 of phylstm_kaggle, 100 units)
    ck = torch.load(weights_file(f"{a.split}__cummins_e60.pt"), map_location=dev)
    assert (torch.allclose(ck["mu_eta"].to(dev), job.mu_eta, rtol=1e-4, atol=1e-6)
            and torch.allclose(ck["sd"].to(dev), job.sd, rtol=1e-4, atol=1e-6)), "normalisation differs from training"
    net1 = pk.Stack(4, 9, 100).to(dev)
    net1.load_state_dict(ck["best"])
    net1.eval()
    pinn = pk.to_report_units(pk.forecast(job, net1, runs_t, orig_t, variant="cummins")).cpu().numpy()

    # TimeXer at the lr chosen on the cv0 validation fold
    lr = json.load(open(os.path.join(RESULTS_SIM, "tsf_tune", "chosen_lr.json")))["timexer"]["lr"]
    try:
        f = weights_file(f"{a.split}__timexer_e60.pt")
    except SystemExit:
        f = weights_file(f"{a.split}__timexer_lr{lr:g}_e60.pt")
    ck2 = torch.load(f, map_location=dev)
    model, cfg = tm.build("timexer", pk.CTX, pk.HORIZON, 3)
    assert ck2["cfg"] == cfg and float(ck2["lr"]) == lr, (ck2["cfg"], cfg, ck2["lr"])
    model.load_state_dict(ck2["best"])
    net2 = tb.Adapter(model).to(dev)
    net2.eval()
    timexer = pk.to_report_units(pk.forecast(job, net2, runs_t, orig_t)).cpu().numpy()

    eta = pk.to_report_units(torch.tensor(S[row, :, :3])).numpy()            # (T, 3)
    # check against the stored per-run metrics: origins one per second (as the tables), per-lead MSE
    sel = (origins - pk.CTX) % pk.STRIDE == 0
    truth = np.stack([eta[o:o + pk.HORIZON] for o in origins[sel]])          # (O5, 100, 3), time o + j
    check = {}
    for name, pred, tag in (("pinn", pinn, "cummins_e60"), ("timexer", timexer, "timexer_e60")):
        mse = ((pred[sel].astype(np.float64) - truth) ** 2).mean(axis=0)     # (100, 3)
        stored = phy[f"{a.split}|{tag}|mse"][r].astype(np.float64)
        check[name] = float(np.max(np.abs(mse / stored - 1)))
        print(f"{name}: per-lead MSE vs stored metrics of the run, max relative difference {check[name]:.2e}")
        assert check[name] < 1e-2, f"{name}: forecasts do not reproduce the stored metrics"

    meta, _ = load()
    m = meta.set_index("id").loc[run_id]
    sea = {k: (m[k].item() if hasattr(m[k], "item") else m[k]) for k in ("vessel", "Hs", "Tp", "beta_deg", "U")}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"pred_{a.split}_run{run_id}.npz")
    np.savez_compressed(path, t=np.arange(T_len) / FS, eta=eta, origins=origins, pinn=pinn.astype(np.float32),
                        timexer=timexer.astype(np.float32), run_id=run_id, split=a.split, sea=json.dumps(sea),
                        rank=(f"median of the {len(r2)} {a.split} test runs by PINN Cummins 2 s RMSE "
                              f"(rank {int(np.sum(r2 < r2[r])) + 1})"),
                        check=json.dumps(check))
    print(f"run {run_id} ({sea}), test run {r + 1}/{len(r2)}, PINN 2 s normalised RMSE {r2[r]:.4f} "
          f"(split median {np.median(r2):.4f}) -> {path}")


if __name__ == "__main__":
    main()
