"""PINN Cummins against the time-series-forecasting (TSF) baselines PatchTST, iTransformer and TimeXer
(tsf_baselines.py), the paper's LSTM, PhyLSTM2, the same LSTM network without physics, and two references without
training from experiments/diagnostics/wiener.py. Metric of table_rmse_mae.py: for a horizon of H s, per channel
RMSE = sqrt(mean e^2 over test runs, origins and leads <= H) and MAE = mean |e| over the same, both divided by the
channel's standard deviation over the training runs of the split (float64), then the mean over heave, roll and pitch.

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/report_tsf.py

Writes results/sim/table_tsf.md: (1) 6-fold mean, (2) the two out-of-distribution tests, (3) PINN Cummins vs each model
split by split, (4) the TSF baselines with their last-epoch weights, (5) the learning rates chosen on the cv0
validation fold, the fp32 check and the best epochs.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
import table_rmse_mae as T  # noqa: E402
from sim_common import FS, OOD as OOD_LEVEL, RESULTS_SIM, load, motion  # noqa: E402

CV = [f"cv{k}" for k in range(6)]
OOD = ["ood_Hs", "ood_beta"]
# (label, source, result tag, trainable parameters of the forecasting network); source: 'lstm' = lstm_cv_metrics.npz,
# 'phy' = results/sim/phylstm/**, 'ref' = results/sim/diagnostics/wiener_metrics.npz (no training)
ROWS = [
    ("Mean of the past 60 s (reference)", "ref", "mean_past", None),
    ("Wiener filter on the past 60 s (no training)", "ref", "wiener_past", None),
    ("LSTM [Wang 2026]", "lstm", "lstm", 11153),
    ("PatchTST [Nie 2023]", "phy", "patchtst_e60", 873444),
    ("iTransformer [Liu 2024]", "phy", "itransformer_e60", 4940388),
    ("TimeXer [Wang 2024]", "phy", "timexer_e60", 1410660),
    ("PhyLSTM w/o physics", "phy", "data_e60", 215009),
    ("PhyLSTM2 [Zhang 2020]", "phy", "phy_e60", 215009),
    ("PINN Cummins (proposed)", "phy", "cummins_e60", 215009),
]
TSF = ["patchtst_e60", "itransformer_e60", "timexer_e60"]
WIENER = os.path.join(RESULTS_SIM, "diagnostics", "wiener_metrics.npz")
TSF_DIR = os.path.join(RESULTS_SIM, "phylstm", "tsf")
TUNE_DIR = os.path.join(RESULTS_SIM, "tsf_tune")


def metrics(src, tag, s, sd, H):
    """(RMSE, MAE) of one model on split s at horizon H; LSTM [Wang 2026] = mean over its 3 seeds."""
    n = int(H * FS)
    if tag == "lstm":
        v = [T.score(src[f"{s}|seed{k}|mse"], src[f"{s}|seed{k}|mae"], sd, n) for k in L.SEEDS]
        return tuple(np.mean(v, axis=0))
    return T.score(src[f"{s}|{tag}|mse"], src[f"{s}|{tag}|mae"], sd, n)


def has(src, tag, s):
    return all(f"{s}|seed{k}|mse" in src for k in L.SEEDS) if tag == "lstm" else f"{s}|{tag}|mse" in src


def mean_rmse(src, tag, splits, sd):
    """Normalised RMSE at every horizon, mean over the splits."""
    return [np.mean([metrics(src, tag, s, sd[s], H)[0] for s in splits]) for H in T.HORIZONS]


def ood_title(meta, s):
    col, level = OOD_LEVEL[s[4:]]
    name, unit = ("Hs", " m") if col == "Hs" else ("wave direction", "°")
    train = np.unique(meta[col].to_numpy(float)[L.split_rows(meta, s)["train"]])
    where = "an interior level" if train.min() < level < train.max() else "outside the training range"
    return (f"## Out of distribution: {name} = {level:g}{unit}, not in training (training levels "
            f"{', '.join(f'{v:g}' for v in train)}{unit}, so {where})")


def table(lines, title, splits, rows, sd):
    lines += [title, "", "| Model | Params | " + " | ".join(f"{H}s RMSE | {H}s MAE" for H in T.HORIZONS) + " |",
              "|---|---|" + "---|" * (2 * len(T.HORIZONS))]
    best = {}
    vals = {}
    for name, tag, npar, src in rows:
        if not all(has(src, tag, s) for s in splits):
            continue
        v = [np.mean([metrics(src, tag, s, sd[s], H) for s in splits], axis=0) for H in T.HORIZONS]
        vals[name] = (npar, v)
        for i, (r, m) in enumerate(v):
            best[i, 0] = min(best.get((i, 0), np.inf), r)
            best[i, 1] = min(best.get((i, 1), np.inf), m)
    for name, (npar, v) in vals.items():
        cells = []
        for i, (r, m) in enumerate(v):
            cells += [f"**{r:.4f}**" if r == best[i, 0] else f"{r:.4f}", f"**{m:.4f}**" if m == best[i, 1] else f"{m:.4f}"]
        lines.append(f"| {name} | {f'{npar:,}' if npar else '–'} | " + " | ".join(cells) + " |")
    missing = [name for name, tag, _, src in rows if name not in vals]
    if missing:
        lines.append(f"\nNot yet available on every split: {', '.join(missing)}.")
    lines.append("")


def last_epoch(lines, rows, phy, ref, sd):
    """The TSF baselines (and PINN Cummins) with their epoch-60 weights ('<tag>_last') next to the selected epoch."""
    sets = [("6-fold", CV)] + [(s, [s]) for s in OOD]
    models = [(n, t) for n, t, _, _ in rows if t in TSF + ["cummins_e60"]]
    lines += ["## Best-validation epoch vs last epoch", "",
              "The tables above score every trained model at the epoch with the lowest validation MSE of the whole 20 s "
              "forecast. That criterion is dominated by the long, barely predictable leads, so the epoch it picks need "
              "not be the best one at short leads. Normalised RMSE with the selected and with the epoch-60 weights:", "",
              "| Split set | Model | Weights | " + " | ".join(f"{H}s RMSE" for H in T.HORIZONS) + " |",
              "|---|---|---|" + "---|" * len(T.HORIZONS)]
    best = {}
    for set_name, splits in sets:
        for name, tag in models:
            for rule, t in (("best validation", tag), ("last epoch", tag + "_last")):
                if not all(has(phy, t, s) for s in splits):
                    lines.append(f"| {set_name} | {name} | {rule} | " + " | ".join("–" for _ in T.HORIZONS) + " |")
                    continue
                v = mean_rmse(phy, t, splits, sd)
                if tag in TSF:
                    best.setdefault((set_name, rule), []).append(v)
                lines.append(f"| {set_name} | {name} | {rule} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
    lines += ["", "PINN Cummins (best-validation epoch, as in the tables above) against the best TSF baseline of each "
              "column under each rule and against the Wiener filter: % change of normalised RMSE (negative = PINN "
              "Cummins better).", "",
              "| Split set | PINN Cummins vs | " + " | ".join(f"{H} s" for H in T.HORIZONS) + " |",
              "|---|---|" + "---|" * len(T.HORIZONS)]
    for set_name, splits in sets:
        pinn = mean_rmse(phy, "cummins_e60", splits, sd)
        others = [(f"best TSF baseline, {rule}", np.min(best[set_name, rule], axis=0))
                  for rule in ("best validation", "last epoch") if (set_name, rule) in best]
        others.append(("Wiener filter on the past 60 s", mean_rmse(ref, "wiener_past", splits, sd)))
        for what, b in others:
            lines.append(f"| {set_name} | {what} | " + " | ".join(f"{100 * (p / x - 1):+.2f}" for p, x in zip(pinn, b))
                         + " |")
    lines.append("")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lstm, phy, ref = dict(np.load(L.METRICS)), T.load_phylstm(), dict(np.load(WIENER))
    meta, X = load()
    Y = motion(X)
    splits = CV + OOD
    sd = {s: Y[L.split_rows(meta, s)["train"]].std(axis=(0, 1), dtype=np.float64) for s in splits}
    for s in splits:
        for src in (phy, ref):
            if f"{s}|ids" in src:
                assert np.array_equal(src[f"{s}|ids"], lstm[f"{s}|ids"]), f"test runs differ on {s}"
    rows = [(n, t, p, {"lstm": lstm, "phy": phy, "ref": ref}[src]) for n, src, t, p in ROWS]
    lines = ["# PINN Cummins vs time-series-forecasting baselines (dataset_v2)", "",
             "Every model forecasts the next 20 s of heave, roll and pitch (5 Hz) from their past alone (no wave "
             "input). Input: the past 60 s; LSTM [Wang 2026] follows its paper instead (10 samples = 2 s of input, "
             "trained one step ahead and rolled out recursively for 100 steps, 12 epochs, last-epoch weights, mean of "
             "3 seeds). RMSE and MAE over the leads 0.2 .. H s, divided by the training standard deviation of each "
             "channel in the split (float64), mean over heave, roll and pitch; bold = best in the column. 1.0 = an "
             "error as large as the training standard deviation; the row 'Mean of the past 60 s' is the no-skill "
             "level of each test set.", "",
             "Trained models other than LSTM [Wang 2026]: one seed each (seed 0); 60 epochs of 64 random origins per "
             "training run, batch 256; weights of the epoch with the lowest validation MSE of the 20 s forecast. "
             "PatchTST, iTransformer, TimeXer (tsf_models.py: Time-Series-Library architectures, settings of the "
             "authors' Weather scripts): loss = MSE of the 100 future samples, per-window instance normalisation, lr "
             "chosen on the cv0 validation fold, fp16 mixed precision. PhyLSTM w/o physics, PhyLSTM2, PINN Cummins "
             "(phylstm_kaggle.py): Network 1 decodes the whole 400-step window with the future input masked; the "
             "loss covers the whole window (reconstruction of the past 60 s, plus velocity targets: eta_dot, for PINN "
             "Cummins the body velocities w, p, q), plus the physics residuals of PhyLSTM2 / PINN Cummins; global "
             "z-score; lr 1e-3, 1e-4 after epoch 40; fp32. References (experiments/diagnostics/wiener.py): the Wiener "
             "filter is the linear minimum-MSE predictor with the autocovariance estimated at each forecast origin "
             "from the past 60 s of the same run (settings chosen on 24 cv0 validation runs). "
             "Source: experiments/phylstm/report_tsf.py.", ""]
    table(lines, "## 6-fold cross-validation grouped by wave seed (mean over the folds)", CV, rows, sd)
    for s in OOD:
        table(lines, ood_title(meta, s), [s], rows, sd)
    lines += ["## PINN Cummins vs each model, split by split", "",
              "% change of the normalised RMSE of PINN Cummins against the model (negative = PINN Cummins better).", ""]
    for name, tag, _, src in rows:
        if tag == "cummins_e60":
            continue
        common = [s for s in splits if has(src, tag, s) and has(phy, "cummins_e60", s)]
        if not common:
            continue
        lines += [f"**vs {name}**", "", "| Horizon | " + " | ".join(common) + " | better on |",
                  "|---|" + "---|" * (len(common) + 1)]
        for H in T.HORIZONS:
            d = [100 * (metrics(phy, "cummins_e60", s, sd[s], H)[0] / metrics(src, tag, s, sd[s], H)[0] - 1)
                 for s in common]
            lines.append(f"| {H} s | " + " | ".join(f"{x:+.2f}" for x in d) + f" | {sum(x < 0 for x in d)}/{len(d)} |")
        lines.append("")
    last_epoch(lines, rows, phy, ref, sd)
    chosen = os.path.join(TUNE_DIR, "chosen_lr.json")
    if os.path.exists(chosen):
        val0 = L.split_rows(meta, "cv0")["val"]
        shared = ", ".join(f"{len(np.intersect1d(val0, L.split_rows(meta, s)['test']))} of the "
                           f"{len(L.split_rows(meta, s)['test'])} {s}" for s in OOD)
        same = np.array_equal(np.sort(val0), np.sort(L.split_rows(meta, "cv1")["test"]))
        lines += ["## Learning rate (cv0 validation fold), fp32 check and best epochs", "",
                  "Validation MSE = normalised MSE of the 20 s forecast on 16 fixed origins per validation run (the "
                  "epoch-selection criterion, on the same windows as for the PhyLSTM models). The lr is chosen once on "
                  "cv0 and used on every split. cv0's validation fold "
                  f"{'is' if same else 'is not'} cv1's test fold and holds {shared} test runs, so the choice can only "
                  "flatter the baselines on cv1 and the two OOD tests. fp32 check: cv0 retrained at the chosen lr "
                  "without mixed precision (validation only; best validation MSE and its epoch, fp16 run in brackets).",
                  "",
                  "| Model | lr tried: best validation MSE | chosen lr | fp32 check | best epoch / epochs trained per split |",
                  "|---|---|---|---|---|"]
        hist, tune_hist = {}, {}
        for f in glob.glob(os.path.join(TSF_DIR, "history_*.json")):
            hist.update(json.load(open(f)))
        for f in glob.glob(os.path.join(TUNE_DIR, "history_*.json")):
            tune_hist.update(json.load(open(f)))
        for m, c in json.load(open(chosen)).items():
            tried = ", ".join(f"{lr}: {v:.4f}" for lr, v in c["val_by_lr"].items())
            at_lr = [(k, h) for k, h in tune_hist.items() if h["model"] == m and float(h["lr"]) == c["lr"]]
            fp32 = [h for k, h in at_lr if h.get("fp32") or "_fp32" in k]
            fp16 = [h for k, h in at_lr if not (h.get("fp32") or "_fp32" in k)]
            check = (f"{fp32[0]['best_val']:.4f}, epoch {fp32[0]['best_epoch']}"
                     + (f" ({fp16[0]['best_val']:.4f}, epoch {fp16[0]['best_epoch']})" if fp16 else "")) if fp32 else "not run"
            eps = ", ".join(f"{k.split(':')[0]} {h['best_epoch']}/{h.get('n_trained', h['epochs'])}"
                            for k, h in sorted(hist.items()) if h["model"] == m)
            lines.append(f"| {m} | {tried} | {c['lr']:g} | {check} | {eps} |")
        lines.append("")
    path = os.path.join(RESULTS_SIM, "table_tsf.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
