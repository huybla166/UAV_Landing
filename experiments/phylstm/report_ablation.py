"""Ablation report for PhyLSTM2 on dataset_v2 (evaluation protocol (a)): a second seed of 'phy' / 'data' and the controls
of phylstm_kaggle.py ('je', 'phy_past', 'anchor', 'sobolev', 45 epochs).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/report_ablation.py [--extra DIR ...] [--out FILE]

Reads results/sim/phylstm/phylstm_metrics.npz (kernel 'cv'), the downloaded outputs of the kernels 'seed1', 'controls'
and the phase-2 kernels 'p2-*' (phylstm_metrics.npz, else part_*.npz, in results/sim/phylstm/seed1/, controls/, p2-*/;
--extra replaces these directories) and results/sim/lstm_cv_metrics.npz (run_sim_lstm_cv.py). Writes results/sim/phylstm_ablation.md:
  (a) normalised MSE / MAE at horizons 2, 5, 10, 20 s for every model on each set of splits where it was run
      (normalisation of table_mse_mae.py: errors divided by the train std of each signal in each split, mean over
      runs, origins, leads 0.2..H s and the three signals, then mean over the splits of the set)
  (b) 'phy' and 'data' averaged over seeds 0 and 1, and the per-seed difference phy - data
  (c) paired per-fold differences control - data on cv0..cv2 (the same test runs, the same seed 0)
  (d) 45 epochs against 30 on cv0
  (e) paired differences against 'data' at the same seed and epochs for the phase-2 variants (cummins, seastate,
      phy_seastate, cummins_seastate) and every run with epochs != 30, per set of splits where both were run
A model whose keys are missing on a split of the set is listed as 'pending'. Result tags not in MODELS (new variants,
'_s<seed>', '_e<epochs>') get a row automatically (auto_models), labelled from VARIANT_LABELS + seed + epochs, on the
splits where they were found.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as L  # noqa: E402
from sim_common import FS, K_FOLDS, RESULTS_SIM, load, motion  # noqa: E402

PHY_DIR = os.path.join(RESULTS_SIM, "phylstm")
EXTRA = ([os.path.join(PHY_DIR, "seed1"), os.path.join(PHY_DIR, "controls")]
         + sorted(glob.glob(os.path.join(PHY_DIR, "p2-*"))))     # downloaded outputs of the phase-2 kernels p2-*
OUT = os.path.join(RESULTS_SIM, "phylstm_ablation.md")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(RESULTS_SIM)))
HORIZONS = [2, 5, 10, 20]
CV = [f"cv{k}" for k in range(K_FOLDS)]
C3 = ["cv0", "cv1", "cv2"]
ALL = CV + ["ood_Hs", "ood_beta"]
# tag -> (label, splits on which it is run)
MODELS = {
    "lstm": ("LSTM [Wang 2026], mean of 3 seeds, last epoch", ALL),
    "data": ("PhyLSTM w/o physics (data)", ALL), "phy": ("PhyLSTM2 (phy)", ALL),
    "data_s1": ("data, seed 1", ALL), "phy_s1": ("phy, seed 1", ALL),
    "je": ("J_d + J_e only (je)", C3), "phy_past": ("phy, J_g on the past only (phy_past)", C3),
    "anchor": ("data + increment on last observation (anchor)", C3),
    "sobolev": ("data + smoothness of error (sobolev)", C3),
    "data_e45": ("data, 45 epochs", ["cv0"]), "phy_e45": ("phy, 45 epochs", ["cv0"]),
}
CONTROLS = ["je", "phy_past", "anchor", "sobolev", "phy"]
# variant -> label of its automatic rows (tags 'variant[_s<seed>][_e<epochs>]' of phylstm_kaggle.result_tag)
VARIANT_LABELS = {
    "data": "PhyLSTM w/o physics (data)", "phy": "PhyLSTM2 (phy)", "je": "J_d + J_e only (je)",
    "phy_past": "phy, J_g on the past only (phy_past)", "anchor": "data + increment on last observation (anchor)",
    "sobolev": "data + smoothness of error (sobolev)",
    "cummins": "Cummins PINN: J_d(eta, w/p/q) + J_kin + J_dyn (cummins)",
    "seastate": "data + sea-state head (seastate)", "phy_seastate": "phy + sea-state head (phy_seastate)",
    "cummins_seastate": "cummins + sea-state head (cummins_seastate)",
}
PHASE2 = ["cummins", "seastate", "phy_seastate", "cummins_seastate"]
TAG_RE = re.compile(r"^(?P<v>.+?)(?:_s(?P<s>\d+))?(?:_e(?P<e>\d+))?$")


def parse_tag(tag):
    """'phy_seastate_s1_e45' -> ('phy_seastate', 1, 45); default seed 0, epochs 30."""
    m = TAG_RE.match(tag)
    return m["v"], int(m["s"] or 0), int(m["e"] or 30)


def auto_label(tag):
    v, seed, ep = parse_tag(tag)
    epochs = f", {ep} epoch{'s' if ep > 1 else ''}" if ep != 30 else ""
    return VARIANT_LABELS.get(v, v) + (f", seed {seed}" if seed else "") + epochs


def auto_models(res):
    """MODELS + a row for every result tag found in res that MODELS lacks, planned on the splits where it was found."""
    models = dict(MODELS)
    tags = sorted({k.split("|")[1] for k in res if k.count("|") == 2 and k.endswith("|mse")} - set(MODELS))
    for t in tags:
        if t.endswith("_last"):
            continue
        models[t] = (auto_label(t), [s for s in ALL if f"{s}|{t}|mse" in res])
    return models
SETS = [("6-fold CV (cv0-cv5)", CV), ("cv0-cv2 (the folds of the controls)", C3),
        ("cv0 only (45-epoch pilots and other cv0-only runs)", ["cv0"]),
        ("OOD Hs = 0.40 m (ood_Hs)", ["ood_Hs"]), ("OOD beta = 45 deg (ood_beta)", ["ood_beta"])]


def rel(f):
    """Path relative to the repository (absolute if elsewhere), with '/'."""
    f = os.path.abspath(f)
    try:
        r = os.path.relpath(f, ROOT)
    except ValueError:                                   # another drive (Windows)
        r = f
    return (f if r.startswith("..") else r).replace(os.sep, "/")


def load_phy(extra):
    """Merged result dict of every PhyLSTM file, and the list of files read."""
    files = [os.path.join(PHY_DIR, "phylstm_metrics.npz")]
    for d in extra:
        m = os.path.join(d, "phylstm_metrics.npz")
        files += [m] if os.path.exists(m) else sorted(glob.glob(os.path.join(d, "part_*.npz")))
    res, used = {}, []
    for f in files:
        if not os.path.exists(f):
            continue
        new = dict(np.load(f))
        for k, v in new.items():
            if k in res and not np.array_equal(res[k], v):
                raise SystemExit(f"{k} differs between {f} and an earlier file")
        res.update(new)
        used.append((f, len(new)))
    return res, used


def norm_metric(res, lstm, sd, tag, split, H):
    """(normalised MSE, MAE) of model `tag` on one split over leads 0.2..H s, or None if missing."""
    n = int(H * FS)
    if tag.startswith("lstm"):
        keys = [f"{split}|seed{s}|{{}}" for s in L.SEEDS]
        if not all(k.format("mse") in lstm for k in keys):
            return None
        m2 = np.mean([lstm[k.format("mse")] for k in keys], axis=0)
        m1 = np.mean([lstm[k.format("mae")] for k in keys], axis=0)
    else:
        if f"{split}|{tag}|mse" not in res:
            return None
        m2, m1 = res[f"{split}|{tag}|mse"], res[f"{split}|{tag}|mae"]
    return (m2[:, :n] / sd[split] ** 2).mean(), (m1[:, :n] / sd[split]).mean()


def per_split(res, lstm, sd, tag, splits):
    """{H: (R splits, 2) array of (MSE, MAE)}, or the list of splits where `tag` is missing."""
    vals = {H: [norm_metric(res, lstm, sd, tag, s, H) for s in splits] for H in HORIZONS}
    missing = [s for s, v in zip(splits, vals[HORIZONS[0]]) if v is None]
    return missing if missing else {H: np.array(v) for H, v in vals.items()}


def pending(missing, n_cols):
    return f"pending (missing: {', '.join(missing)})" + " | " * (n_cols - 1)


def table_a(res, lstm, sd, models=MODELS):
    lines = ["## (a) Normalised MSE and MAE per horizon", "",
             "Mean over the splits of each set; z-score normalisation with the train std of each split (as "
             "`table_mse_mae.py`). Rows without a suffix: the epoch with the lowest validation error (PhyLSTM, as the "
             "paper); 'last epoch': the final weights (as the LSTM baseline). A model appears in every set whose splits "
             "it was run on.", ""]
    head = "| Model | " + " | ".join(f"{H} s MSE | {H} s MAE" for H in HORIZONS) + " |"
    for title, splits in SETS:
        lines += [f"### {title}", "", head, "|---|" + "---|" * (2 * len(HORIZONS))]
        for tag, (label, planned) in models.items():
            if not set(splits) <= set(planned):
                continue
            for suf, extra in ([("", "")] if tag == "lstm" else [("", ""), ("_last", ", last epoch")]):
                got = per_split(res, lstm, sd, tag + suf, splits)
                if isinstance(got, list):
                    lines.append(f"| {label}{extra} | " + pending(got, 2 * len(HORIZONS)) + " |")
                    continue
                cells = [f"{got[H][:, 0].mean():.4f} | {got[H][:, 1].mean():.4f}" for H in HORIZONS]
                lines.append(f"| {label}{extra} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def table_b(res, lstm, sd):
    lines = ["## (b) Two seeds of phy and data", "",
             "Seed 0: kernel 'cv'; seed 1: kernel 'seed1' (same splits, data and code; only the seed of the weights, the "
             "training origins and the validation origins differs). 'Mean' = mean of the two seeds' normalised MSE. "
             "'phy - data' < 0: PhyLSTM2 better; 'folds' = number of splits of the set where phy < data.", ""]
    head = "| Set | Row | " + " | ".join(f"{H} s MSE" for H in HORIZONS) + " |"
    for suf, title in (("", "best-on-validation epoch"), ("_last", "last epoch")):
        lines += [f"### {title}", "", head, "|---|---|" + "---|" * len(HORIZONS)]
        for set_title, splits in [SETS[0], SETS[3], SETS[4]]:
            got = {(v, s): per_split(res, lstm, sd, v + ("_s1" if s else "") + suf, splits)
                   for v in ("phy", "data") for s in (0, 1)}
            for v in ("data", "phy"):
                for s in (0, 1):
                    g = got[v, s]
                    lines.append(f"| {set_title} | {v}, seed {s} | " + (pending(g, len(HORIZONS)) if isinstance(g, list)
                                 else " | ".join(f"{g[H][:, 0].mean():.4f}" for H in HORIZONS)) + " |")
                if any(isinstance(got[v, s], list) for s in (0, 1)):
                    lines.append(f"| {set_title} | {v}, mean of seeds 0, 1 | " + pending(["seed 1"], len(HORIZONS)) + " |")
                else:
                    lines.append(f"| {set_title} | **{v}, mean of seeds 0, 1** | " + " | ".join(
                        f"**{np.mean([got[v, s][H][:, 0].mean() for s in (0, 1)]):.4f}**" for H in HORIZONS) + " |")
            for s in (0, 1, "mean"):
                ss = (0, 1) if s == "mean" else (s,)
                if any(isinstance(got[v, x], list) for v in ("phy", "data") for x in ss):
                    lines.append(f"| {set_title} | phy - data, seed {s} | " + pending(["seed 1"], len(HORIZONS)) + " |")
                    continue
                cells = []
                for H in HORIZONS:
                    d = np.mean([got["phy", x][H][:, 0] - got["data", x][H][:, 0] for x in ss], axis=0)  # per split
                    cells.append(f"{d.mean():+.4f} (folds {(d < 0).sum()}/{len(d)})" if len(d) > 1 else f"{d.mean():+.4f}")
                lines.append(f"| {set_title} | phy - data, seed {s} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def table_c(res, lstm, sd):
    lines = ["## (c) Paired differences against data on cv0-cv2 (seed 0)", "",
             "Delta = normalised MSE of the model - normalised MSE of 'data' on the same fold (same test runs, same seed "
             "0, same Network 1 and training budget). < 0: the model is better than 'data'. 'rel.' = mean of Delta / "
             "data. 'phy' is the reference: the full PhyLSTM2 with J_g on the whole window (true future wave force in the "
             "training loss).", ""]
    head = "| Model | Horizon | Delta cv0 | Delta cv1 | Delta cv2 | mean Delta | rel. | folds better |"
    for suf, title in (("", "best-on-validation epoch"), ("_last", "last epoch")):
        lines += [f"### {title}", "", head, "|---|---|---|---|---|---|---|---|"]
        base = per_split(res, lstm, sd, "data" + suf, C3)
        for tag in CONTROLS:
            got = per_split(res, lstm, sd, tag + suf, C3)
            for H in HORIZONS:
                miss = (got if isinstance(got, list) else []) + (base if isinstance(base, list) else [])
                if miss:
                    lines.append(f"| {tag}{suf} | {H} s | " + pending(sorted(set(miss)), 6) + " |")
                    continue
                d = got[H][:, 0] - base[H][:, 0]
                lines.append(f"| {tag}{suf} | {H} s | " + " | ".join(f"{x:+.4f}" for x in d)
                             + f" | {d.mean():+.4f} | {100 * (d / base[H][:, 0]).mean():+.1f} % | {(d < 0).sum()}/3 |")
        lines.append("")
    return lines


def table_d(res, lstm, sd):
    lines = ["## (d) 45 epochs against 30 on cv0 (seed 0)", "",
             "One fold only: differences of normalised MSE on cv0 (< 0: the first model is better). With 45 epochs the "
             "learning rate drops to 1e-4 after epoch 30 instead of epoch 20.", "",
             "| Comparison | " + " | ".join(f"{H} s" for H in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    for a, b in (("data_e45", "data"), ("phy_e45", "phy"), ("phy", "data"), ("phy_e45", "data_e45")):
        for suf in ("", "_last"):
            ga, gb = per_split(res, lstm, sd, a + suf, ["cv0"]), per_split(res, lstm, sd, b + suf, ["cv0"])
            miss = (ga if isinstance(ga, list) else []) + (gb if isinstance(gb, list) else [])
            name = f"{a}{suf} - {b}{suf}"
            if miss:
                names = [t + suf for t, g in ((a, ga), (b, gb)) if isinstance(g, list)]
                lines.append(f"| {name} | " + pending(names, len(HORIZONS)) + " |")
                continue
            lines.append(f"| {name} | " + " | ".join(f"{ga[H][0, 0] - gb[H][0, 0]:+.4f}" for H in HORIZONS) + " |")
    lines.append("")
    return lines


def table_e(res, lstm, sd, models):
    lines = ["## (e) Phase 2 and longer training: paired differences against data (same seed, same epochs)", "",
             "Delta = normalised MSE of the model - normalised MSE of 'data' with the same seed and number of epochs, "
             "on the same split (same test runs), averaged over the splits of the set; < 0: the model is better. "
             "'rel.' = mean of Delta / data; 'folds' = splits of the set where the model is better. Rows: the phase-2 "
             "variants and every run with epochs != 30, on each set where both the model and its 'data' were run.", "",
             "| Model | Set | Reference | " + " | ".join(f"{H} s Delta (rel.; folds)" for H in HORIZONS) + " |",
             "|---|---|---|" + "---|" * len(HORIZONS)]
    n = 0
    for tag in models:
        v, seed, ep = parse_tag(tag)
        if v == "data" or v not in VARIANT_LABELS or (v not in PHASE2 and ep == 30):
            continue
        ref = "data" + (f"_s{seed}" if seed else "") + (f"_e{ep}" if ep != 30 else "")
        for suf, extra in (("", ""), ("_last", ", last epoch")):
            full_cv = False
            for set_title, splits in [SETS[0], SETS[2], SETS[3], SETS[4]]:
                if splits == ["cv0"] and full_cv:
                    continue                                  # cv0 alone only for runs not on all six folds (pilots)
                got, base = per_split(res, lstm, sd, tag + suf, splits), per_split(res, lstm, sd, ref + suf, splits)
                if isinstance(got, list) or isinstance(base, list):
                    continue                                  # not run on this set (table (a) lists it as pending)
                full_cv = full_cv or splits == CV
                cells = []
                for H in HORIZONS:
                    d = got[H][:, 0] - base[H][:, 0]
                    cells.append(f"{d.mean():+.4f} ({100 * (d / base[H][:, 0]).mean():+.1f} %; {(d < 0).sum()}/{len(d)})")
                lines.append(f"| {models[tag][0]}{extra} | {set_title} | {ref}{suf} | " + " | ".join(cells) + " |")
                n += 1
    if not n:
        lines.append("| (no phase-2 or longer runs with a matching 'data' run yet) | | |" + " |" * len(HORIZONS))
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=EXTRA, help="directories with the outputs of the extra kernels")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    res, used = load_phy(a.extra)
    lstm = dict(np.load(L.METRICS))
    for s in ALL:                                          # the same test runs, in the same order
        if f"{s}|ids" in res:
            assert np.array_equal(res[f"{s}|ids"], lstm[f"{s}|ids"]), f"test runs differ on {s}"
    meta, X = load()
    Y = motion(X)
    sd = {s: Y[L.split_rows(meta, s)["train"]].std(axis=(0, 1), dtype=np.float64) for s in ALL}           # (3,) per split
    found = sorted({k.split("|")[1] for k in res if k.count("|") == 2})
    models = auto_models(res)
    lines = ["# PhyLSTM2 ablation: second seed and controls (dataset_v2, evaluation protocol (a))", "",
             "Every model forecasts 20 s from the past 60 s of heave, roll and pitch only; none sees future waves when it "
             "forecasts. Variants (experiments/phylstm/phylstm_kaggle.py): 'data' = Network 1 trained on J_d; 'phy' = "
             "PhyLSTM2 (J_d + J_e + J_g, J_g with the true wave force on the whole 80 s window, future 20 s included); "
             "'je' = J_d + J_e; 'phy_past' = 'phy' with J_g on the 60 s past only (no future wave force anywhere); "
             "'anchor' = 'data' forecasting increments from the last observation; 'sobolev' = 'data' + a physics-free "
             "penalty on the first and second time differences of the error. Phase 2: 'cummins' = Network 1 "
             "predicting eta and the body velocities w, p, q, trained on J_d + J_kin (kinematics) + J_dyn (linearised "
             "Cummins equation of the run's vessel/speed group, subtraction form; no wave force); '*_seastate' = the "
             "same base + a training-only head on the last past hidden state predicting Hs, Tp, beta, U. "
             "'_s<n>' = seed n, '_e<n>' = n epochs (default seed 0, 30 epochs). Script "
             "`experiments/phylstm/report_ablation.py`.", "",
             "Files read: " + "; ".join(f"`{rel(f)}` ({n} arrays)" for f, n in used)
             + f"; `{rel(L.METRICS)}`. PhyLSTM result tags found: "
             + ", ".join(found) + ".", ""]
    trunc = sorted(f"{k.rsplit('|', 1)[0]} ({res[k][0]}/{res[k][1]} epochs)" for k in res
                   if k.endswith("|n_epochs") and res[k][0] < res[k][1])
    if trunc:
        lines += ["**Truncated runs (stopped by the time limit before the planned epochs):** " + ", ".join(trunc) + ".", ""]
    lines += (table_a(res, lstm, sd, models) + table_b(res, lstm, sd) + table_c(res, lstm, sd) + table_d(res, lstm, sd)
              + table_e(res, lstm, sd, models))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
