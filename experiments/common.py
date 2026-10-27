"""Shared data loading, forecast-origin definition and metrics for the NTUA deck-motion experiments.

Protocol (identical for every model):
  * 5 Hz series from data/processed/*_5Hz.csv, 5 s trimmed at both ends (band-pass end effects).
  * Leave-one-run-out (LORO): trained models fit on 3 runs, are tested on the held-out run.
  * Forecast origins every 1 s in the test run, from t = MAX_CONTEXT to the end minus HORIZON,
    so every model is scored on exactly the same targets regardless of its own context length.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

RUNS = ["noWing_headQ", "noWing_followQ", "wing_headQ", "wing_followQ"]
CHANNELS = ["heave_m", "roll_deg", "pitch_deg"]
FS = 5.0              # Hz, as in Wang et al. (2026)
TRIM = int(5 * FS)    # drop 5 s at both ends
MAX_CONTEXT = 300     # 60 s, longest context used by any model
HORIZON = 100         # 20 s ahead (paper's landing window was 19.6 s)
STRIDE = 5            # one forecast origin per second
REPORT_STEPS = {"1 s": 5, "2 s": 10, "5 s": 25, "10 s": 50, "20 s": 100}


def load_run(name):
    """Return (T, 3) array of [heave, roll, pitch] for one run, trimmed."""
    x = np.loadtxt(os.path.join(DATA, f"{name}_5Hz.csv"), delimiter=",", skiprows=1)
    return x[TRIM:-TRIM, 1:4]


def load_all():
    return {r: load_run(r) for r in RUNS}


def origins(n):
    """Forecast origin indices: forecast uses x[:o] and predicts x[o:o+HORIZON]."""
    return np.arange(MAX_CONTEXT, n - HORIZON + 1, STRIDE)


def targets(x):
    o = origins(len(x))
    return np.stack([x[i:i + HORIZON] for i in o])  # (n_origins, HORIZON, 3)


def save_preds(model, preds, quantiles=None, levels=None):
    """preds: {run: (n_origins, HORIZON, 3)}; quantiles: {run: (n_origins, HORIZON, 3, Q)}."""
    out = {f"pred_{r}": p for r, p in preds.items()}
    if quantiles is not None:
        out.update({f"q_{r}": q for r, q in quantiles.items()})
        out["levels"] = np.asarray(levels)
    np.savez_compressed(os.path.join(RESULTS, f"preds_{model}.npz"), **out)


def load_preds(model):
    d = np.load(os.path.join(RESULTS, f"preds_{model}.npz"))
    preds = {r: d[f"pred_{r}"] for r in RUNS}
    q = {r: d[f"q_{r}"] for r in RUNS} if f"q_{RUNS[0]}" in d else None
    levels = d["levels"] if "levels" in d else None
    return preds, q, levels


def rmse_by_step(pred, truth):
    """RMSE per lead step and channel: (HORIZON, 3)."""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))
