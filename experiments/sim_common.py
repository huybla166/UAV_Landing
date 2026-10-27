"""Simulated dataset data/sim/dataset_v1 under the same forecasting protocol as common.py (NTUA).

Same as common.py:
  * 5 Hz, decimated from 20 Hz with scipy.signal.decimate(q=4, zero_phase=True) like data/processed,
  * forecast origins every 1 s (STRIDE), context up to MAX_CONTEXT = 60 s, HORIZON = 20 s,
  * nRMSE = RMSE / std of the run's own (trimmed) series, per channel, then averaged over runs.
Differences, specific to the simulation:
  * the first 30 s are dropped (start-up transient from calm water, run_one.m starts at rest) and the
    last 5 s (decimation edge), so each run has T = 2826 samples at 5 Hz and 486 forecast origins,
  * train/test split is the fixed `split` column of the design matrix (no leave-one-run-out):
    train (1440 runs), test_id (576, rep 3), test_ood_Hs (630, Hs = 0.40 m), test_ood_beta (504, beta = 45 deg).
Channels: heave_m = D (NED, positive DOWN), roll_deg, pitch_deg. Other columns of the CSV
(zeta, tau_w1..6, N, E, psi, u..r, tauX, tauN) are kept, in the units of run_one.m (rad for angles).

Build the cache once (reads the 3150 CSVs, ~750 MB float32):  python experiments/sim_common.py
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import HORIZON, MAX_CONTEXT, STRIDE, REPORT_STEPS, FS  # noqa: F401  (same protocol)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "data", "sim", os.environ.get("DATASET", "dataset_v2"))   # DATASET=dataset_v1 for the old set
CACHE = os.path.join(SIM, "cache_5Hz")
RESULTS_SIM = os.path.join(ROOT, "results", "sim" if os.path.basename(SIM) == "dataset_v2" else "sim_" + os.path.basename(SIM))
os.makedirs(RESULTS_SIM, exist_ok=True)

Q = 4                      # 20 Hz -> 5 Hz
TRIM_START = int(30 * FS)  # start-up transient
TRIM_END = int(5 * FS)     # decimation edge
CSV_COLS = ["zeta", "tau_w1", "tau_w2", "tau_w3", "tau_w4", "tau_w5", "tau_w6", "N", "E", "D",
            "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "tauX", "tauN"]
CHANNELS = ["heave_m", "roll_deg", "pitch_deg"]
SPLITS = ["train", "test_id", "test_ood_Hs", "test_ood_beta"]


def _one(path):
    from scipy import signal
    x = pd.read_csv(path).to_numpy()[:, 1:]                       # drop t
    y = signal.decimate(x, Q, axis=0, zero_phase=True)
    return y[TRIM_START:len(y) - TRIM_END].astype(np.float32)


def build_cache(workers=12):
    man = pd.read_csv(os.path.join(SIM, "manifest.csv")).sort_values("id").reset_index(drop=True)
    paths = [os.path.join(SIM, "runs", n + ".csv") for n in man.name]
    with Pool(workers) as pool:
        arrs = pool.map(_one, paths, chunksize=8)
    X = np.stack(arrs)                                             # (N, T, 21)
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "X.npy"), X)
    man.to_csv(os.path.join(CACHE, "meta.csv"), index=False)
    with open(os.path.join(CACHE, "info.json"), "w") as f:
        json.dump(dict(columns=CSV_COLS, fs=FS, q=Q, trim_start=TRIM_START, trim_end=TRIM_END,
                       shape=list(X.shape)), f, indent=1)
    print(f"cache: {X.shape} float32, {X.nbytes / 1e6:.0f} MB -> {CACHE}")


def load(split=None, mmap=True):
    """meta (DataFrame), X (N, T, 21) float32 in run_one.m units, rows in meta order."""
    meta = pd.read_csv(os.path.join(CACHE, "meta.csv"))
    X = np.load(os.path.join(CACHE, "X.npy"), mmap_mode="r" if mmap else None)
    if split is not None:
        idx = np.nonzero((meta.split == split).to_numpy())[0]
        return meta.iloc[idx].reset_index(drop=True), X[idx]
    return meta, X


def motion(X):
    """(..., T, 21) -> (..., T, 3): heave_m (= D, down positive), roll_deg, pitch_deg."""
    c = CSV_COLS
    return np.stack([X[..., c.index("D")], np.degrees(X[..., c.index("phi")]),
                     np.degrees(X[..., c.index("theta")])], axis=-1).astype(np.float32)


def origins(n):
    """Forecast origin indices: forecast uses x[:o] and predicts x[o:o+HORIZON] (as common.py)."""
    return np.arange(MAX_CONTEXT, n - HORIZON + 1, STRIDE)


if __name__ == "__main__":
    build_cache()
    meta, X = load()
    print(meta.split.value_counts().to_string())
    print(f"origins per run: {len(origins(X.shape[1]))}")
