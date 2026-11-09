"""Optimal-linear (Wiener) diagnostics for the 20 s motion forecast on dataset_v2, evaluation protocol (a).

  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/wiener.py                       # all 8 splits, all methods
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/wiener.py --splits cv0 --max-runs 60
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/wiener.py study                 # design study (L, uni/multi, ridge, estimator)
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/wiener.py study-sib             # lag window of wiener_sib (M = 100 vs >= L + H)
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/wiener.py table [--npz FILE]    # normalised MSE/MAE tables

Deterministic point forecasts (conditional mean only, no variances) of heave D (m), roll, pitch (deg) for x[o:o+100]
given x[:o], at every forecast origin o of sim_common.origins(2826) of every test run. For a zero-mean stationary
process with autocovariance G(h) = E[y_{t+h} y_t^T] (3x3), the linear minimum-MSE forecast of the next H = 100 samples
from the last L samples p is  f_hat = C_fp C_pp^-1 p  (joint Gaussian conditional mean), with C_pp, C_fp block-Toeplitz
built from G(-(L+H-1)) .. G(L+H-1). A relative ridge (ridge x the channel's variance on the diagonal of C_pp) keeps
C_pp invertible: the 5 Hz series is low-pass (decimated from 20 Hz), so its spectrum is ~0 near Nyquist.

Why the run's own autocovariance is not a bound. MSS draws each wave frequency once per seed at random inside its bin
(D:/tools/MSS/LIBRARY/environment/waveDirectionalSpectrum.m lines 116-124: Omega(i) = omega_mid + (rand - 0.5) dOmega),
so the wave elevation of a run is a sum of 100 fixed sinusoids (x spreading directions). The sample autocovariance of
the run's OWN full record carries exactly these discrete lines (it does not decay with lag), and a long-past predictor
built from it extrapolates them far ahead. A real sea has a continuous spectrum; its autocovariance (the ensemble over
seeds) decays, and the lines of one realisation are unknowable from another. `study-sib` shows it: the heave
autocorrelation of a run at lags >= 20 s is uncorrelated with that of its siblings (same condition, other seeds).

Methods (the mean is subtracted before predicting and added back). All use the joint heave/roll/pitch predictor.
  wiener_own_lines  LINE-SPECTRUM ARTEFACT ORACLE, excluded from conclusions. G from the test run's own full series
               (all 2826 samples, biased estimator, no lag window), run mean, L = 300, ridge 1e-6. It knows this run's
               discrete wave lines (see above) and uses its future; its long-lead skill (normalised MSE 0.478 at
               exactly 20 s, first 40 cv0 test runs) is an artefact of the simulator's line spectrum, not a bound for a real sea.
  wiener_sib   Sea-state reference (not a bound, not legitimate either: it uses the condition's other runs). G =
               average of the biased G of the OTHER 'ok' runs of the identical condition (vessel, Hs, Tp, beta_deg, U,
               kappa4, dzeta3; different wave seed), Parzen lag window M = 500 >= L + H = 400 so the window does not
               truncate correlation at the forecast leads (weight >= 0.81 at every lag < 100; window M = 100 is zero
               from lag 100 = 20 s on and forces the long-lead forecast to the mean), mean = average of the siblings'
               run means (not the test run's), L = 300, ridge 1e-6. Runs without siblings (448 'ok' conditions with a
               single run, rep 1 only) get NaN and are excluded from every row of the
               tables that show wiener_sib; '<split>|n_sib' = number of siblings per test run. study_wiener_sib.txt
               (54 cv0 validation runs): every M above 100 is worse at every lead, and from M = 400 on the 20 s error
               exceeds the mean forecast; pooling 2 siblings instead of 1 helps a lot at large M, and the heave
               autocorrelation at lag 20 s correlates only +0.12 across siblings: with 1-2 siblings the long-lag G is
               dominated by the siblings' own lines, so wiener_sib is an under-estimated reference, not a bound.
  wiener_sib_m100  the same with Parzen M = 100 (the setting tuned on the cv0 validation fold in `study`). Its
               long-lead reversion to the mean is by construction of the window; at 1-2 s it is the stronger of the
               two (study_wiener_sib.txt), so both are reported.
  wiener_past  LEGITIMATE. Per origin, G from the past 60 s x[o-300:o] only (window mean removed, biased estimator =
               PSD, Parzen M = 100), predictor from the last L = 10 samples, ridge 1e-4. The 60 s window limits G to
               lags < 300 and the Parzen window to lags < 100 (20 s), so its reversion to the window mean at long
               leads is partly by construction (at exactly 20 s it equals mean_past).
  mean_past    reference: the mean of the past 60 s (the forecast every windowed predictor decays to).

Choices (univariate vs joint, L, ridge, estimator, lag window) come from `study` on 24 runs of the cv0 VALIDATION fold
(not the cv0 test runs used for validation), written to results/sim/diagnostics/study_wiener.txt (written by this
script before the rename: 'own' = wiener_own_lines, 'cond' = wiener_sib with the lag window given on the line);
the wiener_sib lag window by `study-sib` -> results/sim/diagnostics/study_wiener_sib.txt.

Output results/sim/diagnostics/wiener_metrics.npz (with --max-runs N: wiener_metrics_max<N>.npz), same layout as
run_sim_lstm_cv.py: '<split>|<method>|mse' and '|mae' = per test run mean over origins of e^2 and |e| per lead
(R, 100, 3), '|mse_o' = per run per origin MSE over the 20 s (R, 486, 3), errors in m / deg; '<split>|ids' = test run
ids, asserted equal to results/sim/lstm_cv_metrics.npz; '<split>|n_sib' (R,); 'cfg|<method>' = the method's settings
(a method is recomputed when they change, a split when its ids change). Tables (same normalisation as
experiments/phylstm/table_mse_mae.py: error / train std of the split per channel, mean over runs, leads <= H s and
channels; cv folds averaged) -> wiener_tables[_max<N>].md: every row of a table on the SAME runs (all test runs
without wiener_sib; the runs with siblings with it), including the paper LSTM (mean of 3 seeds), PhyLSTM w/o physics
and PhyLSTM2 restricted to those run ids.
"""
import argparse
import os
import sys
import time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"                # many small batched solves: multithreaded BLAS was ~70x slower (2.3 s vs 165 s/run)
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sim_lstm_cv as LCV  # noqa: E402  (split_rows, SPLITS, METRICS)
from sim_common import FS, HORIZON, MAX_CONTEXT, RESULTS_SIM, load, motion, origins  # noqa: E402

OUT = os.path.join(RESULTS_SIM, "diagnostics")
COND = ["vessel", "Hs", "Tp", "beta_deg", "U", "kappa4", "dzeta3"]
H = HORIZON
WIN = MAX_CONTEXT                       # 60 s of past for wiener_past
HORIZONS = [1, 2, 5, 10, 20]
L_MAX = 300                             # longest past used by own / sib (autocovariances kept to lag L_MAX+H-1)
ESTIMATORS = ("biased", "tukey", "unbiased_pd")
# defaults chosen by `study` / `study-sib` (results/sim/diagnostics/study_wiener*.txt, cv0 validation-fold runs)
CFG = {
    "wiener_own_lines": dict(L=300, ridge=1e-6, lagwin=None, uni=False),
    "wiener_sib": dict(L=300, ridge=1e-6, lagwin=500, uni=False, mean="sib"),
    "wiener_sib_m100": dict(L=300, ridge=1e-6, lagwin=100, uni=False, mean="sib"),
    "wiener_past": dict(L=10, ridge=1e-4, lagwin=100, uni=False, est="biased"),
    "mean_past": dict(),
}
METHODS = list(CFG)
SIB_METHODS = ("wiener_sib", "wiener_sib_m100")
assert CFG["wiener_sib"]["lagwin"] >= CFG["wiener_sib"]["L"] + H
LABELS = {
    "wiener_own_lines": "wiener_own_lines (ARTEFACT oracle: own run's lines + future; excluded)",
    "wiener_sib": "wiener_sib (siblings' spectrum, M=500; not legitimate)",
    "wiener_sib_m100": "wiener_sib_m100 (siblings' spectrum, M=100; not legitimate)",
    "wiener_past": "wiener_past (legitimate, past 60 s)",
    "mean_past": "mean_past (mean of past 60 s)",
}
REFS = {"lstm": "LSTM [Wang 2026] (mean of 3 seeds)", "data": "PhyLSTM w/o physics", "phy": "PhyLSTM2"}


# ---------------------------------------------------------------- autocovariance
def acov(Y, maxlag, est="biased", alpha=0.5):
    """Y (..., N, 3), already demeaned -> G (..., maxlag+1, 3, 3) with G[h][a, b] = mean_t y[t+h, a] y[t, b].
    biased: / N (positive semi-definite); tukey: Tukey(alpha) data taper, / sum(w^2) (PSD, less leakage);
    unbiased_pd: / (N - h), then made PSD by clipping negative eigenvalues of the block-circulant embedding."""
    N = Y.shape[-2]
    Y = np.asarray(Y, np.float64)
    norm = np.full(maxlag + 1, float(N))
    if est == "tukey":
        from scipy.signal.windows import tukey
        w = tukey(N, alpha)
        Y = Y * w[:, None]
        norm[:] = (w ** 2).sum()
    elif est == "unbiased_pd":
        norm = N - np.arange(maxlag + 1, dtype=float)
    nfft = 1 << int(np.ceil(np.log2(N + maxlag + 1)))
    F = np.fft.rfft(Y, n=nfft, axis=-2)                                             # (..., nf, 3)
    S = F[..., :, None] * F[..., None, :].conj()                                    # F_a conj(F_b)
    G = np.fft.irfft(S, n=nfft, axis=-3)[..., :maxlag + 1, :, :] / norm[:, None, None]
    return pd_fix(G) if est == "unbiased_pd" else G


def parzen_w(K, M):
    """Parzen lag window of half-width M at lags 0..K-1 (all ones for M None)."""
    if not M:
        return np.ones(K)
    u = np.arange(K) / M
    return np.where(u <= 0.5, 1 - 6 * u ** 2 + 6 * u ** 3, np.where(u <= 1, 2 * (1 - u) ** 3, 0.0))


def parzen(G, M):
    """Parzen lag window of half-width M on G (..., K, 3, 3) (M None: unchanged). Its spectral window is non-negative,
    so a PSD sequence stays PSD; it smooths the spectrum over ~1/M cycles per sample and is 0 from lag M on."""
    return G if not M else G * parzen_w(G.shape[-3], M)[:, None, None]


def pd_fix(G):
    """Nearest-PSD-in-spectrum fix: embed lags 0..K-1 (and G(-h) = G(h)^T) in a block circulant, clip negative
    eigenvalues of every 3x3 spectral matrix to 0, transform back. Every block-Toeplitz section of size <= K is PSD."""
    K = G.shape[-3]
    n = 1 << int(np.ceil(np.log2(2 * K - 1)))
    c = np.zeros(G.shape[:-3] + (n, 3, 3))
    c[..., :K, :, :] = G
    c[..., n - K + 1:, :, :] = np.swapaxes(G[..., 1:, :, :], -1, -2)[..., ::-1, :, :]
    S = np.fft.fft(c, axis=-3)
    S = 0.5 * (S + np.conj(np.swapaxes(S, -1, -2)))
    w, V = np.linalg.eigh(S)
    S = (V * np.clip(w, 0, None)[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    return np.fft.ifft(S, axis=-3).real[..., :K, :, :]


# ---------------------------------------------------------------- Wiener predictor
def cov_blocks(G, L, uni=False):
    """G (..., >= L+H, 3, 3) -> C_pp (..., 3L, 3L), C_pf (..., 3L, 3H) for p = y[o-L:o].reshape(-1) (time-major) and
    f = y[o:o+H].reshape(-1); relative ridge not yet added."""
    K = L + H
    G = G[..., :K, :, :]
    if uni:
        G = G * np.eye(3)
    Gf = np.concatenate([np.swapaxes(G[..., 1:, :, :], -1, -2)[..., ::-1, :, :], G], axis=-3)   # lags -(K-1)..K-1
    tp, tf = np.arange(-L, 0), np.arange(H)
    lead = G.shape[:-3]
    Cpp = Gf[..., tp[:, None] - tp[None, :] + K - 1, :, :]                          # (..., L, L, 3, 3)
    Cpp = np.swapaxes(Cpp, -3, -2).reshape(lead + (3 * L, 3 * L))
    Cpf = Gf[..., tp[:, None] - tf[None, :] + K - 1, :, :]                          # (..., L, H, 3, 3)
    Cpf = np.swapaxes(Cpf, -3, -2).reshape(lead + (3 * L, 3 * H))
    return Cpp, Cpf


def add_ridge(Cpp, G, L, ridge):
    """C_pp + ridge x (variance of each channel) on the diagonal, in place."""
    var = np.diagonal(G[..., 0, :, :], axis1=-2, axis2=-1)                         # (..., 3)
    idx = np.arange(3 * L)
    Cpp[..., idx, idx] += ridge * np.tile(var, L)
    return Cpp


def weights(G, L, ridge, uni=False):
    """G (..., >= L+H, 3, 3) -> X (..., 3L, 3H) with f_hat = p @ X = (C_fp C_pp^-1 p)^T: the linear minimum-MSE
    predictor for the autocovariance G."""
    Cpp, Cpf = cov_blocks(G, L, uni)
    return np.linalg.solve(add_ridge(Cpp, G, L, ridge), Cpf)


def windows(y, starts, n):
    """y (T, 3) -> (len(starts), n, 3) = y[s:s+n] for each start s (a view)."""
    return np.lib.stride_tricks.sliding_window_view(y, n, axis=0)[starts].transpose(0, 2, 1)


def errors(pred, y, o):
    """pred (O, H, 3) -> per-run mse, mae per lead (H, 3) and mse over the horizon per origin (O, 3)."""
    e = pred - windows(y, o, H)
    return (e ** 2).mean(0), np.abs(e).mean(0), (e ** 2).mean(1)


def predict_fixed(y, G, mu, L, ridge, uni=False):
    """One predictor for every origin of the run y (T, 3): G (L+H, 3, 3) and mean mu (3,) given."""
    o = origins(len(y))
    X = weights(G, L, ridge, uni)
    P = (windows(y, o - L, L) - mu).reshape(len(o), -1)
    return (P @ X).reshape(len(o), H, 3) + mu


def predict_past(y, L, ridge, lagwin, uni=False, est="biased", chunk=96):
    """Per origin: G and mean from y[o-WIN:o] only, then the predictor from the last L samples."""
    o = origins(len(y))
    W = np.asarray(windows(y, o - WIN, WIN), np.float64)                            # (O, WIN, 3)
    mu = W.mean(1, keepdims=True)
    W = W - mu
    G = parzen(acov(W, L + H - 1, est), lagwin)                                     # (O, L+H, 3, 3)
    pred = np.empty((len(o), H, 3))
    for i in range(0, len(o), chunk):
        Cpp, Cpf = cov_blocks(G[i:i + chunk], L, uni)                                # (c, 3L, 3L), (c, 3L, 3H)
        P = W[i:i + chunk, -L:].reshape(len(Cpp), -1, 1)
        a = np.linalg.solve(add_ridge(Cpp, G[i:i + chunk], L, ridge), P)            # C_pp^-1 p, one right-hand side
        pred[i:i + chunk] = (np.swapaxes(a, -1, -2) @ Cpf).reshape(len(Cpp), H, 3)
    return pred + mu


# ---------------------------------------------------------------- data
class Data:
    """motion Y (N, T, 3) float64 of every run, per-run mean and biased autocovariance (up to lag Kmax-1),
    sibling lists (other 'ok' runs of the identical condition) for wiener_sib."""

    def __init__(self, kmax=L_MAX + H):
        self.meta, X = load()
        self.Y = motion(X)                                                          # float32 (N, T, 3), m / deg
        self.kmax = kmax
        self._G, self._mu = {}, {}
        ok = (self.meta.status == "ok").to_numpy()
        key = self.meta[COND].astype(str).agg("|".join, axis=1).to_numpy()
        self.sib = {}
        groups = {}
        for i in np.nonzero(ok)[0]:
            groups.setdefault(key[i], []).append(i)
        for g in groups.values():
            for i in g:
                self.sib[i] = [j for j in g if j != i]
        sizes = np.bincount([len(g) for g in groups.values()])
        self.group_sizes = {k: int(v) for k, v in enumerate(sizes) if v}          # {runs per condition: conditions}

    def stats(self, i):
        if i not in self._G:
            y = np.asarray(self.Y[i], np.float64)
            self._mu[i] = y.mean(0)
            self._G[i] = acov(y - self._mu[i], self.kmax - 1)
        return self._G[i], self._mu[i]

    def train_sd(self, split):
        rows = LCV.split_rows(self.meta, split)["train"]
        return self.Y[rows].std(axis=(0, 1), dtype=np.float64)


def run_method(d, i, method, sib=None, **kw):
    """Forecasts (O, H, 3) of run row i, or None (wiener_sib* without sibling). kw overrides CFG[method];
    sib overrides the sibling list (study-sib)."""
    c = {**CFG[method], **kw}
    y = np.asarray(d.Y[i], np.float64)
    if method == "wiener_own_lines":
        G, mu = d.stats(i)
        return predict_fixed(y, parzen(G, c["lagwin"]), mu, c["L"], c["ridge"], c["uni"])
    if method in SIB_METHODS:
        sib = d.sib.get(i, []) if sib is None else sib
        if not sib:
            return None
        st = [d.stats(j) for j in sib]
        G = np.mean([s[0] for s in st], axis=0)
        mu = np.mean([s[1] for s in st], axis=0) if c["mean"] == "sib" else y.mean(0)
        return predict_fixed(y, parzen(G, c["lagwin"]), mu, c["L"], c["ridge"], c["uni"])
    if method == "wiener_past":
        return predict_past(y, **c)
    if method == "mean_past":                                                       # reference: mean of the past 60 s
        o = origins(len(y))
        return np.repeat(windows(y, o - WIN, WIN).mean(1)[:, None], H, axis=1)
    raise ValueError(method)


def score_rows(d, rows, method, progress=True, **kw):
    R, O = len(rows), len(origins(d.Y.shape[1]))
    mse, mae = np.full((R, H, 3), np.nan, np.float32), np.full((R, H, 3), np.nan, np.float32)
    mse_o = np.full((R, O, 3), np.nan, np.float32)
    t0 = time.time()
    for r, i in enumerate(rows):
        pred = run_method(d, i, method, **kw)
        if pred is not None:
            mse[r], mae[r], mse_o[r] = errors(pred, np.asarray(d.Y[i], np.float64), origins(d.Y.shape[1]))
        if progress and ((r + 1) % 100 == 0 or r + 1 == R):
            el = time.time() - t0
            print(f"      {method}: {r + 1}/{R} runs, {el:.0f} s, eta {el / (r + 1) * (R - r - 1):.0f} s", flush=True)
    return mse, mae, mse_o


# ---------------------------------------------------------------- normalised tables
def norm_scores(mse, mae, sd):
    """-> {h: (normalised MSE, MAE)} for h in HORIZONS (mean over leads <= h) + lead 0.2 s under key 0.2.
    Plain means: callers pass only rows without NaN."""
    out = {}
    for h in [0.2] + HORIZONS:
        n = int(round(h * FS))
        out[h] = (float(np.mean(mse[:, :n] / sd ** 2)), float(np.mean(mae[:, :n] / sd)))
    return out


def reference(name):
    """Earlier models' metrics: 'lstm' (mean of 3 seeds' mse/mae, as table_mse_mae.py), 'phy' (PhyLSTM2),
    'data' (PhyLSTM w/o physics). -> function split -> (ids, mse, mae) or None if the split is missing."""
    if name == "lstm":
        r = np.load(LCV.METRICS)
        def get(s):
            if f"{s}|seed0|mse" not in r.files:
                return None
            return (r[f"{s}|ids"], np.mean([r[f"{s}|seed{k}|mse"] for k in LCV.SEEDS], 0),
                    np.mean([r[f"{s}|seed{k}|mae"] for k in LCV.SEEDS], 0))
        return get
    r = np.load(os.path.join(RESULTS_SIM, "phylstm", "phylstm_metrics.npz"))
    def get(s):
        if f"{s}|{name}|mse" not in r.files:
            return None
        return r[f"{s}|ids"], r[f"{s}|{name}|mse"], r[f"{s}|{name}|mae"]
    return get


def print_table(res, d, splits, sib_only, methods=METHODS, refs=tuple(REFS)):
    """Normalised MSE / MAE per method at the HORIZONS (and MSE at lead 0.2 s), mean over `splits`, every row on the
    SAME runs: all test runs in res (wiener_sib* rows left out, they are NaN without siblings), or with sib_only the
    test runs with >= 1 sibling. Reference models are restricted to the same run ids. A row missing in any of the
    splits, or NaN on a selected run, is left out of the table (named below it), never averaged over fewer runs."""
    cols = {}                                                                        # row label -> [scores per split]
    dropped, n_sel, n_all = [], [], []
    getters = {r: reference(r) for r in refs}
    for s in splits:
        ids = res[f"{s}|ids"]
        sel = res[f"{s}|n_sib"] > 0 if sib_only else np.ones(len(ids), bool)
        n_sel.append(int(sel.sum()))
        n_all.append(len(ids))
        sd = d.train_sd(s)
        for m in methods:
            if not sib_only and m in SIB_METHODS:
                continue
            k = f"{s}|{m}|mse"
            if k not in res or np.isnan(res[k][sel]).any():
                dropped.append(f"{m} ({s})")
                continue
            cols.setdefault(LABELS[m], []).append(norm_scores(res[k][sel], res[f"{s}|{m}|mae"][sel], sd))
        for r, get in getters.items():
            g = get(s)
            if g is None or not set(ids[sel]) <= set(g[0]):
                dropped.append(f"{REFS[r]} ({s})")
                continue
            pos = {v: k for k, v in enumerate(g[0])}
            ix = np.array([pos[v] for v in ids[sel]])
            cols.setdefault(REFS[r], []).append(norm_scores(g[1][ix], g[2][ix], sd))
    per = ", ".join(f"{s} {a}/{b}" for s, a, b in zip(splits, n_sel, n_all))
    lines = [f"### {'+'.join(splits)}: {sum(n_sel)} of {sum(n_all)} test runs "
             f"({'runs with >= 1 sibling' if sib_only else 'all test runs; wiener_sib rows need siblings, see next table'})",
             "", f"Same {sum(n_sel)} runs in every row ({per}). Normalised MSE / MAE (error / train std of the split, "
             f"mean over heave/roll/pitch and leads <= h){'; mean over the ' + str(len(splits)) + ' folds' if len(splits) > 1 else ''}.",
             "", "| Method | MSE 0.2s | " + " | ".join(f"{h}s MSE | {h}s MAE" for h in HORIZONS) + " |",
             "|---|---|" + "---|" * (2 * len(HORIZONS))]
    for m, v in cols.items():
        if len(v) != len(splits):
            continue                                                                 # listed in `dropped`
        avg = {h: np.mean([x[h] for x in v], axis=0) for h in v[0]}
        lines.append(f"| {m} | {avg[0.2][0]:.4f} | " + " | ".join(f"{avg[h][0]:.4f} | {avg[h][1]:.4f}" for h in HORIZONS) + " |")
    if dropped:
        lines += ["", "Rows left out (missing or NaN on these runs): " + ", ".join(dropped)]
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    return text


def tables(res, d, splits, path=None):
    """cv folds present (mean over folds), then each OOD split; all test runs, then the runs with siblings."""
    have = [s for s in splits if f"{s}|ids" in res]
    groups = [[s for s in have if s.startswith("cv")]] + [[s] for s in have if s.startswith("ood")]
    text = ["# Wiener diagnostics: normalised MSE / MAE (generated by experiments/diagnostics/wiener.py)", "",
            "wiener_own_lines is a line-spectrum artefact (it knows the run's own discrete wave lines and future) and "
            "is excluded from conclusions; wiener_sib* use the condition's other runs (not legitimate); wiener_past "
            "and the trained models use only the past of the test run.", ""]
    for g in [g for g in groups if g]:
        text += [print_table(res, d, g, sib_only=False), print_table(res, d, g, sib_only=True)]
    if path:
        open(path, "w", encoding="utf-8").write("\n".join(text))
        print(f"-> {path}", flush=True)


# ---------------------------------------------------------------- design studies
def study(n_runs=24):
    d = Data()
    rows = LCV.split_rows(d.meta, "cv0")["val"]                                    # not the cv0 test runs validated on
    rows = rows[np.linspace(0, len(rows) - 1, n_runs).astype(int)]
    sd = d.train_sd("cv0")
    sib = np.array([len(d.sib.get(i, [])) > 0 for i in rows])
    out = [f"Wiener design study: {n_runs} cv0 validation-fold runs (ids {', '.join(map(str, d.meta.id.to_numpy()[rows]))}), "
           f"{sib.sum()} with siblings. Normalised MSE (cv0 train std) at lead 0.2 s and mean over leads <= "
           f"{'/'.join(map(str, HORIZONS))} s. own = wiener_own_lines, cond = wiener_sib (lag window M as given).",
           f"{'config':58s} {'0.2s':>7s} " + " ".join(f"{h:>6d}s" for h in HORIZONS) + "   time"]

    def line(tag, method, mask=None, **kw):
        t0 = time.time()
        r = rows if mask is None else rows[mask]
        mse, mae, _ = score_rows(d, r, method, progress=False, **kw)
        sc = norm_scores(mse, mae, sd)
        s = f"{tag:58s} {sc[0.2][0]:7.4f} " + " ".join(f"{sc[h][0]:7.4f}" for h in HORIZONS) + f" {time.time() - t0:5.1f}s"
        print(s, flush=True)
        out.append(s)

    def head(text):
        out.append("-- " + text)
        print(out[-1], flush=True)

    head("wiener_own_lines: univariate vs joint, past length L (ridge 1e-6)")
    for uni in (True, False):
        for L in (10, 25, 50, 100, 200, 300):
            line(f"own {'uni' if uni else 'multi'} L={L}", "wiener_own_lines", uni=uni, L=L, ridge=1e-6)
    head("wiener_own_lines: ridge, Parzen lag window M (multi, L=300)")
    for rg in (1e-10, 1e-8, 1e-4, 1e-2):
        line(f"own multi L=300 ridge={rg:g}", "wiener_own_lines", L=300, ridge=rg)
    for M in (100, 200, 400):
        line(f"own multi L=300 ridge=1e-6 Parzen M={M}", "wiener_own_lines", L=300, lagwin=M)
    head(f"wiener_sib on the {sib.sum()} runs with siblings: uni vs joint, L, ridge, Parzen M, sibling vs run mean")
    for uni in (True, False):
        for L in (50, 100, 300):
            line(f"cond {'uni' if uni else 'multi'} L={L} ridge=1e-6", "wiener_sib", sib, uni=uni, L=L,
                 ridge=1e-6, lagwin=None)
    for rg in (1e-4, 1e-2, 1e-1):
        line(f"cond multi L=300 ridge={rg:g}", "wiener_sib", sib, L=300, ridge=rg, lagwin=None)
    for uni in (True, False):
        for L, M in ((100, 50), (100, 100), (300, 50), (300, 100), (300, 200)):
            for rg in (1e-6, 1e-2):
                line(f"cond {'uni' if uni else 'multi'} L={L} ridge={rg:g} Parzen M={M}", "wiener_sib", sib, uni=uni,
                     L=L, ridge=rg, lagwin=M)
    line("cond M=100, test-run mean (uses its future)", "wiener_sib", sib, lagwin=100, mean="run")
    line("cond M=100", "wiener_sib", sib, lagwin=100)
    line("own (defaults), same runs", "wiener_own_lines", sib)
    head("wiener_past: estimator (multi, L=50, ridge 1e-2, no lag window)")
    for est in ESTIMATORS:
        line(f"past multi L=50 {est}", "wiener_past", uni=False, est=est, L=50, ridge=1e-2, lagwin=None)
    head("wiener_past: L, uni vs joint (biased, ridge 1e-2, no lag window)")
    for uni in (True, False):
        for L in (5, 10, 25, 50, 100):
            line(f"past {'uni' if uni else 'multi'} L={L} ridge=1e-2", "wiener_past", uni=uni, L=L, ridge=1e-2,
                 lagwin=None)
    head("wiener_past: ridge and Parzen lag window M (biased, L=10)")
    for uni in (True, False):
        for rg in (1e-6, 1e-4, 1e-3, 1e-1):
            line(f"past {'uni' if uni else 'multi'} L=10 ridge={rg:g}", "wiener_past", uni=uni, L=10, ridge=rg,
                 lagwin=None)
        for M in (50, 100, 200):
            line(f"past {'uni' if uni else 'multi'} L=10 ridge=1e-2 Parzen M={M}", "wiener_past", uni=uni, L=10,
                 ridge=1e-2, lagwin=M)
    for L in (10, 25):
        for rg in (1e-6, 1e-4):
            for M in (100, 200):
                line(f"past multi L={L} ridge={rg:g} Parzen M={M}", "wiener_past", uni=False, L=L, ridge=rg, lagwin=M)
    head("defaults")
    for m in METHODS:
        line(f"{m} (defaults)", m, sib if m in SIB_METHODS else None)
    path = os.path.join(OUT, "study_wiener.txt")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"-> {path}")


def study_sib(n_runs=60):
    """wiener_sib lag window on the cv0 VALIDATION-fold runs with siblings: (1) M = 100 vs M >= L + H (and none) x
    ridge, with the MSE at exactly 20 s; (2) one vs two siblings (does more pooling help at large M?); (3) heave
    autocorrelation of a run vs that of its siblings per lag (do long-lag correlations transfer across seeds?)."""
    d = Data()
    rows = LCV.split_rows(d.meta, "cv0")["val"]
    rows = rows[np.linspace(0, len(rows) - 1, n_runs).astype(int)]
    rows = np.array([i for i in rows if d.sib.get(i)])
    sd = d.train_sd("cv0")
    out = [f"wiener_sib lag-window study: {len(rows)} cv0 validation-fold runs with siblings (of {n_runs} evenly "
           f"spaced; ids {', '.join(map(str, d.meta.id.to_numpy()[rows]))}). Normalised MSE (cv0 train std) at lead "
           f"0.2 s, mean over leads <= {'/'.join(map(str, HORIZONS))} s, and at exactly 20 s. Parzen window weight "
           f"at lag 99 (lead 19.8 s): " + ", ".join(f"M={M} {parzen_w(100, M)[-1]:.3f}" for M in (100, 200, 400, 500, 700)),
           f"{'config':40s} {'0.2s':>7s} " + " ".join(f"{h:>6d}s" for h in HORIZONS) + "   @20s"]

    def line(tag, r, **kw):
        mse, mae, _ = score_rows(d, r, "wiener_sib", progress=False, **kw)
        sc = norm_scores(mse, mae, sd)
        s = (f"{tag:40s} {sc[0.2][0]:7.4f} " + " ".join(f"{sc[h][0]:7.4f}" for h in HORIZONS)
             + f" {np.mean(mse[:, -1] / sd ** 2):7.4f}")
        print(s, flush=True)
        out.append(s)

    out.append("-- (1) lag window M x ridge, L = 300")
    for M in (100, 200, 400, 500, 700, None):
        for rg in (1e-6, 1e-2):
            line(f"L=300 M={M} ridge={rg:g}", rows, L=300, lagwin=M, ridge=rg)
    two = np.array([i for i in rows if len(d.sib[i]) == 2])
    out.append(f"-- (2) one vs two siblings, L = 300, ridge 1e-6, on the {len(two)} runs with two siblings")
    for M in (100, 500, None):
        mse1 = [score_one(d, i, d.sib[i][:1], M) for i in two]
        mse2 = [score_one(d, i, d.sib[i], M) for i in two]
        for tag, m in (("1 sibling", mse1), ("2 siblings", mse2)):
            m = np.array(m) / sd ** 2
            s = (f"{f'M={M} {tag}':40s} {m[:, 0].mean():7.4f} " + " ".join(f"{m[:, :int(h * FS)].mean():7.4f}" for h in HORIZONS)
                 + f" {m[:, -1].mean():7.4f}")
            print(s, flush=True)
            out.append(s)
    out.append(f"-- (3) heave autocorrelation rho(h) of the run vs the mean over its siblings, {len(rows)} runs: "
               "corr across runs, rms difference, sd of the run's own rho(h)")
    for h in (5, 10, 25, 50, 100, 150, 200, 300, 399):
        a, b = [], []
        for i in rows:
            Go = d.stats(i)[0][:, 0, 0]
            Gs = np.mean([d.stats(j)[0][:, 0, 0] for j in d.sib[i]], 0)
            a.append(Go[h] / Go[0])
            b.append(Gs[h] / Gs[0])
        a, b = np.array(a), np.array(b)
        s = (f"lag {h:3d} ({h / FS:5.1f} s): corr {np.corrcoef(a, b)[0, 1]:+.3f}  rms diff {np.sqrt(np.mean((a - b) ** 2)):.3f}"
             f"  sd own {a.std():.3f}")
        print(s, flush=True)
        out.append(s)
    path = os.path.join(OUT, "study_wiener_sib.txt")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"-> {path}")


def score_one(d, i, sib, M):
    """study_sib helper: per-lead MSE (H, 3) of wiener_sib (L = 300, ridge 1e-6) for run i with the given siblings."""
    y = np.asarray(d.Y[i], np.float64)
    pred = run_method(d, i, "wiener_sib", sib=sib, lagwin=M)
    return errors(pred, y, origins(len(y)))[0]


# ---------------------------------------------------------------- main
def clean(res, s, ids, rows, d):
    """Drop every key of split s if its stored ids differ from ids (e.g. other --max-runs); set ids and n_sib."""
    if f"{s}|ids" in res and not np.array_equal(res[f"{s}|ids"], ids):
        old = [k for k in res if k.startswith(f"{s}|")]
        print(f"    {s}: stored ids differ ({len(res[f'{s}|ids'])} vs {len(ids)} runs): dropping {len(old)} keys", flush=True)
        for k in old:
            del res[k]
    res[f"{s}|ids"] = ids
    res[f"{s}|n_sib"] = np.array([len(d.sib.get(i, [])) for i in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "study", "study-sib", "table"])
    ap.add_argument("--splits", nargs="+", default=LCV.SPLITS, choices=LCV.SPLITS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--max-runs", type=int, default=None, help="first N test runs of each split (testing)")
    ap.add_argument("--npz", default=None, help="metrics file (default by --max-runs)")
    ap.add_argument("--force", action="store_true", help="recompute keys already in the file")
    ap.add_argument("--study-runs", type=int, default=None, help="study: 24, study-sib: 60")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "study":
        return study(a.study_runs or 24)
    if a.cmd == "study-sib":
        return study_sib(a.study_runs or 60)
    tag = "" if a.max_runs is None else f"_max{a.max_runs}"
    path = a.npz or os.path.join(OUT, f"wiener_metrics{tag}.npz")
    md = os.path.splitext(path)[0].replace("wiener_metrics", "wiener_tables") + ".md"
    res = dict(np.load(path)) if os.path.exists(path) else {}
    known = {f"{s}|{k}" for s in LCV.SPLITS for k in ("ids", "n_sib")} | {f"cfg|{m}" for m in METHODS} | \
            {f"{s}|{m}|{k}" for s in LCV.SPLITS for m in METHODS for k in ("mse", "mae", "mse_o")}
    stale = sorted(k for k in res if k not in known)                               # e.g. wiener_own / wiener_cond keys
    if stale:
        print(f"  dropping {len(stale)} keys of earlier method names/layout: {', '.join(stale[:6])}"
              f"{' ...' if len(stale) > 6 else ''}", flush=True)
        for k in stale:
            del res[k]
    d = Data()
    print(f"  conditions by number of 'ok' runs: {d.group_sizes}", flush=True)
    if a.cmd == "table":
        return tables(res, d, a.splits, md)
    for m in METHODS:                                                               # settings changed -> recompute
        cfg = repr(sorted(CFG[m].items()))
        if f"cfg|{m}" in res and str(res[f"cfg|{m}"]) != cfg:
            old = [k for k in res if k.endswith(f"|{m}|mse") or k.endswith(f"|{m}|mae") or k.endswith(f"|{m}|mse_o")]
            print(f"  {m}: settings changed, dropping {len(old)} keys", flush=True)
            for k in old:
                del res[k]
        res[f"cfg|{m}"] = np.array(cfg)
    ref = np.load(LCV.METRICS)
    t_all = time.time()
    for s in a.splits:
        rows = LCV.split_rows(d.meta, s)["test"][:a.max_runs]
        ids = d.meta.id.to_numpy()[rows]
        ref_ids = ref[f"{s}|ids"]
        assert np.array_equal(ids, ref_ids[:len(ids)]) and (a.max_runs is not None or len(ids) == len(ref_ids)), \
            f"{s}: test ids differ from {LCV.METRICS}"
        clean(res, s, ids, rows, d)
        print(f"  {s}: {len(rows)} test runs ({(res[f'{s}|n_sib'] == 0).sum()} without siblings)", flush=True)
        for m in a.methods:
            if f"{s}|{m}|mse" in res and not a.force:
                print(f"    {m} already in {os.path.basename(path)}", flush=True)
                continue
            t0 = time.time()
            res[f"{s}|{m}|mse"], res[f"{s}|{m}|mae"], res[f"{s}|{m}|mse_o"] = score_rows(d, rows, m)
            np.savez_compressed(path, **res)
            ok = ~np.isnan(res[f"{s}|{m}|mse"][:, 0, 0])
            mm, _ = LCV.metrics(res[f"{s}|{m}|mse"][ok], res[f"{s}|{m}|mae"][ok])
            print(f"    {m} ({time.time() - t0:.0f} s, {ok.sum()} runs): RMSE heave {mm['RMSE'][0]:.4f} m, "
                  f"roll {mm['RMSE'][1]:.3f} deg, pitch {mm['RMSE'][2]:.3f} deg", flush=True)
        d._G.clear(); d._mu.clear()
    np.savez_compressed(path, **res)
    print(f"  total {time.time() - t_all:.0f} s -> {path}", flush=True)
    tables(res, d, a.splits, md)


if __name__ == "__main__":
    main()
