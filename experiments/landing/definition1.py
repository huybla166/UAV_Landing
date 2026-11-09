"""Landing-window detector of Wang et al. 2026 (J. Franklin Inst. 363:108381), Definition 1 + Section 5.1, vectorised.

    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/definition1.py          # self-check + truth statistics

What the paper writes (references/Wang2026_JFI_108381.pdf; line numbers of the pdftotext -layout text wang.txt):
  * Definition 1 (wang.txt l. 336-345, PDF p. 5, Eq. 9): given angular thresholds phi_bar, theta_bar > 0,
      C1 = {(z_sd, dz_s) | z_s > 0, dz_s < 0}                          (heave above the horizontal plane and descending)
      C2 = {(phi_s, theta_s) | sum_{t1}^{t1+dt} |phi_s| / dt <= phi_bar,  sum_{t1}^{t1+dt} |theta_s| / dt <= theta_bar}
    "Then the landing maneuver is activated at t1 and constrained to terminate within [t1, t1 + dt]."
    Remark 2 (l. 347-351): C1 = descent starts after the heave peak and before the horizontal plane; C2 = roll and
    pitch "relatively stable with a small range of variation ... during a future period".
  * l. 357-359 (p. 6): the predicted window dt sets the predefined convergence time T_c < dt of the controller.
  * Section 5.1 (l. 937-950, p. 13-14): "angular thresholds are set to 0.5 deg for roll and 0.3 deg for pitch" (the
    degree signs are lost in wang.txt; the Unicode extraction analysis/sim/scratch_from_C/wang2026.txt l. 896-899 has
    them); Fig. 6(a, b): "the moving averages remain within the stability thresholds"; Fig. 6(c, d): "both roll and
    pitch variations consistently staying below the variation threshold of phi_bar = theta_bar = 0.22 deg"; the window
    "had a duration of 98 sample points (from sample 388 to 485)" = 98/5 s = 19.6 s; T_c strictly less than 19.6 s.
  * Remark 6 (l. 883-886): T_c is chosen strictly less than the duration of the predicted safe landing window.
  * The text gives NO moving-average window length, no definition of 'variation' and no rule that turns the curves into
    a window. Fig. 6(c, d) labels the variation axis sum|phi_s|/dt (deg), the C2 quantity.

What the paper's figure actually computes (experiments/landing/digitize_wang_fig6.py ->
results/sim/landing/wang_fig6_fit.txt, from the PDF's vector paths, 2500 samples):
  * moving average = MATLAB movmean(x, 20) (centred, 20 samples = 4 s, shrinking end points): RMS residual 0.006 deg
    roll / 0.005 deg pitch (the moving average differs from the raw angle by 0.06-0.07 deg RMS, so the fit is sharp);
  * variation = MATLAB movstd(x, 20) (sample std, N-1): RMS 0.010 / 0.007 deg; NOT the mean |x| over the window that
    the axis label and Eq. (9) state (best mean|x| fits are worse, see the fit file);
  * the reported window 388..485 (98 samples) is reproduced exactly as the LONGEST run of samples where
    movmean <= threshold (signed, one-sided: roll in the window goes to -0.54 deg, below -0.5) and movstd <= 0.22 for
    both roll and pitch. The movstd condition alone gives the same run; with |movmean| (two-sided) the run breaks up
    (longest 52 samples elsewhere). The pitch threshold line is drawn at 0.35 deg, the text says 0.3 (no effect here).
  * The literal Eq. (9) C2 over that window also holds (mean|roll| 0.199 <= 0.5, mean|pitch| 0.289 <= 0.3 deg).
  * Fig. 6 shows no heave panel: C1 is not used in the reported window.
  * Scale of the paper's data (digitised): roll std 0.672 deg, pitch std 0.462 deg, zero-up-crossing periods 17.9 s
    and 13.5 s (at the stated 5 Hz). Ours (dataset_v2, cv0 train runs, pooled): roll 2.268 deg, pitch 2.810 deg,
    periods ~1.7 s (cv0 test median). The 4 s window spans ~0.2 of their period but ~2.3 of ours.

Detector implemented here (per-sample mask, then windows = maximal runs of stable samples):
  stable[t] = MA_r[t] <=(|.|) ma_roll and MA_p[t] <=(|.|) ma_pitch and SD_r[t] <= var_roll and SD_p[t] <= var_pitch,
  MA = movmean(x, L), SD = movstd(x, L) (centred, MATLAB end-point 'shrink'), L = 20 samples at 5 Hz.
  Optional C1 (heave phase) is a condition on the window START only (Definition 1 activates the landing at t1).
Threshold sets (deg; THRESHOLDS below):
  wang_literal      0.5 / 0.3 (moving average, signed as in Fig. 6), 0.22 / 0.22 (movstd), L = 20.
  otter_scaled      the same rule, each channel's thresholds x (our std / the paper's std): roll x 3.376,
                    pitch x 6.082 (OTTER_STD / WANG_STD). Amplitude scaling only: L stays 4 s (see open questions).
  otter_scaled_abs  otter_scaled with a two-sided moving-average test (|MA| <= thr), the physically meaningful reading
                    (a deck tilted -3 deg is as bad as +3 deg); not what Fig. 6 does. On our data the 4 s moving
                    average spans > 2 periods and stays near 0, so it changes little (truth stats file).
  otter_share       otter_scaled x SHARE_C = 1.764 (both channels), the factor at which the stable share of the cv0
                    TRAIN runs equals the share of the paper's own Fig. 6 record under its rule (0.229). Reason: with
                    amplitude scaling alone the stable share on our runs is only 0.057 (train), because the 4 s
                    window spans ~0.2 of the paper's period (movstd measures the local slope, well below the std) but
                    > 2 of ours (movstd ~ the local std). Scaling L by the period ratio would give L ~ 2 samples,
                    which is degenerate at 5 Hz, so the share is matched instead. Statistical, not a physical limit.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results", "sim", "landing")

FS = 5.0
MA_LEN = 20                                     # samples, identified from Fig. 6 (movmean(x, 20))
WANG_STD = np.array([0.672, 0.462])             # roll, pitch deg: Fig. 6 data, results/sim/landing/wang_fig6_fit.txt
OTTER_STD = np.array([2.2684, 2.8098])          # roll, pitch deg: pooled std of the cv0 train runs (checked in main())
SCALE = OTTER_STD / WANG_STD                    # (3.376, 6.082)
WANG_SHARE = 0.229                              # stable share of the Fig. 6 record under wang_literal (fit file)
SHARE_C = 1.764                                 # bisection on the cv0 train runs (checked in main())


def _set(ma, var, scale=(1.0, 1.0), two_sided=False, L=MA_LEN):
    return dict(ma_roll=ma[0] * scale[0], ma_pitch=ma[1] * scale[1], var_roll=var[0] * scale[0],
                var_pitch=var[1] * scale[1], two_sided=two_sided, L=L)


THRESHOLDS = {
    "wang_literal": _set((0.5, 0.3), (0.22, 0.22)),
    "otter_scaled": _set((0.5, 0.3), (0.22, 0.22), SCALE),
    "otter_scaled_abs": _set((0.5, 0.3), (0.22, 0.22), SCALE, two_sided=True),
    "otter_share": _set((0.5, 0.3), (0.22, 0.22), SCALE * SHARE_C),
}


# ---------------------------------------------------------------- moving statistics (MATLAB movmean / movstd, 'shrink')
def movstats(x, L=MA_LEN):
    """x (..., T) -> (movmean, movstd) (..., T), centred window of L samples as MATLAB: even L covers [i-L/2, i+L/2-1],
    odd L [i-(L-1)/2, i+(L-1)/2]; the window shrinks at both ends; std with N-1 (0 for a 1-sample window)."""
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    b = L // 2
    a = L - 1 - b
    x0 = x - x[..., :1]                                            # offset for numerical stability of the sums
    z = np.zeros(x.shape[:-1] + (1,))
    S1 = np.concatenate([z, np.cumsum(x0, -1)], -1)
    S2 = np.concatenate([z, np.cumsum(x0 * x0, -1)], -1)
    i = np.arange(T)
    lo, hi = np.maximum(0, i - b), np.minimum(T, i + a + 1)
    n = (hi - lo).astype(np.float64)
    s1, s2 = S1[..., hi] - S1[..., lo], S2[..., hi] - S2[..., lo]
    m0 = s1 / n
    var = np.where(n > 1, (s2 - n * m0 * m0) / np.maximum(n - 1, 1), 0.0)
    return m0 + x[..., :1], np.sqrt(np.clip(var, 0, None))


def stable_mask(roll, pitch, thr):
    """roll, pitch (..., T) in deg -> per-sample stable mask (..., T) bool (rule in the module docstring)."""
    mr, sr = movstats(roll, thr["L"])
    mp, sp = movstats(pitch, thr["L"])
    if thr["two_sided"]:
        mr, mp = np.abs(mr), np.abs(mp)
    return (mr <= thr["ma_roll"]) & (mp <= thr["ma_pitch"]) & (sr <= thr["var_roll"]) & (sp <= thr["var_pitch"])


def heave_phase_ok(heave_down, ref=None):
    """Definition 1, C1, per sample: deck above the reference plane and moving down. heave_down (..., T) = NED D (m,
    positive down) as sim_common.motion; ref (..., 1) = reference level (default: the series mean). z_up = -(D - ref);
    C1[t] = z_up[t] > 0 and z_up[t+1] - z_up[t-1] < 0 (central difference; one-sided at the ends)."""
    D = np.asarray(heave_down, np.float64)
    ref = D.mean(-1, keepdims=True) if ref is None else ref
    z = -(D - ref)
    dz = np.gradient(z, axis=-1)
    return (z > 0) & (dz < 0)


def windows(mask, fs=FS):
    """1-D mask -> list of (start, end, duration_s), start/end 0-based sample indices, end inclusive."""
    d = np.diff(np.r_[0, np.asarray(mask, int), 0])
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(int(a), int(b) - 1, (b - a) / fs) for a, b in zip(s, e)]


def run_lengths(mask):
    """mask (N, T) -> durations in samples of all maximal True runs, concatenated (1-D int array)."""
    m = np.asarray(mask, np.int8)
    z = np.zeros((m.shape[0], 1), np.int8)
    d = np.diff(np.concatenate([z, m, z], 1), axis=1)
    r_s, c_s = np.nonzero(d == 1)
    r_e, c_e = np.nonzero(d == -1)
    return c_e - c_s                                               # row-major order keeps starts/ends paired


def first_window(mask, n_req, k_max, start_ok=None):
    """Decision on a horizon mask (O, K): the first start k <= k_max such that mask[k:k+n_req] is all True (and
    start_ok[k], Definition 1 C1, if given). Returns go (O,) bool and k (O,) int (-1 when no go). The window must fit in
    the K samples (k + n_req <= K)."""
    K = mask.shape[-1]
    ok = np.lib.stride_tricks.sliding_window_view(mask, n_req, axis=-1).all(-1)    # (O, K-n_req+1)
    ok = ok[:, :min(k_max, K - n_req) + 1]
    if start_ok is not None:
        ok = ok & start_ok[:, :ok.shape[1]]
    go = ok.any(-1)
    return go, np.where(go, ok.argmax(-1), -1)


def eq9_c2(roll, pitch, t1, n, phi_bar, theta_bar):
    """Literal Eq. (9) C2 for the window [t1, t1 + n) of 1-D series: mean |phi| <= phi_bar and mean |theta| <= theta_bar."""
    return bool(np.abs(roll[t1:t1 + n]).mean() <= phi_bar and np.abs(pitch[t1:t1 + n]).mean() <= theta_bar)


# ---------------------------------------------------------------- checks and statistics on the true motion
BINS = [0.2, 1.0, 2.5, 5.0, 10.0, 20.0, np.inf]                   # window duration bins (s)


def truth_stats(Y, Hs, thr):
    """Y (R, T, 3) true motion of runs, Hs (R,). -> lines: stable share, windows per hour by duration bin, per Hs."""
    out = []
    for h in [None] + sorted(set(Hs.tolist())):
        sel = np.ones(len(Y), bool) if h is None else np.isclose(Hs, h)
        m = stable_mask(Y[sel, :, 1].astype(np.float64), Y[sel, :, 2].astype(np.float64), thr)
        dur = run_lengths(m) / FS
        hours = m.size / FS / 3600
        cnt = np.histogram(dur, BINS)[0]
        ge = [(dur >= t).sum() / hours for t in (2.5, 5.0)]
        runs_with = np.mean([(run_lengths(mi[None]) >= 2.5 * FS).any() for mi in m])
        out.append(f"| {'all' if h is None else f'Hs {h:.1f} m'} | {sel.sum()} | {m.mean():.3f} | "
                   + " | ".join(f"{c / hours:.1f}" for c in cnt)
                   + f" | {ge[0]:.1f} | {ge[1]:.1f} | {np.median(dur) if len(dur) else float('nan'):.1f} | "
                   f"{dur.max() if len(dur) else 0:.1f} | {runs_with:.2f} |")
    return out


def main():
    import run_sim_lstm_cv as LCV
    from sim_common import load, motion
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Definition 1 detector: checks and statistics on the TRUE motion", "",
             "Written by `experiments/landing/definition1.py`.", ""]
    # 1. reproduce Fig. 6
    z = np.load(os.path.join(OUT, "wang_fig6_digitized.npz"))
    m = stable_mask(z["roll"], z["pitch"], THRESHOLDS["wang_literal"])
    w = max(windows(m), key=lambda t: t[2])
    lines += [f"Fig. 6 reproduction (wang_literal on the digitised data): longest window samples {w[0] + 1}..{w[1] + 1} "
              f"(1-based), {w[1] - w[0] + 1} samples = {w[2]:.1f} s; paper: 388..485, 98 samples, 19.6 s.",
              f"Literal Eq. (9) C2 on that window (0.5 / 0.3 deg): "
              f"{eq9_c2(z['roll'], z['pitch'], w[0], w[1] - w[0] + 1, 0.5, 0.3)}.", ""]
    assert (w[0] + 1, w[1] + 1) == (388, 485), w
    # vectorised movstats == loop reference
    from digitize_wang_fig6 import movstat
    for L in (19, 20):
        a, s = movstats(z["roll"], L)
        assert np.allclose(a, movstat(z["roll"], L, "mean"), atol=1e-9) and np.allclose(s, movstat(z["roll"], L, "std"), atol=1e-7)
    # 2. our scale
    meta, X = load()
    tr = LCV.split_rows(meta, "cv0")["train"]
    Ytr = motion(X[tr])
    pooled = Ytr[:, :, 1:].astype(np.float64).std(axis=(0, 1))
    lines += [f"Our scale (cv0 train runs, n = {len(tr)}): pooled std roll {pooled[0]:.4f}, pitch {pooled[1]:.4f} deg "
              f"(hard-coded OTTER_STD {OTTER_STD.tolist()}); paper (Fig. 6) roll {WANG_STD[0]}, pitch {WANG_STD[1]} deg; "
              f"scale factors roll {SCALE[0]:.3f}, pitch {SCALE[1]:.3f}.", ""]
    assert np.allclose(pooled, OTTER_STD, atol=5e-4), pooled
    sh = stable_mask(Ytr[:, :, 1].astype(np.float64), Ytr[:, :, 2].astype(np.float64), THRESHOLDS["otter_share"]).mean()
    sh1 = stable_mask(Ytr[:, :, 1].astype(np.float64), Ytr[:, :, 2].astype(np.float64), THRESHOLDS["otter_scaled"]).mean()
    lines += [f"Stable share on the cv0 train runs: otter_scaled {sh1:.3f}; otter_share (SHARE_C = {SHARE_C}) {sh:.3f} "
              f"(target = the paper's Fig. 6 share {WANG_SHARE}).", ""]
    assert abs(sh - WANG_SHARE) < 2e-3, sh
    del Ytr
    lines += ["Threshold sets (deg): " + "; ".join(
        f"**{k}** MA roll {v['ma_roll']:.3f} pitch {v['ma_pitch']:.3f} ({'|MA|' if v['two_sided'] else 'signed MA'}), "
        f"movstd roll {v['var_roll']:.3f} pitch {v['var_pitch']:.3f}, L = {v['L']}" for k, v in THRESHOLDS.items()), ""]
    # 3. truth statistics on the cv0 test runs (whole 2826-sample runs)
    te = LCV.split_rows(meta, "cv0")["test"]
    Y = motion(X[te])
    Hs = meta.Hs.to_numpy(float)[te]
    hdr = ("| runs | n | stable share | " + " | ".join(f"[{a:g},{b:g}) s /h" for a, b in zip(BINS[:-1], BINS[1:]))
           + " | >=2.5 s /h | >=5 s /h | median dur s | max dur s | share of runs with a >=2.5 s window |")
    for k, thr in THRESHOLDS.items():
        lines += [f"## {k}", "", hdr, "|" + "---|" * (hdr.count("|") - 1)] + truth_stats(Y, Hs, thr) + [""]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "definition1_truth_stats.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
