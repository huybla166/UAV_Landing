"""Recover the data behind Fig. 6 of Wang et al. 2026 (J. Franklin Inst. 363:108381) from the vector graphics of the PDF
and identify how the 'moving average' and 'variation' curves were computed.

    D:/tools/venv-gpu/Scripts/python.exe experiments/landing/digitize_wang_fig6.py

Input:  references/Wang2026_JFI_108381.pdf, page 15 (Fig. 6). The curves are vector paths (MATLAB export): the green
        dashed 'Moving Average' is a polyline of exactly 2500 segments (one vertex per sample), the black 'Roll/Pitch
        Angle' and blue 'Variation' curves are polylines + cubic Bezier pieces (sampled here at 12 points each).
        Axis calibration from the light-grey grid lines (x: 0..2500 at 500 steps; y: panel (a) -2..2, (b) -1.5..1.5,
        (c) 0..0.8, (d) 0..1).
Output: results/sim/landing/wang_fig6_digitized.npz (roll/pitch raw, moving average, variation, sample 1..2500)
        results/sim/landing/wang_fig6_fit.txt (the identification below, with RMS residuals in degrees).

Identification: candidate formulas (MATLAB movmean / movstd with 'shrink' end points, window L samples, centred as
MATLAB does: even L -> [i-L/2, i+L/2-1]; trailing/leading variants; moving range; mean |x - movmean|) are fitted to
the digitised curves; the window-selection rule is identified by checking which mask reproduces the reported window
(samples 388..485, 98 samples, Section 5.1).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "references", "Wang2026_JFI_108381.pdf")
OUT = os.path.join(ROOT, "results", "sim", "landing")
N = 2500


def _points(drawing):
    P = []
    for it in drawing["items"]:
        if it[0] == "l":
            P += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            p = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0, 1, 12):
                P.append(tuple((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2]
                               + t ** 3 * p[3]))
    P = np.array(P)
    keep = np.r_[True, np.any(np.diff(P, axis=0) != 0, axis=1)]
    return P[keep]


def extract():
    import pymupdf
    page = pymupdf.open(PDF)[14]
    curves, hlines = {}, {}
    for d in page.get_drawings():
        r = d["rect"]
        if r.y1 > 400:                                    # Fig. 7 is below
            continue
        col = tuple(round(c, 2) for c in (d["color"] or ()))
        side = "L" if r.x0 < 300 else "R"
        pos = "top" if r.y0 < 200 else "bot"
        if len(d["items"]) > 20 and col in [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]:
            curves[(col, side, pos)] = _points(d)
        if len(d["items"]) == 1 and r.width > 150 and col in [(1, 0.3, 1), (1, 0.3, 0.3)]:
            hlines[(side, pos)] = r.y0                    # threshold lines
    # grid calibration (page coordinates read from the grey grid lines, see module docstring)
    x0 = {"L": 77.48, "R": 314.24}
    xw = 257.22 - 77.48
    cal = {("L", "top"): (71.91, 194.02, 2.0, -2.0), ("R", "top"): (71.91, 194.02, 1.5, -1.5),
           ("L", "bot"): (241.79, 363.91, 0.8, 0.0), ("R", "bot"): (241.79, 363.91, 1.0, 0.0)}

    def series(key):
        P = curves[key]
        side, pos = key[1], key[2]
        y0, y1, v0, v1 = cal[(side, pos)]
        return (P[:, 0] - x0[side]) / xw * N, v0 + (P[:, 1] - y0) / (y1 - y0) * (v1 - v0)

    def level(side, pos):
        y0, y1, v0, v1 = cal[(side, pos)]
        return v0 + (hlines[(side, pos)] - y0) / (y1 - y0) * (v1 - v0)

    n = np.arange(1, N + 1)
    out = dict(sample=n)
    for side, nm in [("L", "roll"), ("R", "pitch")]:
        out[nm] = np.interp(n, *series(((0, 0, 0), side, "top")))
        out[nm + "_ma"] = np.interp(n, *series(((0, 1, 0), side, "top")))
        out[nm + "_var"] = np.interp(n, *series(((0, 0, 1), side, "bot")))
        s, _ = series(((1, 0, 0), side, "top"))
        out[nm + "_window"] = np.array([s.min(), s.max()])
        out[nm + "_thr_ma"] = level(side, "top")
        out[nm + "_thr_var"] = level(side, "bot")
    return out


def movstat(x, L, kind, ddof=1):
    """MATLAB movmean(x, L) / movstd(x, L) (default 'shrink' end points), 1-D, loop version used as reference."""
    n, b = len(x), L // 2
    a = L - 1 - b
    out = np.empty(n)
    for i in range(n):
        w = x[max(0, i - b):min(n, i + a + 1)]
        out[i] = w.mean() if kind == "mean" else w.std(ddof=ddof if len(w) > 1 else 0)
    return out


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return sorted([(int(a) + 1, int(b), int(b - a)) for a, b in zip(s, e)], key=lambda t: -t[2])  # 1-based, inclusive


def main():
    os.makedirs(OUT, exist_ok=True)
    z = extract()
    np.savez(os.path.join(OUT, "wang_fig6_digitized.npz"), **z)
    L_ = ["Wang et al. 2026, Fig. 6, digitised from the PDF vector paths (experiments/landing/digitize_wang_fig6.py)", ""]
    rms = lambda a, b: float(np.sqrt(np.mean((a - b)[30:-30] ** 2)))         # noqa: E731  (interior, deg)
    for nm in ("roll", "pitch"):
        x = z[nm]
        L_ += [f"[{nm}] std {x.std():.3f} deg, mean {x.mean():+.3f}, max|x| {np.abs(x).max():.2f}; "
               f"zero-up-crossing period {N / 5 / ((x[:-1] - x.mean() < 0) & (x[1:] - x.mean() >= 0)).sum():.1f} s "
               f"(assuming the stated 5 Hz)",
               f"  red 'Selected Landing Phase' spans samples {z[nm + '_window'][0]:.1f} .. {z[nm + '_window'][1]:.1f}",
               f"  drawn thresholds: moving average {z[nm + '_thr_ma']:.3f} deg, variation {z[nm + '_thr_var']:.3f} deg",
               f"  RMS(moving average - raw) = {rms(z[nm + '_ma'], x):.4f} deg (identity, for scale)"]
        fits = []
        for L in range(2, 41):
            fits.append((rms(movstat(x, L, "mean"), z[nm + "_ma"]), f"movmean(x,{L}) vs moving average"))
            for dd in (0, 1):
                fits.append((rms(movstat(x, L, "std", dd), z[nm + "_var"]), f"movstd(x,{L}) ddof={dd} vs variation"))
            k = np.ones(L) / L
            full = np.convolve(np.abs(x), k, "full")
            fits.append((rms(full[:N], z[nm + "_var"]), f"trailing mean|x| over {L} vs variation"))
            fits.append((rms(movstat(np.abs(x), L, "mean"), z[nm + "_var"]), f"centred mean|x| over {L} vs variation"))
        fits = sorted(fits)
        L_ += ["  best moving-average fits: " + "; ".join(f"{t} rms {r:.4f}" for r, t in
                                                          [f for f in fits if 'moving average' in f[1]][:3]),
               "  best variation fits:      " + "; ".join(f"{t} rms {r:.4f}" for r, t in
                                                          [f for f in fits if 'variation' in f[1]][:3]),
               "  literal Eq. (9) mean|x| over the whole 98-sample window 388..485: "
               f"{np.abs(x[387:485]).mean():.3f} deg", ""]
    r, p = z["roll"], z["pitch"]
    mr, mp = movstat(r, 20, "mean"), movstat(p, 20, "mean")
    vr, vp = movstat(r, 20, "std"), movstat(p, 20, "std")
    L_ += ["Window-selection rule (movmean/movstd with L = 20, thresholds 0.5 / 0.3 / 0.22 deg): longest runs of the mask"]
    for name, m in [("signed MA <= thr and movstd <= 0.22 (both)", (mr <= .5) & (mp <= .3) & (vr <= .22) & (vp <= .22)),
                    ("signed MA <= thr, pitch 0.35 as drawn", (mr <= .5) & (mp <= .35) & (vr <= .22) & (vp <= .22)),
                    ("|MA| <= thr and movstd <= 0.22 (two-sided)", (abs(mr) <= .5) & (abs(mp) <= .3) & (vr <= .22) & (vp <= .22)),
                    ("movstd <= 0.22 only", (vr <= .22) & (vp <= .22)),
                    ("|MA| <= thr only", (abs(mr) <= .5) & (abs(mp) <= .3))]:
        L_.append(f"  {name:45s} stable share {m.mean():.3f}; longest runs (start, end, n) {runs(m)[:3]}")
    txt = "\n".join(L_) + "\n"
    with open(os.path.join(OUT, "wang_fig6_fit.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
