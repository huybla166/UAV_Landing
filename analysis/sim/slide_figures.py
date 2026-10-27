"""Simplified, Vietnamese-labelled figures for the progress slides (Giai đoạn 0-1).

All data come from files already produced by the verification scripts:
  F1 data/sim/check_fastwave/{ref_msslib,fast_position}.csv      (wave phase drift)
  F2 analysis/sim/A_regular_wave_b180_{lo,hi}.csv                 (regular waves, pitch, head sea)
  F3 analysis/sim/B/B_B1_b090_series.csv                          (linear limit, roll)
  F4 analysis/sim/B/B_B2_Hs015_k0_series.csv                      (zero-speed vs forward-speed linear model)
  F5 data/processed/noWing_headQ_20Hz.csv, data/sim/D_ntua_like/otter55_noWing_headQ.csv  (NTUA vs Otter)
Palette: categorical slots 1-2 of the dataviz reference palette (#2a78d6, #eb6834), validated
(CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1); ink #13293d, muted #5a6672.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "analysis", "sim", "slides")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#13293d", "#5a6672", "#ddd8cc", "#f5f2ea"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG, "figure.facecolor": BG,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------- F1 wave phase drift
def centroid(z, fs=20.0):
    z = z - z.mean()
    Z = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    w = 2 * np.pi * np.fft.rfftfreq(len(z), 1 / fs)
    return np.sum(w[1:] * Z[1:]) / np.sum(Z[1:])


fig, ax = plt.subplots(figsize=(8, 4.5))
for f, col, lab in (("ref_msslib", ORANGE, "MSS gốc (pha = Ω_e·t)"), ("fast_position", BLUE, "Đã sửa (pha theo vị trí)")):
    x = np.loadtxt(os.path.join(ROOT, "data", "sim", "check_fastwave", f + ".csv"), delimiter=",", skiprows=1)
    t, z = x[:, 0], x[:, 1]
    win, step = 800, 200
    tc, wc = [], []
    for s in range(0, len(z) - win, step):
        tc.append(t[s + win // 2]); wc.append(centroid(z[s:s + win]))
    ax.plot(tc, wc, color=col, label=lab)
    ax.text(tc[-1] + 8, wc[-1] + (2.2 if col == BLUE else 0), lab.split(" (")[0], color=INK, fontsize=12, va="center")
ax.axhline(3.31, color=MUTED, ls="--", lw=1.2)
ax.text(240, 3.31 - 1.9, "đỉnh phổ sóng 3.31 rad/s", color=MUTED, fontsize=12)
ax.axhline(9.5, color=MUTED, ls=":", lw=1.2)
ax.text(330, 9.5 + 0.8, "tần số lớn nhất có trong phổ 9.5 rad/s", color=MUTED, fontsize=12)
ax.set_xlabel("thời gian mô phỏng [s]")
ax.set_ylabel("tần số trung bình của sóng\ntại tàu [rad/s]")
ax.set_xlim(0, 690); ax.set_ylim(0, 36)
ax.legend(loc="upper left", fontsize=12)
save(fig, "F1_phase_drift.png")

# ---------------------------------------------------------------- F2 regular waves, pitch, head sea
A = pd.concat([pd.read_csv(os.path.join(ROOT, "analysis", "sim", f"A_regular_wave_b180_{p}.csv")) for p in ("lo", "hi")])
s = A[A.dof == "pitch"].sort_values("w")
k = np.degrees(1) / s["a"].iloc[0]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axvspan(5, 9.2, color="#e9e4d8", zorder=0)
ax.text(7.45, 215, "vùng cộng hưởng\ncủa Otter", color=MUTED, fontsize=12)
ax.plot(s.w, s.const_amp * k, color=BLUE, label="ODE hệ số hằng (MSS, lấy tại 3.31 rad/s)")
ax.plot(s.w, s.fd_amp * k, color=ORANGE, ls="--", label="lý thuyết phụ thuộc tần số (Capytaine)")
ax.plot(s.w, s.sim_amp * k, "o", color=INK, ms=7, label="mô phỏng miền thời gian")
ax.set_xlabel("tần số sóng ω [rad/s]")
ax.set_ylabel("biên độ pitch / biên độ sóng\n[độ/m]")
ax.set_xlim(1.8, 9.2); ax.set_ylim(0, 310)
ax.legend(loc="upper left", fontsize=11)
save(fig, "F2_regular_pitch.png")

# ---------------------------------------------------------------- F3 linear limit, roll
S = pd.read_csv(os.path.join(ROOT, "analysis", "sim", "B", "B_B1_b090_series.csv"))
m = (S.t >= 300) & (S.t <= 315)
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(S.t[m], np.degrees(S.roll_nl[m]), color=INK, lw=3.0, label="mô hình đầy đủ (phi tuyến)")
ax.plot(S.t[m], np.degrees(S.roll_LU[m]), color=ORANGE, lw=1.8, ls="--", label="ODE tuyến tính = tổng các hình sin")
ax.set_xlabel("thời gian [s]"); ax.set_ylabel("roll [độ]")
ax.legend(loc="upper left", fontsize=12, ncol=2)
lim = np.abs(np.degrees(S.roll_nl[m])).max() * 1.6
ax.set_ylim(-lim, lim)
save(fig, "F3_linear_limit_roll.png")

# ---------------------------------------------------------------- F4 zero-speed vs forward-speed linear model, pitch
S = pd.read_csv(os.path.join(ROOT, "analysis", "sim", "B", "B_B2_Hs015_k0_series.csv"))
m = (S.t >= 300) & (S.t <= 312)
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(S.t[m], np.degrees(S.pitch_nl[m]), color=INK, lw=3.0, label="mô hình đầy đủ")
ax.plot(S.t[m], np.degrees(S.pitch_L0[m]), color=ORANGE, lw=1.8, label="ODE tại vận tốc 0  (R² = −1.17)")
ax.plot(S.t[m], np.degrees(S.pitch_LU[m]), color=BLUE, lw=1.8, ls="--", label="tuyến tính hóa quanh U = 1 m/s  (R² = 0.95)")
ax.set_xlabel("thời gian [s]"); ax.set_ylabel("pitch [độ]")
lim = np.abs(np.degrees(S.pitch_L0[m])).max() * 1.1
ax.set_ylim(-lim, 2.2 * lim)
ax.legend(loc="upper left", fontsize=11, ncol=1)
save(fig, "F4_pitch_L0_vs_LU.png")

# ---------------------------------------------------------------- F5 NTUA vs Otter, roll spectrum
FS = 20.0
nt = np.loadtxt(os.path.join(ROOT, "data", "processed", "noWing_headQ_20Hz.csv"), delimiter=",", skiprows=1)
nt = nt[int(5 * FS):-int(5 * FS), 2]
ot = np.loadtxt(os.path.join(ROOT, "data", "sim", "D_ntua_like", "otter55_noWing_headQ.csv"), delimiter=",", skiprows=1)
ot = np.degrees(ot[ot[:, 0] >= 30, 11])
fig, ax = plt.subplots(figsize=(8, 4.5))
for y, col, lab in ((nt, INK, "NTUA 10–12 m (đo thật)"), (ot, BLUE, "Otter 2 m (mô phỏng)")):
    f, p = signal.welch(signal.detrend(y), fs=FS, nperseg=1024)
    msk = (f >= 0.2) & (f <= 2.5)
    w = 2 * np.pi * f[msk]; P = p[msk] / (2 * np.pi)
    ax.plot(w, P, color=col, lw=2.4 if col == INK else 2.0, label=lab)
    kpk = np.argmax(P)
    tx = (6.6, P[kpk] * 0.92) if col == INK else (5.6, P[kpk] * 0.88)
    ax.annotate(f"đỉnh {w[kpk]:.1f} rad/s (chu kỳ {2*np.pi/w[kpk]:.1f} s)", (w[kpk], P[kpk]),
                xytext=tx, color=INK, fontsize=12, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.axvline(2.95, color=ORANGE, ls=":", lw=1.6)
ax.text(2.85, 0.95 * 0.98, "phao không đo được\nsóng dưới 2.95 rad/s", color=INK, fontsize=11, va="top", ha="right")
ax.set_ylim(0, 1.0)
ax.set_xlabel("tần số ω [rad/s]"); ax.set_ylabel("phổ roll [độ² s/rad]")
ax.set_xlim(0.2, 12)
ax.legend(loc="center right", fontsize=12)
save(fig, "F5_ntua_vs_otter_roll.png")
