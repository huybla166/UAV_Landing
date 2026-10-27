"""Analysis of the NTUA/UiT 10-12 m self-propelled model sea-trial dataset (DOI 10.18710/VXGPII).

Loads heave/roll/pitch time series, wave-buoy spectra, computes sea state and motion
statistics, checks the landing-window condition of Definition 1 in Wang et al. (2026),
exports tidy CSVs (20 Hz and 5 Hz) and figures.
"""
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = "\\\\?\\"  # Windows extended-length path prefix (original file paths exceed 260 chars)
RAW = LONG + os.path.join(ROOT, "data", "VXGPII")
OUT_FIG = os.path.join(ROOT, "analysis", "figures")
OUT_CSV = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

RUNS = {
    # key: (folder, response file, wave spectrum file, response-spectrum file, label)
    "noWing_headQ": ("without_dynamic_wing/Aset_headQwaves",
                     "without_Aset_2_Responses_of_NTUA_model_measured.dat",
                     "without_Aset_1_Frequency_wave_spectrum.dat",
                     "without_Aset_4_ Frequency_spectra_of_NTUA_model_responses_measured.dat",
                     "Không cánh – sóng ngược chéo"),
    "noWing_followQ": ("without_dynamic_wing/Bset_followQwaves",
                       "without_Bset_2_Responses_of_NTUA_model_measured.dat",
                       "without_Bset_1_Frequency_wave_spectrum.dat",
                       "without_Bset_4_Frequency_spectra_of_NTUA_model_responses_measured.dat",
                       "Không cánh – sóng xuôi chéo"),
    "wing_headQ": ("with_dynamic_wing/Aset_headQwaves",
                   "with_Aset_2_Responses_of_ _NTUA_model_measured.dat",
                   "with_Aset_1_Frequency_wave_spectrum.dat",
                   "with_Aset_4_Frequency_spectra_of_NTUA_model_responses_measured.dat",
                   "Có cánh – sóng ngược chéo"),
    "wing_followQ": ("with_dynamic_wing/Bset_followQwaves",
                     "with_Bset_2_Responses_of_NTUA_model_measured.dat",
                     "with_Bset_1_Frequency_wave_spectrum.dat",
                     "with_Bset_4_Frequency_spectra_of_NTUA_model_responses_measured.dat",
                     "Có cánh – sóng xuôi chéo"),
}


def raw_path(folder, name):
    return os.path.join(RAW, *folder.split("/"), name)


def load(folder, name):
    return np.loadtxt(raw_path(folder, name), skiprows=1)


def md5(folder, name):
    return hashlib.md5(open(raw_path(folder, name), "rb").read()).hexdigest()


def sea_state(hs):
    # WMO sea state code (significant wave height, m)
    for code, upper, name in [(0, 0.0, "Calm (glassy)"), (1, 0.1, "Calm (rippled)"), (2, 0.5, "Smooth"),
                              (3, 1.25, "Slight"), (4, 2.5, "Moderate"), (5, 4.0, "Rough"),
                              (6, 6.0, "Very rough"), (7, 9.0, "High")]:
        if hs <= upper:
            return code, name
    return 8, "Very high"


def spectral_stats(f, s):
    df = np.gradient(f)
    m0 = np.sum(s * df)
    m1 = np.sum(f * s * df)
    m2 = np.sum(f ** 2 * s * df)
    fp = f[np.argmax(s)]
    return {"m0": m0, "Hs_m": 4 * np.sqrt(m0), "Tp_s": 1 / fp if fp > 0 else np.nan,
            "Tm01_s": m0 / m1 if m1 > 0 else np.nan, "Tz_s": np.sqrt(m0 / m2) if m2 > 0 else np.nan}


def welch_peak(x, fs):
    f, p = signal.welch(signal.detrend(x), fs=fs, nperseg=min(len(x), 2048))
    k = np.argmax(p[1:]) + 1
    above = p[f > 2.5].sum() / p.sum()  # variance fraction lost if resampled to 5 Hz
    return f, p, f[k], above


def decorrelation_time(x, fs):
    x = signal.detrend(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    zero = np.argmax(ac < 0)
    return zero / fs


def landing_windows(t, z, phi, theta, phi_th, theta_th, var_th=None, min_len=0.0):
    """Definition 1 (Wang et al. 2026): start where z>0 and dz/dt<0; window lasts while
    |roll|<=phi_th and |pitch|<=theta_th (and max-min variation <= var_th if given)."""
    dt = t[1] - t[0]
    dz = np.gradient(z, dt)
    ok = (np.abs(phi) <= phi_th) & (np.abs(theta) <= theta_th)
    windows = []
    i, n = 0, len(t)
    while i < n:
        if z[i] > 0 and dz[i] < 0 and ok[i]:
            j = i
            pmin = pmax = phi[i]
            qmin = qmax = theta[i]
            while j + 1 < n and ok[j + 1]:
                if var_th is not None:
                    pmin, pmax = min(pmin, phi[j + 1]), max(pmax, phi[j + 1])
                    qmin, qmax = min(qmin, theta[j + 1]), max(qmax, theta[j + 1])
                    if pmax - pmin > var_th or qmax - qmin > var_th:
                        break
                j += 1
            length = t[j] - t[i]
            if length >= min_len:
                windows.append((t[i], length))
            i = j + 1
        else:
            i += 1
    return windows


summary = {}
data = {}
for key, (folder, rfile, wfile, sfile, label) in RUNS.items():
    r = load(folder, rfile)
    t, z, phi, theta = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    dts = np.diff(t)
    fs = 1 / np.median(dts)
    w = load(folder, wfile)
    ws = spectral_stats(w[:, 0], w[:, 1])
    code, ssname = sea_state(ws["Hs_m"])
    rs = load(folder, sfile)

    zd = signal.detrend(z)
    dz = np.gradient(z, 1 / fs)
    stats = {}
    for name, x in [("heave_m", z), ("roll_deg", phi), ("pitch_deg", theta)]:
        f, p, fpk, above = welch_peak(x, fs)
        xd = signal.detrend(x)
        stats[name] = {
            "mean": float(np.mean(x)), "std": float(np.std(xd)), "min": float(np.min(x)),
            "max": float(np.max(x)), "sig_double_amp_4std": float(4 * np.std(xd)),
            "peak_freq_Hz": float(fpk), "peak_period_s": float(1 / fpk),
            "var_frac_above_2p5Hz": float(above),
            "decorrelation_time_s": float(decorrelation_time(x, fs)),
        }
    stats["heave_vel_m_s"] = {"std": float(np.std(dz)), "max_abs": float(np.max(np.abs(dz)))}

    win_paper = landing_windows(t, z, phi, theta, 0.5, 0.3, var_th=0.22)
    win_stanag = landing_windows(t, z, phi, theta, 2.5, 1.5)
    summary[key] = {
        "label": label,
        "n_samples": int(len(t)), "fs_Hz": float(fs), "duration_s": float(t[-1] - t[0]),
        "dt_uniform": bool(np.allclose(dts, np.median(dts), atol=1e-6)),
        "nan_count": int(np.isnan(r).sum()),
        "wave_spectrum_md5": md5(folder, wfile),
        "wave": {k: float(v) for k, v in ws.items()}, "sea_state_WMO": f"{code} ({ssname})",
        "motion": stats,
        "windows_paper_def1": {"count": len(win_paper),
                               "durations_s": [round(d, 2) for _, d in win_paper],
                               "n_ge_5s": sum(d >= 5 for _, d in win_paper),
                               "n_ge_19p6s": sum(d >= 19.6 for _, d in win_paper)},
        "windows_stanag_like": {"count": len(win_stanag),
                                "max_s": round(max([d for _, d in win_stanag], default=0), 2)},
    }
    data[key] = dict(t=t, z=z, phi=phi, theta=theta, fs=fs, w=w, rs=rs, label=label,
                     win_paper=win_paper)

    # tidy exports: 20 Hz original and 5 Hz (anti-aliased decimation, as used in the paper)
    hdr = "time_s,heave_m,roll_deg,pitch_deg"
    np.savetxt(os.path.join(OUT_CSV, f"{key}_20Hz.csv"), r, delimiter=",", header=hdr,
               comments="", fmt="%.6f")
    q = int(round(fs / 5))
    dec = np.column_stack([t[::q][: len(signal.decimate(z, q, zero_phase=True))]] +
                          [signal.decimate(x, q, zero_phase=True) for x in (z, phi, theta)])
    np.savetxt(os.path.join(OUT_CSV, f"{key}_5Hz.csv"), dec, delimiter=",", header=hdr,
               comments="", fmt="%.6f")

with open(os.path.join(ROOT, "analysis", "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2, default=lambda o: o.item())

# ---------- Figures ----------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
keys = list(RUNS)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for key in ["noWing_headQ", "noWing_followQ"]:
    d = data[key]
    for ax, x, lab in zip(axes, (d["z"] * 100, d["phi"], d["theta"]),
                          ("Heave z_s [cm]", "Roll φ_s [deg]", "Pitch θ_s [deg]")):
        ax.plot(d["t"], x, lw=0.6, label=d["label"])
        ax.set_ylabel(lab)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Thời gian [s]")
fig.suptitle("Chuỗi thời gian đo thật – tàu 10–12 m, Lávrio (Hy Lạp), 20 Hz (không cánh)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG, "01_timeseries_noWing.png"), dpi=130)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
d = data["noWing_headQ"]
m = (d["t"] >= 60) & (d["t"] <= 120)
for ax, x, lab in zip(axes, (d["z"][m] * 100, d["phi"][m], d["theta"][m]),
                      ("Heave [cm]", "Roll [deg]", "Pitch [deg]")):
    ax.plot(d["t"][m], x, lw=0.9, color="C0")
    ax.set_ylabel(lab)
for t0, L in d["win_paper"]:
    if 60 <= t0 <= 120:
        for ax in axes:
            ax.axvspan(t0, min(t0 + L, 120), color="C2", alpha=0.25)
axes[-1].set_xlabel("Thời gian [s]")
fig.suptitle("Phóng to 60–120 s (không cánh, sóng ngược chéo); vùng xanh = cửa sổ theo Định nghĩa 1 của bài")
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG, "02_zoom_windows.png"), dpi=130)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for key in ["noWing_headQ", "noWing_followQ"]:
    d = data[key]
    axes[0].plot(d["w"][:, 0], d["w"][:, 1], label=d["label"])
    for ax, x, lab in zip(axes[1:], (d["z"], d["phi"], d["theta"]), ("Heave", "Roll", "Pitch")):
        f, p, _, _ = welch_peak(x, d["fs"])
        ax.semilogy(f, p, label=d["label"])
        ax.set_title(f"PSD {lab}")
        ax.set_xlim(0, 3)
axes[0].set_title("Phổ sóng (phao)")
axes[0].set_xlim(0, 2)
for ax in axes:
    ax.set_xlabel("f [Hz]")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG, "03_spectra.png"), dpi=130)

fig, ax = plt.subplots(figsize=(8, 3.6))
names = ["Heave std [cm]", "Roll std [deg]", "Pitch std [deg]"]
xs = np.arange(3)
for i, key in enumerate(keys):
    s = summary[key]["motion"]
    vals = [s["heave_m"]["std"] * 100, s["roll_deg"]["std"], s["pitch_deg"]["std"]]
    ax.bar(xs + i * 0.2 - 0.3, vals, width=0.2, label=summary[key]["label"])
ax.set_xticks(xs, names)
ax.legend(fontsize=7)
ax.set_title("Mức độ chuyển động (độ lệch chuẩn) theo từng lần chạy")
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG, "04_motion_levels.png"), dpi=130)

print(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda o: o.item()))
