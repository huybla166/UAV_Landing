"""Dataset-wide verification of data/sim/dataset_v1 (results: analysis/sim/V/, report verify_dataset_v1.md).

  python analysis/sim/verify_dataset_v1.py integrity   # V1 files, kinematics, per-run statistics -> V/run_stats.csv
  python analysis/sim/verify_dataset_v1.py forces      # V3 wave elevation and wave loads rebuilt from the wave model
  python analysis/sim/verify_dataset_v1.py regime      # V7 surge/yaw excursions, control effort vs Otter thrusters
  python analysis/sim/verify_dataset_v1.py linear      # V8 std of heave/roll/pitch vs linear prediction (fitted memory)
  python analysis/sim/verify_dataset_v1.py memory      # V5 passivity of the fitted memory, fit vs BEM in the wave band
  python analysis/sim/verify_dataset_v1.py report

V3 is an independent numpy transcription of sim/waveLoads.m + the seakeeping frame of sim/run_one.m
(third implementation after Octave and PyTorch): the logged zeta and tau_w1..6 are recomputed sample by
sample from the wave model that the run used (gpu_inputs/waves_*.mat: Omega, mu, Amp, phases, RAO) and
the seakeeping frame rebuilt from the logged psi, u, v.
V8 uses the fitted memory model K_fit(s) = Cr (sI - Ar)^-1 Br + Dr (analysis/sim/V/bem_*.mat), so the
only differences from the simulation are nonlinear terms, finite-record statistics and integration.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.environ.get("DATASET", "dataset_v1")          # DATASET=dataset_v2 python analysis/sim/verify_dataset_v1.py ...
SIM = os.path.join(ROOT, "data", "sim", DATASET)
RUNS = os.path.join(SIM, "runs")
V = os.path.join(ROOT, "analysis", "sim", "V" if DATASET == "dataset_v1" else "V_" + DATASET)
SCRATCH_GPU = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "claude", "D--USV-landing-prediction",
                           "328cea5c-0bb6-4b38-a98b-b4e3be358938", "scratchpad", "gpu_test", "gpu_inputs")
os.makedirs(V, exist_ok=True)
H, T_LP, N_ROWS, SKIP = 0.05, 20.0, 12001, 600
COLS = "t,zeta,tau_w1,tau_w2,tau_w3,tau_w4,tau_w5,tau_w6,N,E,D,phi,theta,psi,u,v,w,p,q,r,tauX,tauN".split(",")
# Otter thrusters (MSS CRAFT/USV/models/otter.m lines 112, 143-146): per propeller 119.7 N ahead, 66.7 N astern,
# lever arm 0.395 m -> surge [-133, 239] N, yaw moment <= (119.7 + 66.7) * 0.395 = 73.6 N m
TAUX_MAX, TAUX_MIN, TAUN_MAX = 2 * 0.5 * 24.4 * 9.81, -2 * 0.5 * 13.6 * 9.81, (0.5 * 24.4 + 0.5 * 13.6) * 9.81 * 0.395


def design():
    return pd.read_csv(os.path.join(SIM, "design_matrix.csv")).set_index("id")


def read_run(name):
    return pd.read_csv(os.path.join(RUNS, name + ".csv")).to_numpy(np.float64)


# ----------------------------------------------------------------------------- V1 integrity + statistics
def d4(x, h):
    """4th-order central difference along axis 0 (interior points 2..n-3)."""
    return (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * h)


def _integrity(args):
    rid, row = args
    out = dict(id=rid, name=row["name"], ok=True, msg="")
    try:
        x = read_run(row["name"])
        js = json.load(open(os.path.join(RUNS, row["name"] + "_coef.json")))
        r = js["run"]
        if x.shape != (N_ROWS, 22):
            raise ValueError(f"shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("non-finite values")
        if np.abs(x[:, 0] - H * np.arange(N_ROWS)).max() > 1e-6:
            raise ValueError("time grid")
        for k, v in dict(Hs=row.Hs, Tp=row.Tp, beta_deg=row.beta_deg, U_ref=row.U, seed=row.seed).items():
            if abs(float(r[k]) - float(v)) > 1e-9:
                raise ValueError(f"json {k} {r[k]} != design {v}")
        k456 = np.asarray(r["kappa_456"], float).ravel()
        dz = np.asarray(r["delta_zeta_345_used"], float).ravel()
        if abs(k456[0] - row.kappa4) > 1e-9 or abs(k456[1] - row.kappa5) > 1e-9:
            raise ValueError("json kappa != design")
        if abs(dz[0] - row.dzeta3) > 1e-9 or abs(dz[2] - row.dzeta5) > 1e-9:
            raise ValueError("json delta_zeta != design")
        out["engine"] = "pytorch" if "pytorch" in str(r.get("engine", "")) else "octave"
        # kinematics eta_dot = J(eta) nu, 4th-order differences, after 30 s
        eta, nu = x[:, 8:14], x[:, 14:20]
        phi, th, psi = eta[:, 3], eta[:, 4], eta[:, 5]
        cph, sph, cth, sth, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
        Dd = -sth * nu[:, 0] + cth * sph * nu[:, 1] + cth * cph * nu[:, 2]
        phid = nu[:, 3] + sph * sth / cth * nu[:, 4] + cph * sth / cth * nu[:, 5]
        thd = cph * nu[:, 4] - sph * nu[:, 5]
        psid = sph / cth * nu[:, 4] + cph / cth * nu[:, 5]
        num = d4(eta[:, [2, 3, 4, 5]], H)[SKIP:]
        ana = np.stack([Dd, phid, thd, psid], 1)[2:-2][SKIP:]
        for j, nm in enumerate(["D", "phi", "theta", "psi"]):
            out[f"kin_{nm}"] = np.sqrt(np.mean((num[:, j] - ana[:, j]) ** 2)) / max(ana[:, j].std(), 1e-12)
        a = x[SKIP:]
        for j, c in enumerate(COLS[1:], start=1):
            out[f"std_{c}"] = a[:, j].std()
        out["mean_u"], out["mean_psi"] = a[:, 14].mean(), a[:, 13].mean()
        out["max_du"] = np.abs(a[:, 14] - row.U).max()
        out["max_u"], out["min_u"] = a[:, 14].max(), a[:, 14].min()
        out["max_psi"] = np.abs(a[:, 13]).max()
        out["max_phi"], out["max_theta"] = np.abs(a[:, 11]).max(), np.abs(a[:, 12]).max()
        out["max_tauX"], out["min_tauX"], out["max_absN"] = a[:, 20].max(), a[:, 20].min(), np.abs(a[:, 21]).max()
        out["frac_tauX_sat"] = np.mean((a[:, 20] > TAUX_MAX) | (a[:, 20] < TAUX_MIN))
        out["frac_tauN_sat"] = np.mean(np.abs(a[:, 21]) > TAUN_MAX)
        out["Hs_real"] = 4 * a[:, 1].std()
    except Exception as e:                       # noqa: BLE001
        out["ok"], out["msg"] = False, str(e)
    return out


def integrity():
    D = design()
    with Pool(12) as pool:
        rows = pool.map(_integrity, list(D.iterrows()), chunksize=4)
    df = pd.DataFrame(rows).set_index("id").join(D.drop(columns=["name"]))
    df.to_csv(os.path.join(V, "run_stats.csv"))
    bad = df[~df.ok]
    print(f"integrity: {len(df)} runs, {len(bad)} failed")
    print(bad[["name", "msg"]].to_string() if len(bad) else "")
    print("kinematic residual (rel. RMS, 4th-order differences): max over runs",
          df[["kin_D", "kin_phi", "kin_theta", "kin_psi"]].max().round(5).to_dict())
    print(df.engine.value_counts().to_dict())


# ----------------------------------------------------------------------------- wave models
_WAVES = {}


def waves(vessel):
    if vessel in _WAVES:
        return _WAVES[vessel]
    out, angles, g = {}, None, None
    files = sorted(glob.glob(os.path.join(SIM, "gpu_inputs", f"waves_{vessel}_rows*.mat")))
    files += sorted(glob.glob(os.path.join(SCRATCH_GPU, f"waves_{vessel}_rows*.mat")))
    for f in files:
        S = sio.loadmat(f)
        seeds = np.asarray(S["seeds"]).reshape(-1).astype(int)
        n = len(seeds)
        Om, mu = np.asarray(S["Omega"]).reshape(n, -1), np.asarray(S["mu"]).reshape(n, -1)
        Nf, Nd = Om.shape[1], mu.shape[1]
        Amp, ph = np.asarray(S["Amp"]).reshape(Nf, Nd, n), np.asarray(S["phases"]).reshape(Nf, Nd, n)
        R = np.asarray(S["RAOstack"]); R = R.reshape(R.shape[0], R.shape[1], 12, n)
        angles, g = np.asarray(S["raoAngles"]).reshape(-1), float(np.asarray(S["g"]).squeeze())
        for i, s in enumerate(seeds):
            out.setdefault(int(s), dict(Omega=Om[i], mu=mu[i], Amp=Amp[:, :, i], phases=ph[:, :, i], RAO=R[:, :, :, i]))
    _WAVES[vessel] = (out, angles, g)
    return _WAVES[vessel]


def seakeeping_frame(x, U_ref):
    """posS (n,2), psiS (n,) of sim/run_one.m rebuilt from the logged psi, u, v (row i -> frame used for row i)."""
    n = len(x)
    psi, u, v = x[:, 13], x[:, 14], x[:, 15]
    vbar = np.array([U_ref, 0.0]); pos = np.zeros(2); ps = 0.0
    posS, psiS = np.zeros((n, 2)), np.zeros(n)
    for i in range(n - 1):
        vN = np.cos(psi[i]) * u[i] - np.sin(psi[i]) * v[i]
        vE = np.sin(psi[i]) * u[i] + np.cos(psi[i]) * v[i]
        vbar = vbar + (H / T_LP) * (np.array([vN, vE]) - vbar)
        pos = pos + H * vbar
        ps = ps + (H / T_LP) * ((psi[i] - ps + np.pi) % (2 * np.pi) - np.pi)
        posS[i + 1], psiS[i + 1] = pos, ps
    return posS, psiS


def wave_loads(t, w, angles, g, posS, psiS, beta, chunk=100):
    """zeta (n,), tau (n,6) as sim/waveLoads.m ('position' phase) at the given frame samples."""
    Om, mu, Amp, eps, R = w["Omega"], w["mu"], w["Amp"], w["phases"], w["RAO"]
    k = Om ** 2 / g
    dAng = angles[1] - angles[0]
    chi = beta + mu
    cchi, schi = np.cos(chi), np.sin(chi)
    n = len(t)
    zeta, tau = np.zeros(n), np.zeros((n, 6))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        tt, pN, pE, ps = t[a:b], posS[a:b, 0], posS[a:b, 1], psiS[a:b]
        beta_rao = np.mod((beta - ps)[:, None] + mu[None, :], 2 * np.pi)              # (c, Nd)
        pos = beta_rao / dAng
        i0 = np.clip(np.floor(pos).astype(int), 0, len(angles) - 2)                   # 0-based left node
        wgt = pos - i0
        P = R[:, i0, :] * (1 - wgt)[None, :, :, None] + R[:, i0 + 1, :] * wgt[None, :, :, None]   # (Nf, c, Nd, 12)
        phase = (Om[:, None] * tt[None, :])[:, :, None] - k[:, None, None] * (
            pN[None, :, None] * cchi[None, None, :] + pE[None, :, None] * schi[None, None, :]) + eps[:, None, :]
        c = Amp[:, None, :] * np.cos(phase)
        s = Amp[:, None, :] * np.sin(phase)
        zeta[a:b] = c.sum(axis=(0, 2))
        tau[a:b] = np.einsum("fcd,fcdk->ck", c, P[..., :6]) - np.einsum("fcd,fcdk->ck", s, P[..., 6:])
    return zeta, tau


def _forces(args):
    rid, row = args
    try:
        W, angles, g = waves(row.vessel)
        if int(row.seed) not in W:
            return dict(id=rid, name=row["name"], skipped="no wave model")
        x = read_run(row["name"])
        posS, psiS = seakeeping_frame(x, row.U)
        z, tau = wave_loads(x[:, 0], W[int(row.seed)], angles, g, posS, psiS, np.radians(row.beta_deg))
        out = dict(id=rid, name=row["name"], skipped="")
        out["zeta"] = np.abs(z - x[:, 1]).max() / x[:, 1].std()
        for k in range(6):
            out[f"tau{k + 1}"] = np.abs(tau[:, k] - x[:, 2 + k]).max() / max(x[:, 2 + k].std(), 1e-9)
        return out
    except Exception as e:                       # noqa: BLE001
        return dict(id=rid, name=row["name"], skipped=f"error {e}")


def forces():
    D = design()
    ids = sorted(set(D.index[D.index <= 40]) | set(D.index[::10]))
    with Pool(4) as pool:                       # 4 x ~0.3 GB: the machine has 16 GB and a small page file
        rows = pool.map(_forces, [(i, D.loc[i]) for i in ids], chunksize=2)
    df = pd.DataFrame(rows).set_index("id")
    df.to_csv(os.path.join(V, "force_reconstruction.csv"))
    ok = df[df.skipped == ""]
    print(f"forces: {len(ok)} runs rebuilt ({(df.skipped != '').sum()} skipped)")
    cols = ["zeta"] + [f"tau{k}" for k in range(1, 7)]
    print("max |rebuilt - logged| / std, worst run per column:")
    print(ok[cols].max().map(lambda v: f"{v:.2e}").to_string())
    print("median:", ok[cols].median().map(lambda v: f"{v:.1e}").to_dict())


# ----------------------------------------------------------------------------- V7 regime
def regime():
    df = pd.read_csv(os.path.join(V, "run_stats.csv"), index_col="id")
    df["c_p"] = 9.81 * df.Tp / (2 * np.pi)
    df["U_over_c"] = df.U / df.c_p
    df["std_psi_deg"] = np.degrees(df.std_psi); df["max_psi_deg"] = np.degrees(df.max_psi)
    df["violent"] = (df.max_du > 1.5) | (df.std_psi_deg > 5.0)   # heading hold lost, or surge swinging > 1.5 m/s
    df["ctrl_sat"] = (df.frac_tauX_sat > 0.01) | (df.frac_tauN_sat > 0.01)
    df.to_csv(os.path.join(V, "run_stats.csv"))
    pd.set_option("display.width", 200)
    print("surge excursion max|u-U| (m/s) and yaw std (deg), median [max] by (U, beta):")
    g = df.groupby(["U", "beta_deg"])
    t = pd.DataFrame({"max_du_med": g.max_du.median(), "max_du_max": g.max_du.max(),
                      "std_psi_med": g.std_psi_deg.median(), "std_psi_max": g.std_psi_deg.max(),
                      "n_violent": g.violent.sum(), "n": g.size(),
                      "tauX>max_1%": g.frac_tauX_sat.apply(lambda s: (s > 0.01).sum()),
                      "tauN>max_1%": g.frac_tauN_sat.apply(lambda s: (s > 0.01).sum())})
    print(t.round(3).to_string())
    print(f"\nviolent runs (max|u-U| > 1 m/s or yaw std > 5 deg): {df.violent.sum()} of {len(df)}")
    print(f"runs with control beyond Otter thruster limits > 1 % of the time: {df.ctrl_sat.sum()}")
    print(f"max |tauN| over dataset {df.max_absN.max():.1f} N m (limit {TAUN_MAX:.1f}); "
          f"tauX range [{df.min_tauX.min():.1f}, {df.max_tauX.max():.1f}] N (limits [{TAUX_MIN:.0f}, {TAUX_MAX:.0f}])")


# ----------------------------------------------------------------------------- V5 / V8 memory model
_BEM = {}


def bem(vessel, dz3, dz5):
    key = f"{vessel}_dz{dz3:g}_{dz5:g}"
    if key not in _BEM:
        tag = "" if DATASET == "dataset_v1" else "v3_"
        S = sio.loadmat(os.path.join(ROOT, "analysis", "sim", "V", f"bem_{tag}{key}.mat"))
        B = {k: np.asarray(v) for k, v in S.items() if not k.startswith("__")}
        lam, Vm = np.linalg.eig(B["Ar"])
        L = np.linalg.solve(Vm, B["Br"])                     # (n, 6)
        Rr = B["Cr"] @ Vm                                    # (6, n)
        B["lam"], B["Rr"], B["L"] = lam, Rr, L
        _BEM[key] = B
    return _BEM[key]


def K_fit(B, s):
    """K(s) = Cr (sI - Ar)^-1 Br + Dr for an array of s (complex), via the eigen-decomposition: (len(s), 6, 6)."""
    inv = 1.0 / (s[:, None] - B["lam"][None, :])            # (m, n)
    return np.einsum("in,mn,nj->mij", B["Rr"], inv, B["L"]) + B["Dr"][None]


def H_fit(B, w):
    """Vessel transfer function (len(w), 6, 6) of the fitted model at (signed) frequencies w."""
    s = 1j * w
    K = K_fit(B, s)
    Minf = B["MRB"] + B["MAinf"]
    A = -w[:, None, None] ** 2 * Minf[None] + s[:, None, None] * (K + B["Dv"][None]) + B["G"][None]
    return np.linalg.inv(A)


def lti_eigs(B):
    """Eigenvalues of the full linear system d/dt [eta; nu; xr] (U = 0 linearization)."""
    n = len(B["lam"]); Minf = B["MRB"] + B["MAinf"]; Mi = np.linalg.inv(Minf)
    A = np.zeros((12 + n, 12 + n))
    A[:6, 6:12] = np.eye(6)
    A[6:12, :6] = -Mi @ B["G"]; A[6:12, 6:12] = -Mi @ (B["Dv"] + B["Dr"]); A[6:12, 12:] = -Mi @ B["Cr"]
    A[12:, 6:12] = B["Br"]; A[12:, 12:] = B["Ar"]
    return np.linalg.eigvals(A)


def memory():
    lines, diag = [], []
    for vessel in ["otter_55kg", "otter_80kg"]:
        for dz in [(0, 0), (0.2, 0.3)]:
            B = bem(vessel, *dz)
            w = np.linspace(0.05, 30, 600)
            K = K_fit(B, 1j * w)
            Kh = 0.5 * (K + np.conj(np.transpose(K, (0, 2, 1))))
            ev = np.linalg.eigvalsh(Kh).min(axis=1)
            Bw = np.einsum("ijk->kij", B["B"])  # B stored (6,6,Nf) -> (Nf,6,6)
            evB = np.linalg.eigvalsh(0.5 * (Bw + np.transpose(Bw, (0, 2, 1)))).min(axis=1)
            s0 = 1j * 3.0
            K0 = B["Cr"] @ np.linalg.solve(s0 * np.eye(len(B["lam"])) - B["Ar"], B["Br"]) + B["Dr"]
            err = np.abs(K_fit(B, np.array([s0]))[0] - K0).max() / np.abs(K0).max()
            fr = B["freqs"].ravel(); fin = (fr < 10) & (fr > 0)
            Kb = Bw[fin] + 1j * fr[fin][:, None, None] * (np.einsum("ijk->kij", B["A"])[fin] - B["MAinf"][None])
            Kf = K_fit(B, 1j * fr[fin])
            rel = np.linalg.norm((Kf - Kb).reshape(fin.sum(), -1), axis=1) / np.linalg.norm(Kb.reshape(fin.sum(), -1), axis=1)
            lam = lti_eigs(B)
            lines.append(f"| {vessel} dz=({dz[0]:g},{dz[1]:g}) | {ev.min():.3g} at {w[ev.argmin()]:.1f} rad/s | "
                         f"{evB.min():.3g} | {err:.1e} | {np.median(rel):.3f} / {rel.max():.3f} at {fr[fin][rel.argmax()]:.1f} | "
                         f"{lam.real.max():.4f} |")
            # per-DOF diagonal: where Re K_kk < 0, its size relative to the BEM maximum of B_kk, and the
            # transfer-function magnitude above the wave band relative to its value at 9.5 rad/s
            Hd = H_fit(B, w)
            H95 = H_fit(B, np.array([9.5]))[0]
            for k, nm in enumerate(["surge", "sway", "heave", "roll", "pitch", "yaw"]):
                rk = K[:, k, k].real
                bmax = Bw[:, k, k].max()
                neg = rk < 0
                hf = w > 9.5
                diag.append(f"| {vessel} dz=({dz[0]:g},{dz[1]:g}) | {nm} | {rk.min():.3g} ({rk.min() / bmax * 100:.1f} % of max B_kk = {bmax:.3g}) | "
                            f"{(w[neg].min() if neg.any() else np.nan):.1f}-{(w[neg].max() if neg.any() else np.nan):.1f} | "
                            f"{np.abs(Hd[hf, k, k]).max() / np.abs(H95[k, k]):.2f} |")
    hdr = ["| memory model | min eig Re K_fit(jw), 0.05-30 rad/s | min eig B_BEM(w) | eigen-form check | rel. error |K_fit - K_BEM| median / max | max Re eig of the linear system |",
           "|---|---|---|---|---|---|"]
    hdr2 = ["", "| model | DOF | min Re K_kk(jw) | where Re K_kk < 0 (rad/s) | max |H_kk| above 9.5 rad/s / |H_kk(9.5)| |", "|---|---|---|---|---|"]
    txt = "\n".join(hdr + lines + hdr2 + diag)
    print(txt)
    open(os.path.join(V, "memory_passivity.md"), "w").write(txt + "\n")


def _linear(args):
    rid, row, st = args
    W, angles, g = waves(row.vessel)
    if int(row.seed) not in W:
        return None
    B = bem(row.vessel, row.dzeta3, row.dzeta5)
    w = W[int(row.seed)]
    Om, mu, Amp, R = w["Omega"], w["mu"], w["Amp"], w["RAO"]
    k = Om ** 2 / g
    beta = np.radians(row.beta_deg)
    U, psi_m = st["mean_u"], st["mean_psi"]
    chi = beta + mu
    we = Om[:, None] - k[:, None] * U * np.cos(chi - psi_m)[None, :]                 # (Nf, Nd) signed encounter freq
    beta_rao = np.mod(beta - psi_m + mu, 2 * np.pi)
    dAng = angles[1] - angles[0]
    pos = beta_rao / dAng
    i0 = np.clip(np.floor(pos).astype(int), 0, len(angles) - 2); wgt = pos - i0
    P = R[:, i0, :] * (1 - wgt)[None, :, None] + R[:, i0 + 1, :] * wgt[None, :, None]   # (Nf, Nd, 12)
    F = P[..., :6] + 1j * P[..., 6:]                                                    # force phasor per unit amp
    Hm = H_fit(B, we.ravel()).reshape(Om.size, mu.size, 6, 6)
    X = np.einsum("fdij,fdj->fdi", Hm, F) * Amp[:, :, None]                            # response phasors
    var = 0.5 * (np.abs(X) ** 2).sum(axis=(0, 1))                                      # (6,)
    hf = (np.abs(we) > 9.5)[:, :, None]
    var_hf = 0.5 * (np.abs(X) ** 2 * hf).sum(axis=(0, 1))
    out = dict(id=rid, max_abs_we=np.abs(we).max())
    for name, j, col in [("heave", 2, "std_D"), ("roll", 3, "std_phi"), ("pitch", 4, "std_theta")]:
        out[f"pred_{name}"] = np.sqrt(var[j]); out[f"sim_{name}"] = st[col]
        out[f"ratio_{name}"] = st[col] / np.sqrt(var[j])
        out[f"hf_frac_{name}"] = var_hf[j] / var[j]
    return out


def linear():
    D = design()
    st = pd.read_csv(os.path.join(V, "run_stats.csv"), index_col="id")
    with Pool(4) as pool:
        rows = pool.map(_linear, [(i, D.loc[i], st.loc[i]) for i in D.index], chunksize=8)
    df = pd.DataFrame([r for r in rows if r is not None]).set_index("id").join(D.drop(columns=["name"]))
    df["visc"] = df.dzeta3 > 0
    df = df.join(st[["violent"]])
    df.to_csv(os.path.join(V, "linear_prediction.csv"))
    pd.set_option("display.width", 200)
    print(f"linear: {len(df)} runs with a wave model")
    cols = ["ratio_heave", "ratio_roll", "ratio_pitch"]
    print("fraction of predicted response variance at encounter frequencies > 9.5 rad/s (outside the fitted band), by (U, beta):")
    print(df.groupby(["U", "beta_deg"])[["hf_frac_heave", "hf_frac_roll", "hf_frac_pitch", "max_abs_we"]].max().round(4).to_string())
    for by in [["kappa4"], ["visc"], ["U"], ["beta_deg"], ["Hs"], ["violent"]]:
        g = df.groupby(by)[cols]
        t = pd.concat([g.median().add_suffix("_med"), g.quantile(0.05).add_suffix("_p05"), g.quantile(0.95).add_suffix("_p95")], axis=1)
        print(f"\nsim std / linear prediction by {by}:")
        print(t.round(3).to_string())


def report():
    pass


if __name__ == "__main__":
    {"integrity": integrity, "forces": forces, "regime": regime, "linear": linear, "memory": memory,
     "report": report}[sys.argv[1]]()
