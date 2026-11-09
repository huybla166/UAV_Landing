"""Linearised Cummins model per coefficient group, as dense tensors for a 5 Hz training loss (precompute).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/cummins_coef.py            # build + verify (~5 min, < 3 GB)
  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/cummins_coef.py --quick    # fewer runs in the checks

The coefficients of a run (data/sim/dataset_v2/runs/<name>_coef.json, read by experiments/pinn/physics.py) depend only
on the vessel (55 / 80 kg), the viscous variant (dzeta3 > 0, a larger Dv), the forward speed U of the linearisation
and kappa (quadratic roll/pitch damping 0 or 5). The design has 18 of the 2 x 2 x 3 x 2 = 24 combinations (the viscous
runs all have kappa = 0); `main` checks that every coef.json of a group carries the same matrices.

Per group g, the residual the training loss evaluates on 5 Hz sequences (rows heave/roll/pitch = 2, 3, 4 used):

  R(eta, nu)[k] = M nu_dot[k] + F0 + D_lin (nu[k] - nu0) + quad(nu[k]) + mu[k] + G eta[k] - tau_ctrl - tau_wave
  mu[k]   = sum_{j=0..J} K[j] nu[k - j]                  (radiation memory, FIR, direct term Dr in K[0])
  quad    = c_quad * |nu[3:6]| nu[3:6] in rows 3, 4, 5  (kappa4 Dref4, kappa5 Dref5, kappa6 Dv66; physics.py)
  nu_dot  = phylstm_kaggle.ddt(nu)                       (4th-order finite differences, 0.2 s)

nu0 = (U, 0, 0, 0, 0, 0); F0 = lhs_static(nu0, 0), zero in rows 1-5; D_lin = d lhs_static / d nu at nu0 (Coriolis,
added-mass Coriolis, linear damping, the surge-resistance slope; cross-flow drag and |.|. terms have zero slope).
tau_ctrl = (tauX, 0, 0, 0, 0, tauN) is zero in rows 2-4. In the subtraction form R(y_hat) - R(y), F0, nu0, tau_ctrl and
tau_wave cancel.

FIR: exact discretisation of xr' = Ar xr + Br nu, mu = Cr xr + Dr nu with nu linear between samples (first-order hold,
physics.VesselModel._discretize):  xr[k+1] = Phi xr[k] + F_a nu[k] + F_b nu[k+1], hence
  K[0] = Cr F_b + Dr,   K[j] = Cr Phi^(j-1) (F_a + Phi F_b)   (j >= 1),
which reproduces the state-space recursion started at rest exactly (up to the truncation at J taps). J = 300 (60 s)
covers >= 5 x the slowest memory time constant (9.3 s / 11.8 s).

Output data/sim/dataset_v2/kaggle_phylstm_v2/cummins_groups.npz (float32 except the keys):
  vessel (G,) str, visc (G,) 0/1, U (G,), kappa (G,)    the group keys, groups sorted by (vessel, visc, U, kappa)
  M (G,6,6) D_lin (G,6,6) G (G,6,6) F0 (G,6) nu0 (G,6) c_quad (G,3) K (G,J+1,6,6) h (0.2)
  run_group (3150,) int16 group of each run, rows in cache order; ids (3150,) run ids (same order as the packs)
Verification printed (and written to results/sim/phylstm/cummins_coef_check.json, log results/sim/phylstm/
log_cummins_coef.txt, 05/11/2026). Rows heave/roll/pitch:
  FIR vs the 5 Hz state-space recursion (8 runs per group, k >= 300): max rel. error 1e-14 / 8.0e-3 / 2.2e-8; the
    roll error is the 60 s truncation (a slow roll memory mode; a 20 s kernel leaves 14 % of mu_roll, 10 s 48 %).
  FIR on 5 Hz nu vs the 20 Hz memory force decimated like the cache: median 4.4 / 16 / 4.7 % (first-order hold at
    0.2 s); this barely moves the residual (decomposition below), so the causal hold is kept.
  Linearised residual on the true 5 Hz data, RMS / std(tau_wave), 3126 QC-ok runs, samples 300 .. T-3:
    median 4.3 / 6.6 / 7.3 %, p90 16 / 14 / 24 %, max 98 / 52 / 115 % (short waves at U = 2 m/s).
    With a 20 s kernel 4.3 / 7.0 / 7.3 %, 10 s 4.3 / 11 / 7.3 %, 5 s 4.4 / 16 / 7.3 %.
  Decomposition on 6 runs (median): 5 Hz FD + 5 Hz FIR 4.6 / 6.5 / 8.0 %; with the decimated 20 Hz acceleration
    3.8 / 3.4 / 4.5 %; also with the decimated 20 Hz memory force 2.6 / 1.4 / 3.9 % (linearisation and the
    decimation of the nonlinear terms). Hence the training loss uses J = ||R(y_hat) - R(y)||^2.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pinn"))
from sim_common import CSV_COLS, SIM, Q, TRIM_START, TRIM_END, load  # noqa: E402
import physics  # noqa: E402

H = 0.2
J = 300                                   # taps 0..J, 60 s
TRUNC = (5, 10, 25, 50, 100)              # shorter kernels checked on real nu (taps)
RES_TRUNC = (10, 25, 50, 100)             # and in the residual on every run
OUT = os.path.join(SIM, "kaggle_phylstm_v2")
CHECK = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "sim", "phylstm", "cummins_coef_check.json")
KEYS = ["vessel", "visc", "U", "kappa"]
SAME = ["M_inf", "MA_inf", "MRB", "G", "Dv", "Dref", "kappa", "main"]   # must not vary inside a group (+ memory)


def group_table(meta):
    """Group keys per run and the sorted list of distinct groups."""
    keys = list(zip(meta.vessel, (meta.dzeta3 > 0).astype(int), meta.U.astype(float), meta.kappa4.astype(float)))
    groups = sorted(set(keys))
    gi = {k: i for i, k in enumerate(groups)}
    return groups, np.array([gi[k] for k in keys], np.int16)


def coef_path(name):
    return os.path.join(SIM, "runs", name + "_coef.json")


def fingerprint(c):
    """The parts of a coef.json the model uses (run/environment/constant_model vary with the sea state)."""
    parts = [np.asarray(c[k], float).ravel() for k in SAME if k != "main"]
    parts += [np.array([c["main"][k] for k in ("m", "Lpp", "B", "T", "k44", "k55", "k66")], float)]
    parts += [np.asarray(c["memory"][k], float).ravel() for k in ("Ar", "Br", "Cr", "Dr")]
    return np.concatenate(parts)


def fir_kernel(model, h=H, n_taps=J + 1):
    """K (n_taps, 6, 6) float64: mu[k] = sum_j K[j] nu[k-j] for the first-order-hold discretisation."""
    Phi, (Fa, Fb), _ = model._discretize(h)
    Phi, Fa, Fb, Cr, Dr = (x.numpy() for x in (Phi, Fa, Fb, model.Cr, model.Dr))
    K = np.empty((n_taps, 6, 6))
    K[0] = Cr @ Fb + Dr
    A = Fa + Phi @ Fb
    for j in range(1, n_taps):
        K[j] = Cr @ A
        A = Phi @ A
    return K, Phi, Fa, Fb


def fir_force(nu, K):
    """mu (B, T, 6) = causal FIR of nu (B, T, 6) with K (n, 6, 6); samples before t = 0 are taken as zero.
    Torch, any dtype/device: conv1d over time with the kernel flipped (a template for the training loss)."""
    n = K.shape[0]
    w = K.flip(0).permute(1, 2, 0)                                       # (out 6, in 6, n), w[..., n-1] = K[0]
    x = torch.nn.functional.pad(nu.transpose(1, 2), (n - 1, 0))
    return torch.nn.functional.conv1d(x, w).transpose(1, 2)


def ss_force(nu, Phi, Fa, Fb, Cr, Dr):
    """The same memory force from the discrete state-space recursion at rest (numpy float64)."""
    x = np.zeros((nu.shape[0], Phi.shape[0]))
    mu = np.empty_like(nu)
    mu[:, 0] = nu[:, 0] @ Dr.T
    for k in range(1, nu.shape[1]):
        x = x @ Phi.T + nu[:, k - 1] @ Fa.T + nu[:, k] @ Fb.T
        mu[:, k] = x @ Cr.T + nu[:, k] @ Dr.T
    return mu


def build(meta):
    groups, run_group = group_table(meta)
    t0 = time.time()
    out = {k: [] for k in ("M", "D_lin", "G", "F0", "nu0", "c_quad", "K")}
    models, info = [], []
    for g, key in enumerate(groups):
        names = meta.name[run_group == g].tolist()
        ref = json.load(open(coef_path(names[0])))
        fp = fingerprint(ref)
        for n in names[1:]:                                              # every coef.json of the group is the same
            d = np.max(np.abs(fingerprint(json.load(open(coef_path(n)))) - fp))
            if d > 1e-9 * max(1.0, np.max(np.abs(fp))):
                raise RuntimeError(f"group {key}: {n} differs from {names[0]} by {d}")
        m = physics.VesselModel(ref)
        nu0, F0, Dlin = (x.numpy() for x in m.linearize(float(key[2])))
        k4, k5, k6 = m.kappa
        K, *_ = fir_kernel(m)
        Klong, *_ = fir_kernel(m, n_taps=2001)                           # 400 s, for the truncation check
        a = np.abs(Klong[:, 2:5]).sum(axis=(1, 2))                      # l1 of the heave/roll/pitch rows per lag
        tail = a[J + 1:].sum() / a.sum()
        cum = np.cumsum(a) / a.sum()
        ev = np.linalg.eigvals(m.Ar.numpy())
        info.append(dict(group=g, key=[key[0], int(key[1]), key[2], key[3]], runs=len(names), n_mem=m.n_mem,
                         slowest_tau_s=float((-1 / ev.real).max()), tail_l1_rows345_after_60s=float(tail),
                         l1_rows345_within_s={s: float(cum[int(s / H)]) for s in (1, 2, 5, 10, 30)}))
        out["M"].append(m.M.numpy()); out["D_lin"].append(Dlin); out["G"].append(m.G.numpy())
        out["F0"].append(F0); out["nu0"].append(nu0)
        out["c_quad"].append([k4 * float(m.Dref[3]), k5 * float(m.Dref[4]), k6 * float(m.Dv[5, 5])])
        out["K"].append(K)
        models.append(m)
        print(f"group {g:2d} {key}: {len(names)} runs, n_mem {m.n_mem}, slowest tau {info[-1]['slowest_tau_s']:.1f} s, "
              f"FIR tail after 60 s {tail:.1e} (l1, rows 3-5); l1 within 2 s {cum[10]:.3f}, 10 s {cum[50]:.4f}"
              f" ({time.time() - t0:.0f} s)", flush=True)
    arr = {k: np.asarray(v, np.float64) for k, v in out.items()}
    return groups, run_group, arr, models, info


def raw_reference(name, model):
    """Memory force from the 20 Hz CSV (physics.memory_states, cubic hold, the 'full' level's 0.0x % residual path),
    decimated and trimmed exactly as the 5 Hz cache (sim_common._one): the best available 'true' mu at 5 Hz."""
    from scipy import signal
    x = np.loadtxt(os.path.join(SIM, "runs", name + ".csv"), delimiter=",", skiprows=1)
    nu = torch.as_tensor(x[:, 14:20])
    h = float(x[1, 0] - x[0, 0])
    mu = model.memory_force(nu, model.memory_states(nu, h)).numpy()
    y = signal.decimate(mu, Q, axis=0, zero_phase=True)
    return y[TRIM_START:len(y) - TRIM_END]


def verify(meta, X, groups, run_group, arr, models, quick):
    from phylstm_kaggle import ddt
    c = {k: i for i, k in enumerate(CSV_COLS)}
    nu_cols = [c[k] for k in ("u", "v", "w", "p", "q", "r")]
    eta_cols = [c[k] for k in ("N", "E", "D", "phi", "theta", "psi")]
    tw_cols = [c[f"tau_w{i}"] for i in range(1, 7)]
    ok = (meta.status == "ok").to_numpy()
    rng = np.random.default_rng(0)
    res = {"fir_vs_ss": [], "fir_vs_raw20Hz": [], "residual": []}
    rel_all = []
    for g, key in enumerate(groups):
        rows = np.nonzero((run_group == g) & ok)[0]
        m = models[g]
        K = torch.as_tensor(arr["K"][g])
        # 1. FIR vs the discrete state-space recursion at rest, on whole 5 Hz runs (k >= 0: truncation shows after 60 s)
        sub = rng.choice(rows, 3 if quick else 8, replace=False)
        nu = np.asarray(X[sub][..., nu_cols], np.float64)
        _, Phi, Fa, Fb = fir_kernel(m, n_taps=1)
        mu_ss = ss_force(nu, Phi, Fa, Fb, m.Cr.numpy(), m.Dr.numpy())
        mu_fir = fir_force(torch.as_tensor(nu), K).numpy()
        e = np.sqrt(((mu_fir - mu_ss)[:, J:] ** 2).mean((0, 1))) / np.sqrt((mu_ss[:, J:] ** 2).mean((0, 1)))
        res["fir_vs_ss"].append(e[2:5].tolist())
        # 1b. a shorter kernel: relative error of mu (rows 3-5) when only the first L taps are kept
        for L in TRUNC:
            mu_L = fir_force(torch.as_tensor(nu), K[:L]).numpy()
            eL = np.sqrt(((mu_L - mu_fir)[:, J:] ** 2).mean((0, 1))) / np.sqrt((mu_fir[:, J:] ** 2).mean((0, 1)))
            res.setdefault(f"trunc_{L}", []).append(eL[2:5].tolist())
        # 2. FIR on the 5 Hz cache vs the decimated 20 Hz memory force (discretisation error of the 5 Hz FIR)
        sub2 = sub[:1 if quick else 2]
        for r in sub2:
            ref = raw_reference(meta.name[r], m)
            mf = fir_force(torch.as_tensor(np.asarray(X[r:r + 1][..., nu_cols], np.float64)), K).numpy()[0]
            e2 = np.sqrt(((mf - ref)[J:] ** 2).mean(0)) / np.sqrt((ref[J:] ** 2).mean(0))
            res["fir_vs_raw20Hz"].append(e2[2:5].tolist())
        # 3. linearised residual on the true 5 Hz data, every QC-ok run of the group
        Mg, Dg, Gg, F0, nu0 = (torch.as_tensor(arr[k][g]) for k in ("M", "D_lin", "G", "F0", "nu0"))
        cq = torch.as_tensor(arr["c_quad"][g])
        n0 = len(rel_all)
        for b in range(0, len(rows), 100):
            rr = rows[b:b + 100]
            Xb = np.asarray(X[rr], np.float64)
            nu = torch.as_tensor(Xb[..., nu_cols]); eta = torch.as_tensor(Xb[..., eta_cols])
            tw = torch.as_tensor(Xb[..., tw_cols])
            tc = torch.zeros_like(tw); tc[..., 0] = torch.as_tensor(Xb[..., c["tauX"]]); tc[..., 5] = torch.as_tensor(Xb[..., c["tauN"]])
            quad = torch.zeros_like(nu); quad[..., 3:] = cq * nu[..., 3:].abs() * nu[..., 3:]
            base = ddt(nu) @ Mg.T + F0 + (nu - nu0) @ Dg.T + quad + eta @ Gg.T - tc - tw
            sl = slice(J, -2)                                                # full memory history, central stencil
            relL = lambda L: ((base + fir_force(nu, K[:L]))[:, sl].pow(2).mean(1).sqrt() / tw[:, sl].std(1))[:, 2:5].numpy()
            rel = relL(J + 1)
            rel_all.append(rel)
            for L in RES_TRUNC:                                              # the same residual with a shorter kernel
                res.setdefault(f"res_trunc_{L}", []).append(relL(L))
            res["residual"] += [dict(id=int(meta.id[i]), group=g, rel=r_.tolist()) for i, r_ in zip(rr, rel)]
        print(f"group {g:2d}: FIR vs state space {np.array(res['fir_vs_ss'][-1])}, vs 20 Hz memory "
              f"{np.array(res['fir_vs_raw20Hz'][-1])}, residual median {np.median(np.concatenate(rel_all[n0:]), 0)}",
              flush=True)
    rel = np.concatenate(rel_all)
    summ = {
        "fir_vs_ss_max": np.max(res["fir_vs_ss"], 0).tolist(),
        "fir_vs_raw20Hz_median": np.median(res["fir_vs_raw20Hz"], 0).tolist(),
        "fir_vs_raw20Hz_max": np.max(res["fir_vs_raw20Hz"], 0).tolist(),
        "residual_median": np.median(rel, 0).tolist(),
        "residual_p90": np.percentile(rel, 90, 0).tolist(),
        "residual_max": rel.max(0).tolist(),
        "n_runs": int(len(rel)),
        "residual_median_with_kernel": {f"{L * H:g} s": np.median(np.concatenate(res[f"res_trunc_{L}"]), 0).tolist()
                                        for L in RES_TRUNC},
        "truncated_kernel_median": {f"{L * H:g} s": np.median(res[f"trunc_{L}"], 0).tolist() for L in TRUNC},
    }
    return summ, res


def residual_parts(meta, X, groups, run_group, arr, models, n_runs=6):
    """Where the 5 Hz residual comes from, on a few runs (20 Hz CSV): the same linearised residual with (a) the 5 Hz
    FD acceleration replaced by the decimated 20 Hz FD acceleration, (b) also the FIR memory replaced by the
    decimated 20 Hz memory force. Rows heave/roll/pitch, relative to the std of the wave load."""
    from scipy import signal
    from phylstm_kaggle import ddt
    c = {k: i for i, k in enumerate(CSV_COLS)}
    rng = np.random.default_rng(1)
    ok = np.nonzero((meta.status == "ok").to_numpy())[0]
    out = []
    for r in rng.choice(ok, n_runs, replace=False):
        g = int(run_group[r]); m = models[g]
        x = np.loadtxt(os.path.join(SIM, "runs", meta.name[r] + ".csv"), delimiter=",", skiprows=1)
        nu20 = torch.as_tensor(x[:, 14:20]); h = float(x[1, 0] - x[0, 0])
        dec = lambda y: signal.decimate(y, Q, axis=0, zero_phase=True)[TRIM_START:-TRIM_END].copy()
        acc20 = dec(np.concatenate([np.zeros((3, 6)), physics.ddt(nu20, h).numpy(), np.zeros((3, 6))]))
        mu20 = dec(m.memory_force(nu20, m.memory_states(nu20, h)).numpy())
        Xb = torch.as_tensor(np.asarray(X[r:r + 1], np.float64))
        nu = Xb[..., [c[k] for k in ("u", "v", "w", "p", "q", "r")]]
        eta = Xb[..., [c[k] for k in ("N", "E", "D", "phi", "theta", "psi")]]
        tw = Xb[..., [c[f"tau_w{i}"] for i in range(1, 7)]]
        Mg, Dg, Gg, F0, nu0 = (torch.as_tensor(arr[k][g]) for k in ("M", "D_lin", "G", "F0", "nu0"))
        quad = torch.zeros_like(nu); quad[..., 3:] = torch.as_tensor(arr["c_quad"][g]) * nu[..., 3:].abs() * nu[..., 3:]
        stat = F0 + (nu - nu0) @ Dg.T + quad + eta @ Gg.T - tw
        mu5 = fir_force(nu, torch.as_tensor(arr["K"][g]))
        rel = lambda R: (R[:, J:-2].pow(2).mean(1).sqrt() / tw[:, J:-2].std(1))[0, 2:5].numpy()
        out.append([rel(ddt(nu) @ Mg.T + stat + mu5),
                    rel(torch.as_tensor(acc20)[None] @ Mg.T + stat + mu5),
                    rel(torch.as_tensor(acc20)[None] @ Mg.T + stat + torch.as_tensor(mu20)[None])])
    o = np.median(np.array(out), 0)
    return {"5Hz_FD_and_FIR": o[0].tolist(), "20Hz_acc_5Hz_FIR": o[1].tolist(), "20Hz_acc_20Hz_mu": o[2].tolist(),
            "n_runs": n_runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    meta, X = load()
    groups, run_group, arr, models, info = build(meta)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "cummins_groups.npz")
    np.savez_compressed(path, vessel=np.array([k[0] for k in groups]), visc=np.array([k[1] for k in groups], np.int8),
                        U=np.array([k[2] for k in groups], np.float32), kappa=np.array([k[3] for k in groups], np.float32),
                        h=np.float32(H), run_group=run_group, ids=meta.id.to_numpy(),
                        **{k: v.astype(np.float32) for k, v in arr.items()})
    print(f"{len(groups)} groups -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
    summ, res = verify(meta, X, groups, run_group, arr, models, a.quick)
    parts = residual_parts(meta, X, groups, run_group, arr, models, 3 if a.quick else 6)
    print("summary (heave, roll, pitch):", json.dumps(summ, indent=1))
    print("residual decomposition (median, heave/roll/pitch):", json.dumps(parts, indent=1))
    os.makedirs(os.path.dirname(CHECK), exist_ok=True)
    json.dump({"groups": info, "summary": summ, "decomposition": parts, "quick": a.quick,
               "fir_vs_ss": res["fir_vs_ss"], "fir_vs_raw20Hz": res["fir_vs_raw20Hz"]},
              open(CHECK, "w"), indent=1)
    print(f"-> {CHECK}")


if __name__ == "__main__":
    main()
