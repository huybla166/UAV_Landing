"""Batched PyTorch (GPU) version of sim/run_one.m for the design matrix.

Same model and loop as run_one.m with its dataset options (memory = true, waveImpl 'fast',
wavePhase 'position', waveFrame 'seakeeping', waveHold 'rk4', control 'autopilot', Vc = 0):
  dynamics      sim/hydroVesselMemFast.m   (Cummins fluid memory, inlined MSS helpers)
  wave loads    sim/waveLoads.m            (force RAOs, position phase, stacked RAO path)
  heading       MSS GNC/headingAutopilot.m (PID + 3rd-order reference model)
  speed         PI on surge, Kp = Msys(1,1)/10, Ki = Kp/10, feed-forward tauX0
  integration   sim/rk4u.m                 (wave load at t, t+h/2, t+h), h = 0.025 s (dataset_v2; v1 used 0.05),
                CSV logged at 20 Hz
All runs of one (vessel, damping) group are integrated together, float64.
Inputs come from sim/prep_gpu_inputs.m (model constants, wave models after rng(seed), JSON).
Outputs: <out>/runs/<name>.csv (22 columns, '%.7g', as run_one.m) and <out>/qc/qc_wgpu*.csv.

  python sim/batch_sim_gpu.py --out D:/USV-landing-prediction/data/sim/dataset_v1 --ids missing
"""
import argparse
import glob
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io as sio

torch = None   # imported in main(): the CSV writer processes (Windows spawn) must not load torch

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "t,zeta,tau_w1,tau_w2,tau_w3,tau_w4,tau_w5,tau_w6,N,E,D,phi,theta,psi,u,v,w,p,q,r,tauX,tauN"
T_FINAL, H, T_LP = 600.0, 0.025, 20.0       # H = integration step (--h); the CSV is logged every LOG_EVERY steps
LOG_EVERY = 2                                  # 0.025 s x 2 = 0.05 s = 20 Hz, same CSV format as before


# ----------------------------------------------------------------------------- inputs
def load_vessel(path, dev):
    S = sio.loadmat(path)
    V = {}
    for k, v in S.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v, dtype=np.float64)
        V[k] = torch.tensor(a.squeeze() if a.size == 1 else a, device=dev, dtype=torch.float64)
    V["Dref"] = V["Dref"].reshape(6)
    V["xL"] = V["xL"].reshape(-1)
    D = V["Dv"]
    V["Dv_diag"] = torch.diagonal(D).clone()
    V["Dv_offdiag"] = bool((D - torch.diag(torch.diagonal(D))).abs().max() > 0)
    V["Msys"] = V["MRB"] + V["MAinf"]
    return V


def load_waves(gpu_dir, vessel):
    """seed -> dict(Omega (Nf), mu (Nd), Amp (Nf,Nd), phases (Nf,Nd), RAO (Nf,Nang,12)), raoAngles, g."""
    out, angles, g = {}, None, None
    for f in sorted(glob.glob(os.path.join(gpu_dir, f"waves_{vessel}_rows*.mat"))):
        S = sio.loadmat(f)
        seeds = np.asarray(S["seeds"]).reshape(-1).astype(int)
        n = len(seeds)
        Om = np.asarray(S["Omega"]).reshape(n, -1)
        mu = np.asarray(S["mu"]).reshape(n, -1)
        Nf, Nd = Om.shape[1], mu.shape[1]
        Amp = np.asarray(S["Amp"]).reshape(Nf, Nd, n)
        ph = np.asarray(S["phases"]).reshape(Nf, Nd, n)
        R = np.asarray(S["RAOstack"])
        R = R.reshape(R.shape[0], R.shape[1], 12, n)
        a = np.asarray(S["raoAngles"]).reshape(-1)
        if angles is not None and not np.array_equal(a, angles):
            raise ValueError("RAO heading grids differ between wave files")
        angles, g = a, float(np.asarray(S["g"]).squeeze())
        for i, s in enumerate(seeds):
            out[int(s)] = dict(Omega=Om[i], mu=mu[i], Amp=Amp[:, :, i], phases=ph[:, :, i], RAO=R[:, :, :, i])
    return out, angles, g


# ----------------------------------------------------------------------------- model
def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


class Batch:
    def __init__(self, V, W, angles, g, rows, dev):
        B = len(rows)
        f64 = dict(device=dev, dtype=torch.float64)
        self.V, self.B, self.dev = V, B, dev
        self.k4 = torch.tensor(rows.kappa4.values, **f64)
        self.k5 = torch.tensor(rows.kappa5.values, **f64)
        self.k6 = torch.zeros(B, **f64)
        self.U = torch.tensor(rows.U.values, **f64)
        self.beta = torch.tensor(rows.beta_deg.values * math.pi / 180, **f64)
        ws = [W[int(s)] for s in rows.seed.values]
        self.Omega = torch.tensor(np.stack([w["Omega"] for w in ws]), **f64)          # B,Nf
        self.mu = torch.tensor(np.stack([w["mu"] for w in ws]), **f64)                # B,Nd
        self.Amp = torch.tensor(np.stack([w["Amp"] for w in ws]), **f64)              # B,Nf,Nd
        self.phases = torch.tensor(np.stack([w["phases"] for w in ws]), **f64)        # B,Nf,Nd
        self.RAO = torch.tensor(np.stack([w["RAO"] for w in ws]), **f64)              # B,Nf,Nang,12
        self.k = self.Omega ** 2 / g
        self.dAng = float(angles[1] - angles[0])
        if np.any(np.abs(np.diff(angles) - self.dAng) > 1e-9):
            raise ValueError("RAO heading grid must be uniform")
        self.nAng = len(angles)
        self.chi = self.beta[:, None] + self.mu                                        # B,Nd
        self.cchi, self.schi = torch.cos(self.chi), torch.sin(self.chi)
        Nf = self.Omega.shape[1]
        self.bidx = torch.arange(B, device=dev)[:, None, None]
        self.fidx = torch.arange(Nf, device=dev)[None, :, None]

    # sim/waveLoads.m, phaseMode 'position', RAOstack path
    def wave(self, t, posN, posE, psi, method="scatter"):
        beta_rao = torch.remainder((self.beta - psi)[:, None] + self.mu, 2 * math.pi)  # B,Nd
        pos = beta_rao / self.dAng
        i0 = torch.clamp(torch.floor(pos), 0, self.nAng - 2)                           # 0-based left node
        wgt = pos - i0                                                                  # B,Nd
        i0 = i0.long()
        phase = (self.Omega * t)[:, :, None] - self.k[:, :, None] * (
            posN[:, None, None] * self.cchi[:, None, :] + posE[:, None, None] * self.schi[:, None, :]) + self.phases
        c = self.Amp * torch.cos(phase)
        s = self.Amp * torch.sin(phase)
        zeta = c.sum(dim=(1, 2))
        if method == "gather":   # literal transcription: interpolate the RAO per direction, then sum
            w4 = wgt[:, None, :, None]
            i4 = i0[:, None, :]
            P = self.RAO[self.bidx, self.fidx, i4] * (1 - w4) + self.RAO[self.bidx, self.fidx, i4 + 1] * w4
            tau = torch.einsum("bfd,bfdk->bk", c, P[..., :6]) - torch.einsum("bfd,bfdk->bk", s, P[..., 6:])
            return tau, zeta
        # same sum, reordered: spread each direction's weights onto its two RAO heading nodes, then
        # contract with the RAO table (no B x Nf x Nd x 12 temporary)
        B, Nf, Nd = c.shape
        idx = i0[:, None, :].expand(B, Nf, Nd)
        w = wgt[:, None, :]
        Cc = torch.zeros(B, Nf, self.nAng, device=c.device, dtype=c.dtype)
        Cs = torch.zeros_like(Cc)
        Cc.scatter_add_(2, idx, c * (1 - w)); Cc.scatter_add_(2, idx + 1, c * w)
        Cs.scatter_add_(2, idx, s * (1 - w)); Cs.scatter_add_(2, idx + 1, s * w)
        tau = torch.einsum("bfa,bfak->bk", Cc, self.RAO[..., :6]) - torch.einsum("bfa,bfak->bk", Cs, self.RAO[..., 6:])
        return tau, zeta

    # sim/hydroVesselMemFast.m with Vc = 0
    def dyn(self, x, tau):
        V = self.V
        nu, eta, xr = x[:, :6], x[:, 6:12], x[:, 12:]
        v1, w = nu[:, :3], nu[:, 3:6]
        Igw = w @ V["Ig"].T
        crb = torch.cat([V["m"] * cross(w, v1), -cross(Igw, w)], dim=1)
        a1 = v1 @ V["M11"].T + w @ V["M12"].T
        a2 = v1 @ V["M21"].T + w @ V["M22"].T
        ca = torch.cat([-cross(a1, w), -cross(a1, v1) - cross(a2, w)], dim=1)
        u = nu[:, 0]
        Re = torch.clamp(V["L"] * u.abs() / 1e-6, min=1e5)
        Xuu = V["Xuu_const"] * 0.075 / (torch.log10(Re) - 2) ** 2
        d = nu @ V["Dv"].T if V["Dv_offdiag"] else nu * V["Dv_diag"]
        d = d.clone()
        d[:, 0] = (V["Dv_diag"][0] * torch.exp(-3 * u.abs()) - Xuu * u.abs()) * u
        d[:, 3] = d[:, 3] + self.k4 * V["Dref"][3] * nu[:, 3].abs() * nu[:, 3]
        d[:, 4] = d[:, 4] + self.k5 * V["Dref"][4] * nu[:, 4].abs() * nu[:, 4]
        d[:, 5] = d[:, 5] + V["Dv_diag"][5] * self.k6 * nu[:, 5].abs() * nu[:, 5]
        xL = V["xL"]
        Uh = nu[:, 1:2] + xL * nu[:, 5:6]
        Uv = nu[:, 2:3] + xL * nu[:, 4:5]
        Qh, Qv = Uh.abs() * Uh, Uv.abs() * Uv
        z = torch.zeros_like(u)
        tcf = -V["kcf"] * torch.stack([z, Qh.sum(1), Qv.sum(1), z, Qv @ xL, Qh @ xL], dim=1)
        mu = xr @ V["Cr"].T + nu @ V["Dr"].T
        rhs = tau + tcf - crb - ca - d - mu - eta @ V["G"].T
        nudot = rhs @ V["Minv_inf"].T
        cphi, sphi = torch.cos(eta[:, 3]), torch.sin(eta[:, 3])
        cth, sth = torch.cos(eta[:, 4]), torch.sin(eta[:, 4])
        cpsi, spsi = torch.cos(eta[:, 5]), torch.sin(eta[:, 5])
        uu, vv, ww = nu[:, 0], nu[:, 1], nu[:, 2]
        p, q, r = nu[:, 3], nu[:, 4], nu[:, 5]
        Nd = cpsi * cth * uu + (-spsi * cphi + cpsi * sth * sphi) * vv + (spsi * sphi + cpsi * cphi * sth) * ww
        Ed = spsi * cth * uu + (cpsi * cphi + sphi * sth * spsi) * vv + (-cpsi * sphi + sth * spsi * cphi) * ww
        Dd = -sth * uu + cth * sphi * vv + cth * cphi * ww
        phid = p + sphi * sth / cth * q + cphi * sth / cth * r
        thd = cphi * q - sphi * r
        psid = sphi / cth * q + cphi / cth * r
        xrdot = xr @ V["Ar"].T + nu @ V["Br"].T
        return torch.cat([nudot, torch.stack([Nd, Ed, Dd, phid, thd, psid], dim=1), xrdot], dim=1)

    def rk4u(self, x, u0, um, u1):
        k1 = self.dyn(x, u0)
        k2 = self.dyn(x + 0.5 * H * k1, um)
        k3 = self.dyn(x + 0.5 * H * k2, um)
        k4 = self.dyn(x + H * k3, u1)
        return x + (H / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self):
        V, B, dev = self.V, self.B, self.dev
        f64 = dict(device=dev, dtype=torch.float64)
        nMem = V["Ar"].shape[0]
        Msys = V["Msys"]
        x = torch.zeros(B, 12 + nMem, **f64)
        x[:, 0] = self.U
        xd0 = self.dyn(x, torch.zeros(B, 6, **f64))
        tauX0 = -(xd0[:, :6] @ Msys.T)[:, 0]
        Kp_u = Msys[0, 0] / 10
        Ki_u = Kp_u / 10
        u_int = torch.zeros(B, **f64)
        # heading autopilot (MSS), psi_ref = 0
        M66, wn, zeta_h, r_max = Msys[5, 5], V["hc_wn"], V["hc_zeta"], V["hc_r_max"]
        wn_d = wn / 20
        Kp, Kd = M66 * wn ** 2, M66 * 2 * zeta_h * wn
        Ki = 1 / 10 * Kp * wn
        z_int = torch.zeros(B, **f64); a_d = torch.zeros(B, **f64)
        r_d = torch.zeros(B, **f64); psi_d = torch.zeros(B, **f64)

        def ssa(a):
            return torch.remainder(a + math.pi, 2 * math.pi) - math.pi

        n = int(round(T_FINAL / H)) + 1
        n_log = (n - 1) // LOG_EVERY + 1
        FL = 1000                                   # GPU log buffer, flushed to host every FL logged rows
        log = np.empty((n_log, B, 22))
        buf = torch.zeros(FL, B, 22, **f64)
        vbarN, vbarE = self.U * torch.cos(x[:, 11]), self.U * torch.sin(x[:, 11])
        posN, posE, psiS = x[:, 6].clone(), x[:, 7].clone(), x[:, 11].clone()
        tau_w, zeta = self.wave(0.0, x[:, 6], x[:, 7], x[:, 11])
        bad = torch.zeros(B, dtype=torch.bool, device=dev)
        for i in range(n):
            t = i * H
            nu, eta = x[:, :6], x[:, 6:12]
            e_u = self.U - nu[:, 0]
            u_int = u_int + H * e_u
            tauX = tauX0 + Kp_u * e_u + Ki_u * u_int
            a_d_dot = -wn_d ** 3 * ssa(psi_d) - 3 * wn_d ** 2 * r_d - 3 * wn_d * a_d
            tauN = -Kp * ssa(eta[:, 5] - psi_d) - Kd * (nu[:, 5] - r_d) - Ki * z_int
            z_int = z_int + H * ssa(eta[:, 5] - psi_d)
            psi_d = psi_d + H * r_d
            r_d = r_d + H * a_d
            a_d = a_d + H * a_d_dot
            r_d = torch.where(r_d.abs() > r_max, torch.sign(r_d) * r_max, r_d)
            z0 = torch.zeros_like(tauX)
            tau = torch.stack([tauX, z0, z0, z0, z0, tauN], dim=1)
            if i % LOG_EVERY == 0:
                j = i // LOG_EVERY
                buf[j % FL] = torch.cat([torch.full((B, 1), t, **f64), zeta[:, None], tau_w, eta, nu,
                                         tauX[:, None], tauN[:, None]], dim=1)
                if j % FL == FL - 1 or j == n_log - 1:
                    m = j % FL + 1
                    log[j - m + 1:j + 1] = buf[:m].cpu().numpy()
            # seakeeping frame
            cps, sps = torch.cos(eta[:, 5]), torch.sin(eta[:, 5])
            vN = cps * nu[:, 0] - sps * nu[:, 1]
            vE = sps * nu[:, 0] + cps * nu[:, 1]
            vbarN = vbarN + (H / T_LP) * (vN - vbarN)
            vbarE = vbarE + (H / T_LP) * (vE - vbarE)
            pN0, pE0, ps0 = posN, posE, psiS
            posN = posN + H * vbarN
            posE = posE + H * vbarE
            psiS = psiS + (H / T_LP) * (torch.remainder(eta[:, 5] - psiS + math.pi, 2 * math.pi) - math.pi)
            tau_m, _ = self.wave(t + H / 2, 0.5 * (pN0 + posN), 0.5 * (pE0 + posE), 0.5 * (ps0 + psiS))
            tau_1, zeta_1 = self.wave(t + H, posN, posE, psiS)
            x = self.rk4u(x, tau + tau_w, tau + tau_m, tau + tau_1)
            tau_w, zeta = tau_1, zeta_1
            if i % 200 == 0:
                fin = torch.isfinite(x).all(dim=1)
                if not bool(fin.all()):
                    bad |= ~fin
                    x = torch.where(fin[:, None], x, torch.zeros_like(x))
        return log, bad.cpu().numpy()


# ----------------------------------------------------------------------------- outputs
def write_csv(args):
    path, arr = args
    tmp = path + ".tmp"
    fmt = ",".join(["%.7g"] * arr.shape[1]) + "\n"        # same bytes as np.savetxt(fmt='%.7g'), faster
    with open(tmp, "w", newline="\n") as f:
        f.write(HEADER + "\n")
        f.write("".join(fmt % tuple(r) for r in arr.tolist()))
    os.replace(tmp, path)
    return path


def summary(arr):
    k0 = int(np.argmax(arr[:, 0] >= min(30.0, T_FINAL / 2)))
    a = arr[k0:]
    return dict(Hs_real=4 * a[:, 1].std(ddof=1), U_mean=a[:, 14].mean(), std_heave_m=a[:, 10].std(ddof=1),
                std_roll_deg=np.degrees(a[:, 11].std(ddof=1)), std_pitch_deg=np.degrees(a[:, 12].std(ddof=1)),
                max_roll_deg=np.degrees(np.abs(a[:, 11]).max()))


def main():
    global torch, H, LOG_EVERY, T_FINAL
    import torch as _torch
    torch = _torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.025, help="integration step (s); the CSV stays at 20 Hz")
    ap.add_argument("--T", type=float, default=600.0, help="simulated length (s); only for tests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default=os.path.join(HERE, "design_matrix.csv"))
    ap.add_argument("--ids", default="missing", help="'missing' (no CSV yet), 'all', or 'a-b'")
    ap.add_argument("--batch", type=int, default=1400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--tag", default="gpu")
    ap.add_argument("--group", default="", help="only this (vessel,dzeta3) group, e.g. otter_55kg,0")
    a = ap.parse_args()
    H, T_FINAL = a.h, a.T
    LOG_EVERY = int(round(0.05 / H))
    if abs(LOG_EVERY * H - 0.05) > 1e-12:
        raise SystemExit("--h must divide 0.05 s (20 Hz logging)")

    dm = pd.read_csv(a.design)
    run_dir, qc_dir, gpu_dir = (os.path.join(a.out, s) for s in ("runs", "qc", "gpu_inputs"))
    os.makedirs(run_dir, exist_ok=True); os.makedirs(qc_dir, exist_ok=True)
    if a.ids == "all":
        sel = dm
    elif a.ids == "missing":
        sel = dm[[not os.path.exists(os.path.join(run_dir, n + ".csv")) for n in dm.name]]
    else:
        lo, hi = (int(v) for v in a.ids.split("-"))
        sel = dm[(dm.id >= lo) & (dm.id <= hi)]
    if a.group:
        gv, gz = a.group.split(",")
        sel = sel[(sel.vessel == gv) & np.isclose(sel.dzeta3, float(gz))]
    print(f"{len(sel)} runs on {a.device}", flush=True)
    qc_file = os.path.join(qc_dir, f"qc_w{a.tag}.csv")
    if not os.path.exists(qc_file):
        with open(qc_file, "w") as f:
            f.write("id,name,status,wall_s,Hs_target,Hs_real,U_target,U_mean,std_heave_m,std_roll_deg,"
                    "std_pitch_deg,max_roll_deg,message\n")
    pool = Pool(a.writers)
    pending = []
    t_all = time.time()
    for (vessel, dz3, dz5), grp in sel.groupby(["vessel", "dzeta3", "dzeta5"]):
        V = load_vessel(os.path.join(gpu_dir, f"vessel_{vessel}_dz{dz3:g}_{dz5:g}.mat"), a.device)
        W, angles, g = load_waves(gpu_dir, vessel)
        missing = set(int(s) for s in grp.seed) - set(W)
        if missing:
            raise RuntimeError(f"no wave model for seeds {sorted(missing)[:5]}... ({vessel}); run prep_gpu_inputs.m")
        for c0 in range(0, len(grp), a.batch):
            rows = grp.iloc[c0:c0 + a.batch]
            t0 = time.time()
            logs, bad = Batch(V, W, angles, g, rows, a.device).run()
            wall = time.time() - t0
            jobs, qc_lines = [], []
            for b, (_, row) in enumerate(rows.iterrows()):
                arr = logs[:, b, :]
                msg = ""
                if bad[b]:
                    status, msg = "error", "State diverged"
                    s = dict(Hs_real=np.nan, U_mean=np.nan, std_heave_m=np.nan, std_roll_deg=np.nan,
                             std_pitch_deg=np.nan, max_roll_deg=np.nan)
                else:
                    s = summary(arr)
                    fails = []
                    if abs(s["Hs_real"] / row.Hs - 1) > 0.25: fails.append("Hs")
                    if abs(s["U_mean"] - row.U) > 0.10: fails.append("speed")
                    if s["max_roll_deg"] > 30: fails.append("roll>30deg")
                    status = "qc_fail" if fails else "ok"
                    msg = "+".join(fails)
                    jobs.append((os.path.join(run_dir, row["name"] + ".csv"), arr))
                qc_lines.append(f"{row.id},{row['name']},{status},{wall / len(rows):.1f},{row.Hs:.3f},{s['Hs_real']:.4f},"
                                f"{row.U:.2f},{s['U_mean']:.4f},{s['std_heave_m']:.5f},{s['std_roll_deg']:.4f},"
                                f"{s['std_pitch_deg']:.4f},{s['max_roll_deg']:.3f},{msg}\n")
            for p in pending:
                p.wait()
            pending = [pool.map_async(write_csv, jobs, chunksize=4)]
            with open(qc_file, "a") as f:
                f.writelines(qc_lines)
            print(f"  {vessel} dz=({dz3:g},{dz5:g}) rows {c0 + 1}-{c0 + len(rows)} of {len(grp)}: "
                  f"{wall:.0f} s simulate, {int(bad.sum())} diverged", flush=True)
    for p in pending:
        p.wait()
    pool.close(); pool.join()
    print(f"done: {len(sel)} runs in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
