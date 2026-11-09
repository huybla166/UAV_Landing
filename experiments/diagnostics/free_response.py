"""Phase 2b physics diagnostics: what the vessel model alone can forecast (free response) and what the linearised
model costs when it is given the true wave load (oracle rows). CPU only, deterministic.

  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/free_response.py --splits cv0 --max-runs 20 --workers 4
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/free_response.py --workers 4      # all 8 splits (~1 h)
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/free_response.py --report-only
  D:/tools/venv-gpu/Scripts/python.exe experiments/diagnostics/free_response.py --splits cv1 cv3 --workers 2 \
      --runs otter_55kg_Hs0.40_Tp2.28_b090_U2_k5_r2 otter_55kg_Hs0.50_Tp2.55_b000_U2_k5_r1 --tag extreme   # named runs

Model (one per run, from data/sim/dataset_v2/runs/<name>_coef.json through experiments/pinn/physics.py):
  the Cummins model of sim/batch_sim_gpu.py linearised about the run's forward speed U (VesselModel.linearize: rigid
  body + A_inf + Coriolis/added-mass Coriolis + linear and ITTC damping + cross-flow drag, Jacobian at nu0 = [U 0 0 0 0
  0], eta = 0), restoring G, radiation memory xr_dot = Ar xr + Br nu, mu = Cr xr + Dr nu, and the kinematics eta_dot =
  J(eta) nu linearised about the same point (D_dot = w - U theta at U > 0). State z = [nu (6), D phi theta psi (4),
  xr (n_mem), u_int, z_int, 1] (N and E do not feed back and are dropped; the constant 1 carries the steady terms F0
  and J nu0; u_int, z_int are the autopilot integrators, see control). Discretisation: exact (matrix exponential of the
  augmented system) with a first-order hold on the input tau = tau_ctrl + tau_wave - tau_quad, step h = 0.05 s (the
  20 Hz rate of the run CSVs; 4 steps per 5 Hz sample). The propagation runs in float32 by default (--dtype; float64
  gives the same cv0 check numbers to 4 decimals at 2x the time).
  tau_quad: the quadratic roll/pitch/yaw damping kappa_4 Dref_4 |p| p, kappa_5 Dref_5 |q| q, Dv66 kappa_6 |r| r, put in
  as an input with a Heun predictor-corrector inside each step (--no-quadratic drops it). Reason: without it the
  equation residual on v2 is 3.2 % (median) / 46 % (95th pct) of the roll wave load, with it 1.7 % / 4.1 %
  (results/sim/residual_v2/physics_residual.md, levels linear_U and linear_Uq); on the cv0 check (20 runs) it cuts
  the force_20 roll error 0.030 -> 0.001 (normalised MSE). The explicit quadratic step is only stable at h = 0.05 s:
  its linearised stiffness h 2 qc |p|max Mi[3,3] reaches about 1.54 there on the most violent runs (Heun is stable up
  to 2, so ~25 % margin) and about 3.1 at h = 0.1, where otter_55kg_Hs0.40_Tp2.28_b090_U2_k5_r2 and
  otter_80kg_Hs0.40_Tp2.28_b090_U0_k5_r2 overflow to NaN in force_20 (review check). h > 0.05 is therefore refused
  unless --no-quadratic. Non-finite forecasts are never dropped: each (run, method) with any is counted (origins
  affected) in the json and the report, and its NaN stays in the npz and in every mean that includes it.

Signals: everything is taken from the 20 Hz CSV passed through the exact filter of the 5 Hz cache (sim_common._one:
scipy.signal.decimate q = 4, zero-phase Chebyshev I order 8 = sosfiltfilt), checked against the cache on every run.
The model is linear time-invariant apart from tau_quad, and a linear filter commutes with it, so the filtered state
driven by the filtered load follows the same equations; the truth of the protocol (the 5 Hz cache) is that filtered
state. Simulating the raw 20 Hz signals instead would leave the content the cache filter removes (up to 17 % of the
roll std in short head seas at U = 2 m/s) as an error floor of the oracle rows.

Forecast from origin o (sim_common.origins: the forecast sees x[:o] and predicts x[o:o+100]):
  initial state   eta, nu of the last known sample o - 1 (5 Hz cache columns D phi theta psi u v w p q r),
  memory state    rebuilt from the filtered nu of the past 60 s at 20 Hz, xr = 0 60 s before o - 1 (physics.py
                  memory_states discretisation, computed once per run and shifted: xr_win(t) = xr(t) - Phi^W xr(t - W)).
                  Strictly causal: the cubic drive of the step into t uses nu one row after t, so the last step into
                  the origin row j0 is redone with the first-order hold, xr(j0) = Phi xr(j0 - 1) + F0 nu(j0 - 1) +
                  F1 nu(j0) (nu up to j0 only; the steps before use nu up to j0 at most),
  control         --control autopilot (default): the simulator's own controllers closed inside the model (speed PI
                  Kp = Msys11/10, Ki = Kp/10, feed-forward tauX0; heading PID Kp = Msys66 wn^2, Kd = 2 Msys66 zeta wn,
                  Ki = Kp wn/10 with psi_ref = 0, so psi_d = r_d = 0; wn, zeta from gpu_inputs/vessel_*.mat). They are
                  known laws of the vessel, so this is causal; the integrator states at the origin are recovered by
                  inverting the laws on the logged tauX, tauN. Without them the linearised sway-yaw subsystem is
                  unstable at U = 2 m/s (eigenvalue +0.12 1/s) and drifts into roll: force_20 roll error at 20 s is
                  0.12 with 'mean' (tauX, tauN held at their past-60-s mean) and 0.025 with 'oracle' (true future
                  control, open loop) against 0.001 closed loop (cv0 check). 'last' holds the last logged value.
  wave load       free:     zero over the whole horizon (the vessel's own dynamics only),
                  force_k:  the true (filtered) tau_wave for lead 0 .. k s, zero afterwards (FOH ramps it to zero over
                            the next 0.05 s), k = 1, 2, 5, 20. force_20 = linearisation + discretisation cost. ORACLE rows,
                            they use the future wave load and are not predictors.
All 486 origins x 5 variants of a run are propagated together; runs go to --workers processes (one BLAS thread each).

Output (layout of results/sim/lstm_cv_metrics.npz, errors in m and degrees, run order = '<split>|ids' there):
  results/sim/diagnostics/physics_metrics<tag>.npz  '<split>|<method>|mse', '|mae' (R, 100, 3) per-run mean e^2, |e|
      per lead; '<split>|<method>|mse_o' (R, 486, 3) per-origin MSE over the 20 s; '<split>|ids';
      method in free, force_1, force_2, force_5, force_20
  physics_metrics<tag>.json   settings, per-run model facts (slowest memory pole, slowest model mode), runtime,
                              'nonfinite' {split: {run: {method: origins with a non-finite forecast}}}
  physics_diagnostics<tag>.md normalised tables (z-score by the split's train std, as experiments/phylstm/table_mse_mae.py)
                              with, on exactly the same test runs, the paper LSTM (mean of its 3 seeds,
                              results/sim/lstm_cv_metrics.npz), PhyLSTM w/o physics and PhyLSTM2
                              (results/sim/phylstm/phylstm_metrics.npz 'data', 'phy') and the past-60-s mean held
                              over the horizon ('hold mean'), plus the within-run variance / train variance per channel
                              and the non-finite count
<tag> is empty for a full run, '_r<N>' with --max-runs N, '_n<K>' added with --runs (K names), or --tag. A split
already in the output is reused only if its run ids are exactly the requested ones; otherwise it is recomputed.

Reading the plateau: the free response decays to the run's mean within ~1 s, so from then on its error is the run's
own variance around that mean, i.e. the plateau of the normalised MSE is (within-run variance of the scored runs) /
(train variance); the 'hold mean' row has the same plateau. Spread between runs is NOT the reason it can sit below 1:
on cv0 train the within-run share of the pooled variance is 0.9999 / 0.9999 / 0.995 (heave / roll / pitch). It is
below 1 when the scored runs move less than the train set does, e.g. the first 20 cv0 test runs (within-run variance
/ train variance 0.752 / 0.759 / 0.851, against 0.970 / 1.003 / 1.001 for all 518). Compare the physics rows with
learned models only on the same runs; the report prints those rows and the variance ratio per split.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per process (faster here, see --workers)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.linalg import expm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "pinn"))
import run_sim_lstm_cv as LCV  # noqa: E402
from physics import VesselModel, eta_dot  # noqa: E402
from sim_common import CSV_COLS, FS, HORIZON, MAX_CONTEXT, RESULTS_SIM, SIM, load, motion, origins  # noqa: E402

OUT = os.path.join(RESULTS_SIM, "diagnostics")
RAW_H = 0.05                       # 20 Hz CSV
Q = 4                              # 20 Hz -> 5 Hz (sim_common)
TRIM0 = int(30 * FS)               # first 5 Hz cache sample = raw row Q * TRIM0 (sim_common: 30 s start-up dropped)
SOS = signal.cheby1(8, 0.05, 0.8 / Q, output="sos")   # the filter of scipy.signal.decimate(q=4, ftype='iir')
KS = (1, 2, 5, 20)
METHODS = ["free"] + [f"force_{k}" for k in KS]
C = {c: i for i, c in enumerate(CSV_COLS)}
NU = [C[c] for c in ("u", "v", "w", "p", "q", "r")]
ETA4 = [C[c] for c in ("D", "phi", "theta", "psi")]
TW = [C[f"tau_w{i}"] for i in range(1, 7)]
KEEP = [2, 3, 4, 5]                # eta entries kept in the state: D phi theta psi
OUT_IDX = [6, 7, 8]                # state columns D, phi, theta


# ----------------------------------------------------------------------------- model
class LinearModel:
    """Linearised Cummins model of one run, exact FOH discretisation at step h.
    State z = [nu (6), D phi theta psi (4), xr (n_mem), u_int, z_int, 1]. With autopilot=True the simulator's speed PI
    and heading PID (sim/batch_sim_gpu.py, psi_ref = 0 so psi_d = r_d = 0) are closed inside A; u_int, z_int are their
    integrator states. Otherwise the control is an input and u_int, z_int are inert."""

    def __init__(self, coef_path, U, h, quadratic=True, autopilot=None):
        vm = VesselModel.from_json(coef_path)
        self.vm, self.U, self.h, self.quadratic = vm, float(U), h, quadratic
        nu0, F0, K = vm.linearize(float(U))
        e0 = torch.zeros(6, dtype=torch.float64)
        Jn = torch.autograd.functional.jacobian(lambda n: eta_dot(e0, n), nu0).numpy()
        Je = torch.autograd.functional.jacobian(lambda e: eta_dot(e, nu0), e0).numpy()
        ed0 = eta_dot(e0, nu0).numpy()
        assert np.allclose(Je[:, :2], 0), "N, E must not feed back"
        nu0, F0, K = nu0.numpy(), F0.numpy(), K.numpy()
        M, G = vm.M.numpy(), vm.G.numpy()
        Ar, Br, Cr, Dr = vm.Ar.numpy(), vm.Br.numpy(), vm.Cr.numpy(), vm.Dr.numpy()
        n = vm.n_mem
        self.n_mem, self.nz = n, 13 + n
        Mi = np.linalg.inv(M)
        A = np.zeros((self.nz, self.nz))
        iv, ie, ix = slice(0, 6), slice(6, 10), slice(10, 10 + n)
        iu, iz, ic = 10 + n, 11 + n, 12 + n
        self.iu, self.iz = iu, iz
        A[iv, iv] = -Mi @ (K + Dr)
        A[iv, ie] = -Mi @ G[:, KEEP]
        A[iv, ix] = -Mi @ Cr
        A[iv, ic] = Mi @ (K @ nu0 - F0)
        A[ie, iv] = Jn[KEEP]
        A[ie, ie] = Je[np.ix_(KEEP, KEEP)]
        A[ie, ic] = (ed0 - Jn @ nu0)[KEEP]
        A[ix, iv] = Br
        A[ix, ix] = Ar
        self.ap = None
        if autopilot is not None:
            # sim/batch_sim_gpu.py: Msys = MRB + MA_inf; tauX = tauX0 + Kp_u (U - u) + Ki_u u_int, u_int' = U - u;
            # tauN = -Kp psi - Kd r - Ki z_int, z_int' = psi; tauX0 balances the surge equation at nu0, xr = 0
            Msys = np.asarray(vm.coef["MRB"], float) + np.asarray(vm.coef["MA_inf"], float)
            Kpu = Msys[0, 0] / 10
            Kiu = Kpu / 10
            wn, zeta = autopilot
            Kp, Kd = Msys[5, 5] * wn ** 2, Msys[5, 5] * 2 * zeta * wn
            Ki = Kp * wn / 10
            tauX0 = (Msys @ Mi @ (F0 + Dr @ nu0))[0]
            A[iv, 0] += -Kpu * Mi[:, 0]
            A[iv, iu] += Kiu * Mi[:, 0]
            A[iv, ic] += (tauX0 + Kpu * self.U) * Mi[:, 0]
            A[iv, 9] += -Kp * Mi[:, 5]
            A[iv, 5] += -Kd * Mi[:, 5]
            A[iv, iz] += -Ki * Mi[:, 5]
            A[iu, 0], A[iu, ic], A[iz, 9] = -1.0, self.U, 1.0
            self.ap = dict(Kpu=Kpu, Kiu=Kiu, Kp=Kp, Kd=Kd, Ki=Ki, tauX0=tauX0)
        B = np.zeros((self.nz, 6))
        B[iv] = Mi
        self.A, self.B = A, B
        # FOH: z[k+1] = Phi z[k] + G0 u[k] + G1 u[k+1]
        m = 6
        Z = np.zeros((self.nz + 2 * m, self.nz + 2 * m))
        Z[:self.nz, :self.nz] = A * h
        Z[:self.nz, self.nz:self.nz + m] = B * h
        Z[self.nz:self.nz + m, self.nz + m:] = np.eye(m)
        E = expm(Z)
        Phi, P0, P1 = E[:self.nz, :self.nz], E[:self.nz, self.nz:self.nz + m], E[:self.nz, self.nz + m:]
        self.PhiT, self.G0T, self.G1T = Phi.T.copy(), (P0 - P1).T.copy(), P1.T.copy()
        k4, k5, k6 = vm.kappa
        self.qc = np.array([k4 * float(vm.Dref[3]), k5 * float(vm.Dref[4]), float(vm.Dv[5, 5]) * k6])
        # memory recursion of physics.py at the 20 Hz rate (cubic interpolation of nu between samples)
        Pm, (F0m, F1m), cub = vm._discretize(RAW_H)
        self.mem = (Pm.numpy(), F0m.numpy(), F1m.numpy(), [c.numpy() for c in cub])
        self.PhiW = np.linalg.matrix_power(self.mem[0], int(round(MAX_CONTEXT / FS / RAW_H)))
        ev_mem = np.linalg.eigvals(Ar)
        ev = np.linalg.eigvals(A)
        ev = ev[np.abs(ev) > 1e-8]                       # drop the exact integrators (constant state, open-loop psi ...)
        self.facts = dict(U=self.U, n_mem=n, slowest_memory_pole=float(ev_mem.real.max()),
                          slowest_memory_tau_s=float(-1 / ev_mem.real.max()),
                          slowest_mode=float(ev.real.max()), slowest_mode_tau_s=float(-1 / ev.real.max()),
                          slowest_mode_im=float(ev[np.argmax(ev.real)].imag), kappa=list(map(float, vm.kappa)))

    def integrators(self, u, psi, r, tauX, tauN):
        """Autopilot integrator states that reproduce the logged control (inverts the PI / PID laws)."""
        a = self.ap
        ui = (tauX - a["tauX0"] - a["Kpu"] * (self.U - u)) / a["Kiu"]
        zi = -(tauN + a["Kp"] * psi + a["Kd"] * r) / a["Ki"]
        return ui, zi

    def tau_quad(self, nu):
        out = np.zeros((nu.shape[0], 6), nu.dtype)
        if self.quadratic:
            pqr = nu[:, 3:6]
            out[:, 3:6] = self.qc.astype(nu.dtype) * np.abs(pqr) * pqr
        return out

    def memory_pass(self, nu):
        """Memory states (T, n) of the whole run driven by nu (T, 6) at 20 Hz from xr = 0 (physics.memory_states)."""
        Phi, F0, F1, (Cm, C0, C1, C2) = self.mem
        T = nu.shape[0]
        drive = nu[:-1] @ F0.T + nu[1:] @ F1.T
        drive[1:T - 2] = nu[:T - 3] @ Cm.T + nu[1:T - 2] @ C0.T + nu[2:T - 1] @ C1.T + nu[3:] @ C2.T
        PhiT = Phi.T.copy()
        out = np.zeros((T, self.n_mem))
        x = np.zeros(self.n_mem)
        for k in range(T - 1):
            x = x @ PhiT + drive[k]
            out[k + 1] = x
        return out

    def simulate(self, z0, ctrl, wave, ksteps, n_steps, sub, dtype=np.float64):
        """z0 (O, nz); ctrl (O, 6) held or (O, n_steps + 1, 6); wave (O, n_steps + 1, 6) at the step instants;
        ksteps: per variant, the last step index with wave load (-1 = none). Returns D, phi, theta every `sub` steps,
        (V, O, n_steps // sub, 3)."""
        O, V = z0.shape[0], len(ksteps)
        PhiT, G0T, G1T = (a.astype(dtype) for a in (self.PhiT, self.G0T, self.G1T))
        G1Tn = np.ascontiguousarray(G1T[:, :6])
        Z = np.tile(z0, (V, 1)).astype(dtype)
        ctrl, wave = ctrl.astype(dtype), wave.astype(dtype)
        out = np.empty((V, O, n_steps // sub, 3))
        uj = np.empty((V * O, 6), dtype)

        def u(j):
            c = ctrl[:, j] if ctrl.ndim == 3 else ctrl
            for v, ks in enumerate(ksteps):
                uj[v * O:(v + 1) * O] = c + wave[:, j] if j <= ks else c
            return uj.copy()

        uk = u(0)
        qk = self.tau_quad(Z[:, :6])
        for j in range(n_steps):
            uk1 = u(j + 1)
            base = Z @ PhiT + (uk - qk) @ G0T
            if self.quadratic:
                nu_p = base[:, :6] + (uk1 - qk) @ G1Tn                  # predictor (quadratic force held)
                qp = self.tau_quad(nu_p)
                Z = base + (uk1 - qp) @ G1T                              # corrector (FOH of the quadratic force)
                qk = self.tau_quad(Z[:, :6])
            else:
                Z = base + uk1 @ G1T
            uk = uk1
            if (j + 1) % sub == 0:
                out[:, :, (j + 1) // sub - 1] = Z[:, OUT_IDX].reshape(V, O, 3)
        return out


_AP = {}


def autopilot_gains(vessel, dz3, dz5):
    """(wn, zeta) of the heading autopilot from the vessel constants the GPU simulator loaded."""
    key = (vessel, dz3, dz5)
    if key not in _AP:
        S = sio.loadmat(os.path.join(SIM, "gpu_inputs", f"vessel_{vessel}_dz{dz3:g}_{dz5:g}.mat"))
        _AP[key] = (float(S["hc_wn"].squeeze()), float(S["hc_zeta"].squeeze()))
    return _AP[key]


# ----------------------------------------------------------------------------- one run
def run_one(task):
    """task = (name, vessel, dzeta3, dzeta5, xc (T, 21) 5 Hz cache rows, settings dict).
    Returns per method (mse, mae, mse_o), the model facts and {method: origins with a non-finite forecast}."""
    name, vessel, dz3, dz5, xc, opt = task
    raw = pd.read_csv(os.path.join(SIM, "runs", name + ".csv"), dtype=np.float64).to_numpy()[:, 1:]
    filt = signal.sosfiltfilt(SOS, raw, axis=0)
    T = xc.shape[0]
    sub5 = filt[::Q][TRIM0:TRIM0 + T]
    chk = [C[c] for c in ("D", "phi", "theta", "w", "q", "tau_w3")]
    dev = np.abs(sub5[:, chk] - xc[:, chk]).max(0) / (np.abs(xc[:, chk]).max(0) + 1e-12)
    assert dev.max() < 1e-4, f"{name}: filtered CSV does not match the 5 Hz cache ({dev})"

    coef = os.path.join(SIM, "runs", name + "_coef.json")
    with open(coef) as f:
        U = json.load(f)["run"]["U_ref"]
    ap = autopilot_gains(vessel, dz3, dz5) if opt["control"] == "autopilot" else None
    lin = LinearModel(coef, U, opt["h"], quadratic=opt["quadratic"], autopilot=ap)
    step = int(round(opt["h"] / RAW_H))                    # raw rows per simulation step
    sub = int(round(1 / FS / opt["h"]))                    # simulation steps per 5 Hz sample
    n_steps = HORIZON * sub
    W = int(round(MAX_CONTEXT / FS / RAW_H))               # 60 s at 20 Hz

    o = origins(T)
    O = len(o)
    j0 = Q * (o - 1 + TRIM0)                               # raw row of the last known sample
    nu20 = filt[:, NU]
    xr_full = lin.memory_pass(nu20)
    Pm, F0m, F1m, _ = lin.mem
    # strictly causal: xr_full[j0] used the cubic drive with nu[j0 + 1]; redo the last step with the FOH (nu[j0 - 1],
    # nu[j0]). xr_full[j0 - 1] used nu up to j0. The window start subtracts Phi^W xr(j0 - W) (linearity).
    xr_last = xr_full[j0 - 1] @ Pm.T + nu20[j0 - 1] @ F0m.T + nu20[j0] @ F1m.T
    xr0 = xr_last - xr_full[j0 - W] @ lin.PhiW.T
    last = xc[o - 1]                                       # = filt[j0] (checked above)
    integ = np.zeros((O, 2))
    tc = filt[:, [C["tauX"], C["tauN"]]]
    if ap is not None:
        integ[:, 0], integ[:, 1] = lin.integrators(last[:, C["u"]], last[:, C["psi"]], last[:, C["r"]],
                                                   tc[j0, 0], tc[j0, 1])
    z0 = np.concatenate([last[:, NU], last[:, ETA4], xr0, integ, np.ones((O, 1))], axis=1).astype(np.float64)

    idx = j0[:, None] + step * np.arange(n_steps + 1)
    wave = filt[idx][..., TW]                                                       # (O, n_steps + 1, 6)
    if opt["control"] == "oracle":
        c2 = tc[idx]                                                                # (O, n_steps + 1, 2)
    elif opt["control"] == "mean":
        cs = np.vstack([np.zeros((1, 2)), np.cumsum(tc, axis=0)])
        c2 = (cs[j0 + 1] - cs[j0 + 1 - W]) / W
    elif opt["control"] == "last":
        c2 = tc[j0]
    else:                                                                           # autopilot: inside the model
        c2 = np.zeros((O, 2))
    ctrl = np.zeros(c2.shape[:-1] + (6,))
    ctrl[..., 0], ctrl[..., 5] = c2[..., 0], c2[..., 1]

    ksteps = [-1] + [int(round(k / opt["h"])) for k in KS]
    pred = lin.simulate(z0, ctrl, wave, ksteps, n_steps, sub, dtype=np.dtype(opt["dtype"]))   # (V, O, 100, 3)
    # non-finite forecasts are counted (origins affected, per method) and kept as NaN/inf in every metric: nothing is
    # dropped. main() records the counts in the json, the report prints them.
    bad = ~np.isfinite(pred).all(axis=(2, 3))                                       # (V, O)
    nonfinite = {m: int(bad[v].sum()) for v, m in enumerate(METHODS) if bad[v].any()}
    if nonfinite:
        print(f"    NON-FINITE forecast {name}: {nonfinite} origins of {O}", flush=True)
    pred = np.stack([pred[..., 0], np.degrees(pred[..., 1]), np.degrees(pred[..., 2])], -1)
    Y = motion(xc).astype(np.float64)
    truth = Y[o[:, None] + np.arange(HORIZON)]                                      # (O, 100, 3)
    e = pred - truth
    res = {m: ((e[v] ** 2).mean(0), np.abs(e[v]).mean(0), (e[v] ** 2).mean(1)) for v, m in enumerate(METHODS)}
    return res, lin.facts, nonfinite


# ----------------------------------------------------------------------------- report
PHY_METRICS = os.path.join(RESULTS_SIM, "phylstm", "phylstm_metrics.npz")
LEARNED = [("lstm", "LSTM [Wang 2026], mean of 3 seeds"), ("data", "PhyLSTM w/o physics"), ("phy", "PhyLSTM2")]


def train_stats(meta, Y, s):
    """Pooled train std per channel (the z-score of table_mse_mae.py) and the within-run share of the pooled variance."""
    Ytr = np.asarray(Y[LCV.split_rows(meta, s)["train"]], np.float64)
    var = Ytr.var(axis=(0, 1))
    return np.sqrt(var), Ytr.var(axis=1).mean(0) / var


def within_ratio(Y, rows, var):
    """Mean within-run variance of the runs `rows` divided by the train variance, per channel."""
    return np.mean([np.asarray(Y[r], np.float64).var(0) for r in rows], axis=0) / var


def hold_mean_ref(Y, rows):
    """Reference forecast: the mean of the past 60 s held over the horizon. Per-lead mean e^2 and |e| (R, 100, 3)."""
    mse, mae = [], []
    for r in rows:
        y = Y[r].astype(np.float64)
        o = origins(len(y))
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(y, 0)])
        mu = (cs[o] - cs[o - MAX_CONTEXT]) / MAX_CONTEXT
        e = y[o[:, None] + np.arange(HORIZON)] - mu[:, None]
        mse.append((e ** 2).mean(0))
        mae.append(np.abs(e).mean(0))
    return np.array(mse), np.array(mae)


def learned_rows(s, ids):
    """{label: (mse, mae)} (R, 100, 3) of the learned models on exactly the runs `ids` of split s (looked up by id),
    and notes for models that are missing or do not cover these runs."""
    out, notes = {}, []
    srcs = {"lstm": LCV.METRICS, "data": PHY_METRICS, "phy": PHY_METRICS}
    for key, label in LEARNED:
        path = srcs[key]
        if not os.path.exists(path):
            notes.append(f"{label}: {path} not found")
            continue
        z = np.load(path)
        if f"{s}|ids" not in z.files or (key != "lstm" and f"{s}|{key}|mse" not in z.files):
            notes.append(f"{label}: split {s} not in {os.path.basename(path)}")
            continue
        where = {int(i): k for k, i in enumerate(z[f"{s}|ids"])}
        if not all(int(i) in where for i in ids):
            notes.append(f"{label}: not all scored runs are in {os.path.basename(path)}")
            continue
        pos = np.array([where[int(i)] for i in ids])
        if key == "lstm":
            mse = np.mean([z[f"{s}|seed{k}|mse"][pos] for k in LCV.SEEDS], axis=0)
            mae = np.mean([z[f"{s}|seed{k}|mae"][pos] for k in LCV.SEEDS], axis=0)
        else:
            mse, mae = z[f"{s}|{key}|mse"][pos], z[f"{s}|{key}|mae"][pos]
        out[label] = (mse.astype(np.float64), mae.astype(np.float64))
    return out, notes


def trip(a, f="{:.3f}"):
    return "/".join(f.format(v) for v in a)


def report(res, meta, Y, tag, cfg):
    lines = ["# Physics diagnostics: free response and oracle wave-load rows of the linearised Cummins model", "",
             f"Generated by `experiments/diagnostics/free_response.py` on {time.strftime('%Y-%m-%d %H:%M')}. Raw numbers: "
             f"`physics_metrics{tag}.npz`, settings and per-run model facts: `physics_metrics{tag}.json`.", "",
             f"Settings: control `{cfg['control']}`, quadratic roll/pitch/yaw damping {'on' if cfg['quadratic'] else 'off'}, "
             f"step h = {cfg['h']} s, memory rebuilt from the past {MAX_CONTEXT / FS:.0f} s. force_k rows use the TRUE "
             "future wave load for k s (oracle, not predictors). Normalised = divided by the split's train std per "
             "channel (z-score as `experiments/phylstm/table_mse_mae.py`, but the std accumulated in float64: the "
             "float32 std there is 0.4-0.5 % smaller on cv0, so its normalised MSE is ~0.9 % larger); 'hold mean' = the "
             "past-60-s mean held over the horizon (reference). The learned-model rows (LSTM [Wang 2026], mean of its 3 seeds, from "
             "`results/sim/lstm_cv_metrics.npz`; PhyLSTM w/o physics and PhyLSTM2 from "
             "`results/sim/phylstm/phylstm_metrics.npz`) are restricted to exactly the runs scored here, looked up by id.",
             "",
             "Reading the plateau: the free response decays to the run mean within about 1 s, so its plateau is the "
             "scored runs' within-run variance divided by the train variance (the 'hold mean' row has the same "
             "plateau). It is below 1 when the scored runs move less than the train set does, not because of spread "
             "between runs (the within-run share of the train variance, printed per split, is close to 1). Compare "
             "rows only within one table (same runs); do not compare them with full-split tables.", ""]
    splits = [s for s in LCV.SPLITS if f"{s}|ids" in res]
    leads = [0.2, 1, 2, 5, 10, 20]
    id2row = {int(i): k for k, i in enumerate(meta.id.to_numpy())}
    for s in splits:
        ids = res[f"{s}|ids"]
        rows = np.array([id2row[int(i)] for i in ids])
        sd, share = train_stats(meta, Y, s)
        hm2, hm1 = hold_mean_ref(Y, rows)
        learned, notes = learned_rows(s, ids)
        curves = {m: res[f"{s}|{m}|mse"].astype(np.float64).mean(0) / sd ** 2 for m in METHODS}
        maes = {m: res[f"{s}|{m}|mae"].astype(np.float64).mean(0) / sd for m in METHODS}
        curves["hold mean"], maes["hold mean"] = hm2.mean(0) / sd ** 2, hm1.mean(0) / sd
        for lab, (m2, m1) in learned.items():
            curves[lab], maes[lab] = m2.mean(0) / sd ** 2, m1.mean(0) / sd
        names = METHODS + ["hold mean"] + list(learned)
        all_test = LCV.split_rows(meta, s)["test"]
        r_sc = within_ratio(Y, rows, sd ** 2)
        lines += [f"## {s}: {len(ids)} of {len(all_test)} test runs", "",
                  "Variance of the motion (heave / roll / pitch):", "", "| | heave / roll / pitch |", "|---|---|",
                  f"| within-run share of the pooled train variance (train runs) | {trip(share, '{:.4f}')} |",
                  f"| within-run variance / train variance, the {len(ids)} scored test runs | {trip(r_sc)} |"]
        if len(ids) < len(all_test):
            lines.append(f"| within-run variance / train variance, all {len(all_test)} test runs of {s} | "
                         f"{trip(within_ratio(Y, all_test, sd ** 2))} |")
        nf = cfg["nonfinite"].get(s) if "nonfinite" in cfg else None
        if nf is None:
            lines += ["", "Non-finite forecasts: not recorded (output made before the count was added)."]
        elif not nf:
            lines += ["", f"Non-finite forecasts: none (all {len(ids)} runs x {len(METHODS)} methods finite)."]
        else:
            lines += ["", f"**Non-finite forecasts in {len(nf)} of {len(ids)} runs** (origins affected, of "
                      f"{res[f'{s}|free|mse_o'].shape[1]}). Nothing is excluded: every mean that includes them is "
                      "nan/inf.", "",
                      "| run | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
            for run, d in nf.items():
                lines.append(f"| {run} | " + " | ".join(str(d.get(m, 0)) for m in METHODS) + " |")
        if notes:
            lines += [""] + [f"Not shown: {n}." for n in notes]
        lines += ["", "Normalised MSE / MAE averaged over leads 0.2 .. H s and over heave, roll, pitch (same runs in "
                  "every row):", "",
                  "| method | 2 s MSE | 2 s MAE | 5 s MSE | 5 s MAE | 10 s MSE | 10 s MAE | 20 s MSE | 20 s MAE |",
                  "|---|" + "---|" * 8]
        for m in names:
            cells = []
            for H in (2, 5, 10, 20):
                n = int(H * FS)
                cells += [f"{curves[m][:n].mean():.4f}", f"{maes[m][:n].mean():.4f}"]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["", "Normalised MSE at single leads (heave / roll / pitch):", "",
                  "| method | " + " | ".join(f"{t:g} s" for t in leads) + " |", "|---|" + "---|" * len(leads)]
        for m in names:
            c = curves[m]
            lines.append(f"| {m} | " + " | ".join(trip(c[int(round(t * FS)) - 1]) for t in leads) + " |")
        c = curves["free"]
        plateau = c[int(15 * FS):].mean(0)
        dec = [next(((i + 1) / FS for i in range(HORIZON) if c[i, ch] >= 0.9 * plateau[ch]), float("nan"))
               for ch in range(3)]
        lines += ["", "Free response: decay time = first lead at which the normalised MSE reaches 90 % of its plateau "
                  "(mean over leads 15-20 s). The plateau is to be read against the scored runs' within-run variance "
                  "/ train variance:", "", "| | heave | roll | pitch |", "|---|---|---|---|",
                  "| plateau | " + " | ".join(f"{v:.3f}" for v in plateau) + " |",
                  "| within-run var / train var (scored runs) | " + " | ".join(f"{v:.3f}" for v in r_sc) + " |",
                  "| decay time (s) | " + " | ".join(f"{v:.1f}" for v in dec) + " |", ""]
        U = meta.set_index("id").loc[ids, "U"].to_numpy(float)
        cols = [("free", res[f"{s}|free|mse"]), ("force_20", res[f"{s}|force_20|mse"]), ("hold mean", hm2)] + \
            [(lab, m2) for lab, (m2, _) in learned.items()]
        lines += ["By forward speed: normalised MSE averaged over leads 0.2 .. 20 s (heave / roll / pitch):", "",
                  "| U (m/s) | runs | " + " | ".join(n for n, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
        for u in np.unique(U):
            k = U == u
            lines.append(f"| {u:g} | {k.sum()} | " + " | ".join(
                trip(np.asarray(a, np.float64)[k].mean((0, 1)) / sd ** 2, "{:.4f}") for _, a in cols) + " |")
        # the linearisation about U breaks down where the surge speed swings (cache meta.csv 'regime' = violent,
        # e.g. otter_55kg_Hs0.50_Tp2.55_b000_U2_k5_r1, max_du 2.5 m/s: force_20 heave 15 % of its variance at 0.2 s)
        reg = meta.set_index("id").loc[ids, "regime"].to_numpy()
        lines += ["", "By regime (column `regime` of the cache meta.csv): normalised MSE averaged over leads 0.2 .. 20 s "
                  "(heave / roll / pitch):", "",
                  "| regime | runs | " + " | ".join(n for n, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
        for g in np.unique(reg):
            k = reg == g
            lines.append(f"| {g} | {k.sum()} | " + " | ".join(
                trip(np.asarray(a, np.float64)[k].mean((0, 1)) / sd ** 2, "{:.4f}") for _, a in cols) + " |")
        rt = cfg.get("runtime_s", {}).get(s)
        if rt:
            lines += ["", f"Runtime: {rt['seconds']:.0f} s for {rt['runs']} runs with {rt.get('workers', 1)} worker(s)."]
        lines.append("")
    facts = pd.DataFrame(cfg.get("facts", {})).T
    if len(facts):
        facts["vessel"] = [n.split("_Hs")[0] for n in facts.index]
        g = facts.groupby("vessel").agg(slowest_memory_tau_s=("slowest_memory_tau_s", "max"),
                                        slowest_memory_pole=("slowest_memory_pole", "max"),
                                        slowest_mode_tau_s=("slowest_mode_tau_s", "max"), runs=("U", "size"))
        lines += ["## Model facts (runs scored so far)", "",
                  "Slowest radiation-memory pole (max Re eig Ar) and slowest mode of the whole linearised model (psi and "
                  "the constant state excluded), per vessel, worst run:", "",
                  "| vessel | runs | slowest memory pole (1/s) | its time constant (s) | slowest model mode time constant (s) |",
                  "|---|---|---|---|---|"]
        for v, r in g.iterrows():
            lines.append(f"| {v} | {r.runs} | {r.slowest_memory_pole:.4f} | {r.slowest_memory_tau_s:.1f} | {r.slowest_mode_tau_s:.1f} |")
        lines.append("")
    path = os.path.join(OUT, f"physics_diagnostics{tag}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"report -> {path}")


# ----------------------------------------------------------------------------- main
def select_rows(meta, s, max_runs, names):
    """Test rows of split s to score, in the split's order: the first max_runs (all if 0 and no names) plus the named
    runs that are in its test set."""
    test = LCV.split_rows(meta, s)["test"]
    if not max_runs and not names:
        return test
    keep = np.zeros(len(test), bool)
    keep[:max_runs] = True
    if names:
        keep |= np.isin(meta.name.to_numpy()[test], list(names))
    return test[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=LCV.SPLITS, choices=LCV.SPLITS)
    ap.add_argument("--max-runs", type=int, default=0, help="score only the first N test runs of each split")
    ap.add_argument("--runs", nargs="+", default=None, metavar="NAME",
                    help="also score these named runs (data/sim/dataset_v2 names) in every requested split whose test "
                         "set has them; alone (no --max-runs) only these")
    ap.add_argument("--control", default="autopilot", choices=["autopilot", "mean", "last", "oracle"])
    ap.add_argument("--no-quadratic", action="store_true")
    ap.add_argument("--h", type=float, default=0.05, choices=[0.05, 0.1, 0.2],
                    help="simulation step (s); > 0.05 only with --no-quadratic")
    ap.add_argument("--dtype", default="float32", choices=["float64", "float32"], help="of the forecast propagation")
    ap.add_argument("--workers", type=int, default=1, help="processes, one BLAS thread each (~0.4 GB RAM each)")
    ap.add_argument("--tag", default=None, help="output suffix (default '', '_r<N>' with --max-runs, '_n<K>' added "
                                                "with --runs)")
    ap.add_argument("--fresh", action="store_true", help="recompute splits already in the output file")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.h > 0.05 and not args.no_quadratic:
        ap.error(f"--h {args.h:g} with the quadratic damping: the explicit (Heun) quadratic step is stable only at "
                 "h = 0.05 (stiffness h 2 qc |p| Mi44 up to ~1.5 there, ~3.1 at 0.1; NaN on the Hs 0.40 Tp 2.28 "
                 "b090 k5 runs at 0.1); use --h 0.05 or --no-quadratic")
    if args.tag is not None:
        tag = args.tag
    else:
        tag = (f"_r{args.max_runs}" if args.max_runs else "") + (f"_n{len(args.runs)}" if args.runs else "")
    tag = tag if not tag or tag.startswith("_") else "_" + tag
    os.makedirs(OUT, exist_ok=True)
    npz, cfgp = os.path.join(OUT, f"physics_metrics{tag}.npz"), os.path.join(OUT, f"physics_metrics{tag}.json")
    opt = dict(control=args.control, quadratic=not args.no_quadratic, h=args.h, dtype=args.dtype)
    cfg = dict(**opt, memory_window_s=MAX_CONTEXT / FS, methods=METHODS, facts={}, runtime_s={}, nonfinite={})
    res = {}
    if os.path.exists(npz) and not args.fresh:
        res = dict(np.load(npz))
        with open(cfgp) as f:
            old = json.load(f)
        for k in opt:
            assert old[k] == cfg[k], f"{npz} was made with {k} = {old[k]}; use --tag or --fresh"
        cfg["facts"], cfg["runtime_s"] = old.get("facts", {}), old.get("runtime_s", {})
        cfg["nonfinite"] = old.get("nonfinite", {})        # splits made before the count: reported as not recorded
    torch.set_num_threads(1)
    meta, X = load()
    if not args.report_only:
        lstm = np.load(LCV.METRICS)
        sel = {s: select_rows(meta, s, args.max_runs, args.runs) for s in args.splits}
        if args.runs:
            found = set(meta.name.to_numpy()[np.concatenate(list(sel.values()))])
            miss = [n for n in args.runs if n not in found]
            if miss:
                ap.error(f"--runs not in the test set of any of {args.splits}: {miss}")
        pool = Pool(args.workers) if args.workers > 1 else None
        for s in args.splits:
            rows = sel[s]
            if not len(rows):
                print(f"{s}: none of the requested runs is a test run here, skipped", flush=True)
                continue
            ids = meta.id.to_numpy()[rows]
            ref = lstm[f"{s}|ids"]
            if args.max_runs or args.runs:
                assert np.isin(ids, ref).all(), f"{s}: runs missing from {LCV.METRICS}"
                assert not args.max_runs or np.array_equal(ids[:args.max_runs], ref[:args.max_runs]), \
                    f"{s}: run order differs from {LCV.METRICS}"
            else:
                assert np.array_equal(ids, ref), f"{s}: test runs differ from {LCV.METRICS}"
            if not args.fresh and f"{s}|ids" in res and all(f"{s}|{m}|mse" in res for m in METHODS):
                if np.array_equal(res[f"{s}|ids"], ids):
                    print(f"{s}: already in {npz} (same {len(ids)} runs)", flush=True)
                    continue
                print(f"{s}: {npz} has {len(res[f'{s}|ids'])} other runs for this split, recomputing "
                      f"{len(ids)}", flush=True)
            print(f"{s}: {len(rows)} test runs, {args.workers} worker(s)", flush=True)
            R = len(rows)
            out = {m: [np.empty((R, HORIZON, 3), np.float32), np.empty((R, HORIZON, 3), np.float32), None]
                   for m in METHODS}
            tasks = ((meta.name[r], meta.vessel[r], float(meta.dzeta3[r]), float(meta.dzeta5[r]),
                      np.asarray(X[r], np.float64), opt) for r in rows)
            t0 = time.time()
            nonfinite = {}
            it = pool.imap(run_one, tasks, chunksize=2) if pool else map(run_one, tasks)
            for i, (r_res, facts, bad) in enumerate(it):
                cfg["facts"][meta.name[rows[i]]] = facts
                if bad:
                    nonfinite[meta.name[rows[i]]] = bad
                for m, (mse, mae, mse_o) in r_res.items():
                    out[m][0][i], out[m][1][i] = mse, mae
                    if out[m][2] is None:
                        out[m][2] = np.empty((R,) + mse_o.shape, np.float32)
                    out[m][2][i] = mse_o
                if (i + 1) % 10 == 0 or i + 1 == R:
                    el = time.time() - t0
                    print(f"  {i + 1}/{R} runs, {el:.0f} s, {el / (i + 1):.2f} s/run, "
                          f"eta {el / (i + 1) * (R - i - 1):.0f} s", flush=True)
            res[f"{s}|ids"] = ids
            for m in METHODS:
                res[f"{s}|{m}|mse"], res[f"{s}|{m}|mae"], res[f"{s}|{m}|mse_o"] = out[m]
            cfg["nonfinite"][s] = nonfinite
            cfg["runtime_s"][s] = dict(seconds=round(time.time() - t0, 1), runs=R, workers=args.workers,
                                       max_runs=args.max_runs, named=args.runs or [])
            np.savez_compressed(npz, **res)
            with open(cfgp, "w") as f:
                json.dump(cfg, f, indent=1)
            print(f"  {s} done in {time.time() - t0:.0f} s -> {npz}"
                  + (f"  NON-FINITE forecasts in {len(nonfinite)} runs" if nonfinite else ""), flush=True)
        if pool:
            pool.close()
            pool.join()
    report(res, meta, motion(X), tag, cfg)


if __name__ == "__main__":
    main()
