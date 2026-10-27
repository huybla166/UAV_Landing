"""Equations of motion of the simulated USV (data/sim/dataset_v1) as a differentiable residual (step 4.1).

The model is the one integrated by sim/hydroVesselMemFast.m (Cummins fluid memory); its coefficients
are exported per run to <run>_coef.json by sim/export_coef.m:

  M nu_dot + C_RB(nu) nu + C_A(nu) nu + D(nu) nu + mu + G eta - tau_cf(nu) = tau_ctrl + tau_wave
  mu = Cr xr + Dr nu,   xr_dot = Ar xr + Br nu          (radiation memory, xr(0) = 0 in every run)
  eta_dot = J(eta) nu

  M       MRB + A_inf
  C_RB    rigid-body Coriolis about the CG; C_A added-mass Coriolis of A_inf
  D(nu)   Dv nu with ITTC surge resistance, kappa_k Dref_k |nu_k| nu_k (roll, pitch), Dv66 kappa_6 |r| r
  tau_cf  Hoerner cross-flow drag, 20 strips
  eta = [N E D phi theta psi], nu = [u v w p q r]; D is heave, positive down.

Physics levels for the PINN loss (only 'full' is exact on the simulated trajectories):
  full       the simulator's equations
  linear     the same equations linearized about steady forward speed U (memory kept), optionally plus
             the quadratic roll/pitch/yaw damping (quadratic=True); U = 0 gives the zero-speed model that
             Bao_cao_tien_do_Giai_doan_0-1.md section 5 showed fails for pitch
  constant   MSS power-based constant matrices at w0, no memory: M_c nu_dot + D_c nu + G eta
             (the per-DOF equations of Thiet_ke_song_USV_PINN.md section 1.7)

Tensors are (..., T, 6) with time on dim -2; derivatives are central differences, so residuals are
returned on the interior samples [p, T - p) with p = order // 2 (see `interior`).
"""
import json
import math

import numpy as np
import torch

CSV_COLS = ["t", "zeta", "tau_w1", "tau_w2", "tau_w3", "tau_w4", "tau_w5", "tau_w6",
            "N", "E", "D", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r", "tauX", "tauN"]
DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
RHO_CROSSFLOW = 1025.0   # mssConstants().rho_water, used by crossFlowDrag / prepare_fast.m
N_STRIPS = 20
ORDER = 6                # central-difference order: the 5-point stencil leaves up to 1.4 % (roll) on the eta-only path

# MSS Hoerner.m: 2-D cross-flow drag coefficient versus B / (2 T)
_HOERNER = np.array([
    [0.0108623, 1.96608], [0.176606, 1.96573], [0.353025, 1.89756], [0.451863, 1.78718],
    [0.472838, 1.58374], [0.492877, 1.27862], [0.493252, 1.21082], [0.558473, 1.08356],
    [0.646401, 0.998631], [0.833589, 0.87959], [0.988002, 0.828415], [1.30807, 0.759941],
    [1.63918, 0.691442], [1.85998, 0.657076], [2.31288, 0.630693], [2.59998, 0.596186],
    [3.00877, 0.586846], [3.45075, 0.585909], [3.7379, 0.559877], [4.00309, 0.559315]])


def hoerner(B, T):
    ratio = np.clip(B / (2 * T), _HOERNER[0, 0], _HOERNER[-1, 0])
    return float(np.interp(ratio, _HOERNER[:, 0], _HOERNER[:, 1]))


# ----------------------------------------------------------------------------- data
def load_run(csv_path, dtype=torch.float64, device="cpu"):
    """One simulated run as tensors: t (T,), zeta (T,), tau_wave, eta, nu, tau_ctrl (T, 6), and
    tau_ext = held(tau_ctrl) + tau_wave, the load to put on the right-hand side of the residual."""
    x = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    zero = torch.zeros_like(x[:, 0])
    tau_ctrl = torch.stack([x[:, 20], zero, zero, zero, zero, x[:, 21]], -1)
    return {
        "t": x[:, 0], "zeta": x[:, 1], "tau_wave": x[:, 2:8], "eta": x[:, 8:14], "nu": x[:, 14:20],
        "tau_ctrl": tau_ctrl, "tau_ext": held(tau_ctrl) + x[:, 2:8],
        "h": float(x[1, 0] - x[0, 0]),
    }


def held(u):
    """A zero-order-hold input (the simulator keeps tau_ctrl[k] over [t_k, t_k+1)) as seen by a central
    difference at t_k: the mean of the values on the two neighbouring steps. The wave load needs no
    correction, it is evaluated at t, t + h/2, t + h inside each RK4 step (sim/rk4u.m)."""
    out = u.clone()
    out[..., 1:, :] = 0.5 * (u[..., 1:, :] + u[..., :-1, :])
    return out


# ----------------------------------------------------------------------------- calculus
def interior(x, order=ORDER):
    """Samples on which `ddt` (and the residuals) are defined."""
    p = order // 2
    return x[..., p:x.shape[-2] - p, :]


def ddt(x, h, order=ORDER):
    """Central difference along dim -2: order 2 (3-point), 4 (5-point) or 6 (7-point)."""
    n = x.shape[-2]
    s = lambda k: x[..., order // 2 + k:n - order // 2 + k, :]
    if order == 2:
        return (s(1) - s(-1)) / (2 * h)
    if order == 4:
        return (-s(2) + 8 * s(1) - 8 * s(-1) + s(-2)) / (12 * h)
    if order == 6:
        return (s(3) - 9 * s(2) + 45 * s(1) - 45 * s(-1) + 9 * s(-2) - s(-3)) / (60 * h)
    raise ValueError("order must be 2, 4 or 6")


def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def rotation(eta):
    """Rzyx(phi, theta, psi) and the Euler-rate matrix T(phi, theta), shape (..., 3, 3)."""
    cphi, sphi = torch.cos(eta[..., 3]), torch.sin(eta[..., 3])
    cth, sth = torch.cos(eta[..., 4]), torch.sin(eta[..., 4])
    cpsi, spsi = torch.cos(eta[..., 5]), torch.sin(eta[..., 5])
    R = torch.stack([
        torch.stack([cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth], -1),
        torch.stack([spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi], -1),
        torch.stack([-sth, cth * sphi, cth * cphi], -1)], -2)
    one, zero = torch.ones_like(cphi), torch.zeros_like(cphi)
    Tm = torch.stack([
        torch.stack([one, sphi * sth / cth, cphi * sth / cth], -1),
        torch.stack([zero, cphi, -sphi], -1),
        torch.stack([zero, sphi / cth, cphi / cth], -1)], -2)
    return R, Tm


def eta_dot(eta, nu):
    """Kinematics eta_dot = J(eta) nu."""
    R, Tm = rotation(eta)
    return torch.cat([(R @ nu[..., :3, None])[..., 0], (Tm @ nu[..., 3:, None])[..., 0]], -1)


def kinematic_residual(eta, nu, h, order=ORDER):
    """eta_dot (finite differences) - J(eta) nu on the interior samples; psi is unwrapped first."""
    eta = eta.clone()
    eta[..., 5] = torch.as_tensor(np.unwrap(eta[..., 5].detach().cpu().numpy(), axis=-1),
                                  dtype=eta.dtype, device=eta.device)
    return ddt(eta, h, order) - interior(eta_dot(eta, nu), order)


def nu_from_eta(eta, eta_rate, u, v):
    """Body velocities from attitude/heave rates when only (D, phi, theta, psi) and their rates are
    predicted and surge/sway speeds u, v are known (measured or assumed):
      [p q r] = T(eta)^-1 [phi_dot theta_dot psi_dot],  w from D_dot = R[2, :] [u v w]."""
    cphi, sphi = torch.cos(eta[..., 3]), torch.sin(eta[..., 3])
    cth, sth = torch.cos(eta[..., 4]), torch.sin(eta[..., 4])
    dphi, dth, dpsi = eta_rate[..., 3], eta_rate[..., 4], eta_rate[..., 5]
    p = dphi - sth * dpsi
    q = cphi * dth + cth * sphi * dpsi
    r = -sphi * dth + cth * cphi * dpsi
    w = (eta_rate[..., 2] + sth * u - cth * sphi * v) / (cth * cphi)
    return torch.stack([u, v, w, p, q, r], -1)


# ----------------------------------------------------------------------------- model
class VesselModel:
    """Coefficients of one run (<run>_coef.json) and the forces of sim/hydroVesselMemFast.m."""

    def __init__(self, coef, dtype=torch.float64, device="cpu"):
        if coef.get("model") != "cummins_memory":
            raise ValueError("expected a coef v2 file with model = cummins_memory (sim/export_coef.m)")
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=dtype, device=device)
        self.dtype, self.device = dtype, device
        self.coef = coef
        main = coef["main"]
        self.m = float(main["m"])
        self.M = T(coef["M_inf"])
        self.G = T(coef["G"])
        self.Dv = T(coef["Dv"])
        self.Dref = T(coef["Dref"])
        self.kappa = [float(k) for k in coef["kappa"]]            # kappa_4, kappa_5, kappa_6
        self.Ig = self.m * torch.diag(T([main["k44"] ** 2, main["k55"] ** 2, main["k66"] ** 2]))
        MA = T(coef["MA_inf"])
        MA = 0.5 * (MA + MA.T)                                     # m2c symmetrizes
        self.M11, self.M12, self.M21, self.M22 = MA[:3, :3], MA[:3, 3:], MA[:3, 3:].T, MA[3:, 3:]
        mem = coef["memory"]
        self.Ar, self.Br = T(mem["Ar"]), T(mem["Br"]).reshape(-1, 6)
        self.Cr, self.Dr = T(mem["Cr"]).reshape(6, -1), T(mem["Dr"])
        self.n_mem = int(mem["n"])
        # cross-flow drag (prepare_fast.m) and ITTC surge resistance constants
        L, B, Td = float(main["Lpp"]), float(main["B"]), float(main["T"])
        dx = L / N_STRIPS
        self.xL = T(-L / 2 + (np.arange(1, N_STRIPS + 1) - 0.5) * dx)
        self.kcf = 0.5 * RHO_CROSSFLOW * Td * hoerner(B, Td) * dx
        C_B = main.get("C_B") or 0.65
        self.L = L
        self.Xuu_const = -0.5 * float(main["rho"]) * (1.025 * L * (C_B * B + 1.7 * Td)) * 1.1
        self.Dv_offdiag = bool(torch.any(self.Dv - torch.diag(torch.diag(self.Dv)) != 0))
        const = coef["constant_model"]
        self.M_const, self.D_const = T(const["M"]), T(const["D"])
        self._disc = {}
        self._lin = {}

    @classmethod
    def from_json(cls, path, **kw):
        with open(path) as f:
            return cls(json.load(f), **kw)

    # forces on the left-hand side, without inertia and memory
    def quadratic_damping(self, nu):
        """kappa_4 Dref_4 |p| p, kappa_5 Dref_5 |q| q, Dv66 kappa_6 |r| r in rows 4-6 (zero slope at nu = 0,
        so a linearization drops it)."""
        k4, k5, k6 = self.kappa
        p, q, r = nu[..., 3], nu[..., 4], nu[..., 5]
        zero = torch.zeros_like(p)
        return torch.stack([zero, zero, zero, k4 * self.Dref[3] * torch.abs(p) * p,
                            k5 * self.Dref[4] * torch.abs(q) * q, self.Dv[5, 5] * k6 * torch.abs(r) * r], -1)

    def lhs_static(self, nu, eta):
        """C_RB nu + C_A nu + D(nu) nu - tau_cf(nu) + G eta."""
        v1, v2 = nu[..., :3], nu[..., 3:]
        crb = torch.cat([self.m * cross(v2, v1), -cross(v2 @ self.Ig.T, v2)], -1)
        a1 = v1 @ self.M11.T + v2 @ self.M12.T
        a2 = v1 @ self.M21.T + v2 @ self.M22.T
        ca = torch.cat([-cross(a1, v2), -cross(a1, v1) - cross(a2, v2)], -1)

        u = nu[..., 0]
        Re = torch.clamp(self.L * torch.abs(u) / 1e-6, min=1e5)
        Xuu = self.Xuu_const * 0.075 / (torch.log10(Re) - 2) ** 2
        d = nu @ self.Dv.T if self.Dv_offdiag else nu * torch.diag(self.Dv)
        d = torch.cat([
            ((self.Dv[0, 0] * torch.exp(-3 * torch.abs(u)) - Xuu * torch.abs(u)) * u)[..., None],  # replaces row 1, as the simulator
            d[..., 1:]], -1) + self.quadratic_damping(nu)

        Uh = nu[..., 1, None] + self.xL * nu[..., 5, None]
        Uv = nu[..., 2, None] + self.xL * nu[..., 4, None]
        Qh, Qv = torch.abs(Uh) * Uh, torch.abs(Uv) * Uv
        zero = torch.zeros_like(u)
        tcf = -self.kcf * torch.stack([zero, Qh.sum(-1), Qv.sum(-1), zero, Qv @ self.xL, Qh @ self.xL], -1)
        return crb + ca + d - tcf + eta @ self.G.T

    # radiation memory
    def _discretize(self, h):
        """Exact discretization of xr_dot = Ar xr + Br nu over one step for a polynomial input
        nu(t_k + s) = sum_j c_j (s/h)^j:  xr[k+1] = Phi xr[k] + sum_j P_j c_j,
        P_j = int_0^h e^{Ar (h - s)} Br (s/h)^j ds  (augmented matrix exponential, chain of integrators).
        Returns Phi, the first-order-hold weights (nu[k], nu[k+1]) and the cubic weights
        (nu[k-1], nu[k], nu[k+1], nu[k+2], Lagrange interpolation through the four samples)."""
        if h not in self._disc:
            n, m, J = self.n_mem, 6, 4
            Z = torch.zeros(n + J * m, n + J * m, dtype=self.dtype, device=self.device)
            Z[:n, :n], Z[:n, n:n + m] = self.Ar, self.Br
            for j in range(J - 1):
                Z[n + j * m:n + (j + 1) * m, n + (j + 1) * m:n + (j + 2) * m] = torch.eye(
                    m, dtype=self.dtype, device=self.device)
            E = torch.linalg.matrix_exp(Z * h)
            Phi = E[:n, :n]
            # E block j = int e^{A(h-s)} B s^j / j! ds
            P = [math.factorial(j) / h ** j * E[:n, n + j * m:n + (j + 1) * m] for j in range(J)]
            foh = (P[0] - P[1], P[1])
            # Lagrange basis on the nodes s/h = -1, 0, 1, 2, as polynomial coefficients in s/h
            nodes = [-1.0, 0.0, 1.0, 2.0]
            cubic = []
            for i, xi in enumerate(nodes):
                others = [xk for k, xk in enumerate(nodes) if k != i]
                c = np.poly1d(others, r=True) / np.prod([xi - xk for xk in others])
                a = c.coeffs[::-1]                                  # ascending powers
                cubic.append(sum(float(a[j]) * P[j] for j in range(J)))
            self._disc[h] = (Phi, foh, cubic)
        return self._disc[h]

    def memory_states(self, nu, h, xr0=None):
        """Memory states (..., T, n) driven by nu (..., T, 6), starting from xr0 (zeros if None).
        nu is interpolated by the cubic through four neighbouring samples (linear on the first and
        last step); the linear hold alone leaves 0.3-1.3 % of the wave load in the residual."""
        Phi, (F0, F1), (Cm, C0, C1, C2) = self._discretize(h)
        T = nu.shape[-2]
        x = torch.zeros(*nu.shape[:-2], self.n_mem, dtype=nu.dtype, device=nu.device) if xr0 is None else xr0
        drive = nu[..., :-1, :] @ F0.T + nu[..., 1:, :] @ F1.T                  # step k -> k + 1
        if T >= 4:
            mid = (nu[..., :T - 3, :] @ Cm.T + nu[..., 1:T - 2, :] @ C0.T
                   + nu[..., 2:T - 1, :] @ C1.T + nu[..., 3:, :] @ C2.T)          # steps 1 .. T - 3
            drive = torch.cat([drive[..., :1, :], mid, drive[..., -1:, :]], -2)
        out = torch.empty(*nu.shape[:-2], T, self.n_mem, dtype=nu.dtype, device=nu.device)
        out[..., 0, :] = x
        for k in range(T - 1):
            x = x @ Phi.T + drive[..., k, :]
            out[..., k + 1, :] = x
        return out

    def memory_force(self, nu, xr):
        return xr @ self.Cr.T + nu @ self.Dr.T

    # linearization about steady forward speed
    def linearize(self, U):
        """lhs_static ~ F0 + K_nu (nu - nu0) + G eta about nu0 = [U 0 0 0 0 0], eta = 0."""
        if U not in self._lin:
            nu0 = torch.zeros(6, dtype=self.dtype, device=self.device)
            nu0[0] = U
            eta0 = torch.zeros(6, dtype=self.dtype, device=self.device)
            K_nu = torch.autograd.functional.jacobian(lambda n: self.lhs_static(n, eta0), nu0)
            self._lin[U] = (nu0, self.lhs_static(nu0, eta0), K_nu)
        return self._lin[U]

    # residual
    def residual(self, nu, eta, tau_ext, h, level="full", U=None, xr0=None, order=ORDER, quadratic=False, mu=None):
        """Equation-of-motion residual on the interior samples, (..., T - 2p, 6), in N and N m.
        nu, eta, tau_ext: (..., T, 6); tau_ext = held(tau_ctrl) + tau_wave at the same samples.
        level: 'full' | 'linear' (about U) | 'constant'. xr0: memory state at the first sample.
        quadratic: add the roll/pitch/yaw quadratic damping to 'linear' (the PINN physics planned in
        Bao_cao_tien_do_Giai_doan_0-1.md section 9).
        mu: memory force (..., T, 6) already computed for this nu (skips the memory recursion)."""
        nu_i, eta_i, tau_i = interior(nu, order), interior(eta, order), interior(tau_ext, order)
        acc = ddt(nu, h, order)
        if level == "constant":
            return acc @ self.M_const.T + nu_i @ self.D_const.T + eta_i @ self.G.T - tau_i
        if mu is None:
            mu = self.memory_force(nu, self.memory_states(nu, h, xr0))
        mu = interior(mu, order)
        if level == "full":
            lhs = self.lhs_static(nu_i, eta_i)
        elif level == "linear":
            if U is None:
                raise ValueError("level 'linear' needs the forward speed U")
            nu0, F0, K_nu = self.linearize(float(U))
            lhs = F0 + (nu_i - nu0) @ K_nu.T + eta_i @ self.G.T
            if quadratic:
                lhs = lhs + self.quadratic_damping(nu_i)
        else:
            raise ValueError(f"unknown level {level!r}")
        return acc @ self.M.T + lhs + mu - tau_i


def relative_residual(r, tau_wave, order=ORDER, skip=0):
    """RMS of the residual per DOF divided by the std of the wave load in that DOF (after `skip` samples)."""
    tw = interior(tau_wave, order)[..., skip:, :]
    rr = r[..., skip:, :]
    return torch.sqrt((rr ** 2).mean(-2)) / tw.std(-2)
