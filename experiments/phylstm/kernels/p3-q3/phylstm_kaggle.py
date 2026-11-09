"""PhyLSTM2 (Zhang, Liu, Sun 2020, CMAME 369:113226, arXiv:2002.10253) adapted to deck-motion forecasting on dataset_v2,
under evaluation protocol (a): the splits, forecast origins, horizon and metrics of experiments/run_sim_lstm_cv.py.

Self-contained (numpy + torch), for a Kaggle GPU T4 x2 kernel (one worker process per GPU) or a local GPU:
  python phylstm_kaggle.py                          # every job; data under /kaggle/input or data/sim/dataset_v2/kaggle_phylstm
  python phylstm_kaggle.py --smoke                  # tiny run, separate output file
  python phylstm_kaggle.py --jobs cv0:phy cv0:data  # some jobs
Input: usv_phylstm.npz from experiments/phylstm/pack_kaggle.py; the extended pack (pack_kaggle.py --v2, with
cummins_groups.npz beside it) is preferred when found (find_data) and is required by the phase-2 variants. Output (/kaggle/working, else results/sim/phylstm/):
phylstm_metrics.npz with '<split>|<variant>|mse', '|mae' = per-run mean e^2 and |e| per lead (R, 100, 3) and '|mse_o' =
per-origin MSE over the 20 s (R, O, 3), errors in m (heave) and degrees (roll, pitch), the layout of
lstm_cv_metrics.npz; '<split>|ids' = run ids; phylstm_history.json = losses per epoch.

Forecast task, as the LSTM baseline: at an origin o the predictor sees only the past 60 s of eta = (D, phi, theta) and
outputs the next 20 s. Nothing about future waves enters the predictor.
  Network 1 (paper Fig. 2, Box I): a 400-step window = the 300 past samples of eta, then 100 future steps where eta is
    0 and a flag channel is 1. 3 x [LSTM(100), ReLU], Dense(100), Dense(9) (layout of the official PhyLSTM2.py), mapped
    pairwise in time to Z = (z1 = eta, z2 = eta_dot, z3 = r, a latent state as the paper's hysteretic r). Unidirectional,
    so the output at t only sees inputs up to t. The forecast is z1 on the last 100 steps.
  Network 2 (Box II): g = LSTM2([Z, P]), same layout with Dense(3): the mass-normalised restoring force, P = run
    parameters (vessel, U, kappa, visc). Used only inside the training loss: neither P nor f reaches the predictor.
  Differentiator (Box III): 4th-order finite differences (official code: 2nd order, which misses eta_dot by 18-27 % at
    5 Hz on this data).
  Losses (eq. 9-11, 13; alpha = beta = gamma = 1, every term in units of its train std), on the whole window:
    J_d = MSE(z1 - eta) + MSE(z2 - eta_dot)      eta_dot from the kinematics (pack_kaggle.py)
    J_e = MSE(D z1 - z2)
    J_g = MSE(D z2 + g - f)                      f = the true wave excitation tau_w / M_ii: a training-only signal
  variant 'data' = Network 1 with J_d only (the paper's LSTM comparison); variant 'phy' = PhyLSTM2.
  Controls (ablations of 'phy' / 'data'; the predictor is Network 1 in every case):
    'je'       J_d + J_e (no Network 2, no J_g)
    'phy_past' 'phy' with J_g averaged over the 300 past steps only: the true wave force of the future 20 s is never used
    'anchor'   'data' with Network 1's eta on the 100 future steps added to the last observed (normalised) eta, the
               input at step CTX-1: it forecasts increments from the last observation (training, validation and test)
    'sobolev'  'data' + SOBOLEV_LAM * (MSE(De)/s1^2 + MSE(D^2e)/s2^2) on the whole window, e = normalised eta error,
               D = first difference in time, s1, s2 = train std of D eta, D^2 eta (normalised, per channel): a generic
               smoothness-of-error term with no physics
  Phase 2 (need the extended pack kaggle_phylstm_v2: usv_phylstm.npz with nu, group, sea + cummins_groups.npz from
  experiments/phylstm/cummins_coef.py; the predictor is again Network 1 alone, no Network 2):
    'cummins'  Cummins-equation PINN. Network 1's 9 outputs are z1 = eta, z2 = nu_hpr = (w, p, q) body velocities (J_d
               against the logged nu instead of the kinematic eta_dot), z3 = r (latent, unused). On the whole window:
                 J_d   = MSE(z1 - eta) + MSE(z2 - nu_hpr)                         (train-std units)
                 J_kin = MSE([D eta^ - J(eta^) nu^] - [D eta - J(eta) nu]) / sd(eta_dot)^2    rows D, phi, theta;
                         nu^ = true (u, v, r) with the predicted (w, p, q): the subtraction removes the FD error
                 J_dyn = MSE([R(y^) - R(y)] / s_tau)    R = linearised Cummins residual of the run's coefficient group,
                         rows heave/roll/pitch; y^ = prediction with the true surge/sway/yaw states, so with
                         d = y^ - y (zero in u, v, r, N, E, psi):  R(y^) - R(y) = M d_nu' + D_lin d_nu + G d_eta
                         + sum_j K[j] d_nu[t-j] + c_quad (|nu^| nu^ - |nu| nu)  (3 x 3 blocks of rows/cols 2-4;
                         tau_wave, tau_ctrl, F0, nu0 cancel; d = 0 before the window: history = truth);
                         s_tau = train std of tau_w3..5 (= f * diag(M) of the group)
               L = J_d + J_kin + J_dyn. The true wave load never enters; future truth enters only through J_d, J_kin.
    'seastate', 'phy_seastate', 'cummins_seastate'   'data', 'phy', 'cummins' + SEA_W * MSE of an auxiliary
               Linear head on Network 1's last LSTM hidden state at t = CTX-1 (the last past step) predicting
               normalised (Hs, Tp, cos beta, sin beta, U) of the run (design values; beta is already relative to the
               heading). Training only: the forecast path is unchanged and the head is never used at test time.
Training: each epoch draws K random origins per training run; Adam lr 1e-3, 1e-4 for the last third of the epochs
(paper, example 2), gradient norm clipped at 1; the epoch with the lowest validation MSE of the 20 s forecast is kept
(paper: minimum validation loss). One seed per job.
Job spec 'split:variant[:seed[:epochs]]' (seed default --seed, epochs default --epochs). Result keys
'<split>|<tag>|mse' etc. and '<split>|<tag>_last|...' with tag = variant + '_s<seed>' if seed != 0 + '_e<epochs>' if
epochs != 30, so 'cv0:phy' -> 'cv0|phy|mse', 'cv1:phy:1' -> 'cv1|phy_s1|mse', 'cv0:data:0:45' -> 'cv0|data_e45|mse'.
History is keyed by the job spec as given.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

FS, DT = 5.0, 0.2
CTX, HORIZON, STRIDE = 300, 100, 5          # 60 s context, 20 s horizon, one origin per second (sim_common.py)
WIN = CTX + HORIZON
SPLITS = [f"cv{k}" for k in range(6)] + ["ood_Hs", "ood_beta"]
VARIANTS = ["phy", "data"]                  # the default job list (no --jobs): every split x these
CONTROLS = ["je", "phy_past", "anchor", "sobolev"]
PHASE2 = ["cummins", "seastate", "phy_seastate", "cummins_seastate"]
ALL_VARIANTS = VARIANTS + CONTROLS + PHASE2
WITH_NET2 = ("phy", "phy_past", "phy_seastate")
BASE = {"seastate": "data", "phy_seastate": "phy", "cummins_seastate": "cummins"}   # sea-state head on these losses
EXTENDED = ("cummins", "seastate", "phy_seastate", "cummins_seastate")   # need the extended pack (nu, group, sea)
SEA_W = 0.1                                 # weight of the auxiliary sea-state loss
CUMMINS_TAPS = 301                          # FIR taps of the radiation memory in J_dyn (60 s, all of cummins_groups K)
DEFAULT_EPOCHS = 30                         # epochs != this are tagged '_e<epochs>' in the result keys
SOBOLEV_LAM = 1.0
T0 = time.time()                           # process start: Kaggle stops a session after 12 h
DEFAULT_ARGS = ['--jobs', 'cv0:cummins_seastate:0:60', 'cv3:cummins_seastate:0:60', 'cv1:cummins_seastate:0:60', 'cv4:cummins_seastate:0:60', 'cv2:cummins_seastate:0:60', 'cv5:cummins_seastate:0:60', 'cv4:data:0:60', 'cv5:data:0:60', 'ood_Hs:data:0:60', 'ood_beta:data:0:60']  # make_kernel.py p3-q3
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARGV = sys.argv[1:] if any(x.startswith("--") for x in sys.argv[1:]) else DEFAULT_ARGS


def groups_file(data):
    """cummins_groups.npz next to the pack `data`, else anywhere under /kaggle/input, else None."""
    f = os.path.join(os.path.dirname(os.path.abspath(data)), "cummins_groups.npz")
    if os.path.exists(f):
        return f
    hits = sorted(glob.glob("/kaggle/input/**/cummins_groups.npz", recursive=True))
    return hits[0] if hits else None


def find_data(path):
    """The pack to read: --data if given; else under /kaggle/input, else locally, the extended pack (a usv_phylstm.npz
    with cummins_groups.npz beside it: data/sim/dataset_v2/kaggle_phylstm_v2) if present, else the original one. The
    original variants read only S, P, ids and the splits, identical in both packs."""
    if path:
        return path
    hits = sorted(glob.glob("/kaggle/input/**/usv_phylstm.npz", recursive=True))
    if not hits:
        sim = os.path.join(HERE, "..", "..", "data", "sim", "dataset_v2")
        hits = [f for f in (os.path.join(sim, d, "usv_phylstm.npz") for d in ("kaggle_phylstm_v2", "kaggle_phylstm"))
                if os.path.exists(f)] or [os.path.join(sim, "kaggle_phylstm", "usv_phylstm.npz")]
    ext = [f for f in hits if os.path.exists(os.path.join(os.path.dirname(f), "cummins_groups.npz"))]
    return (ext or hits)[0]


def load_extended(d, path):
    """Extra arrays of the extended pack for the phase-2 variants: nu (N, T, 6), group (N,), sea (N, 4) and the
    coefficient groups (cummins_groups.npz, rows aligned with the pack's ids)."""
    gf = groups_file(path)
    if "nu" not in d.files or gf is None:
        raise RuntimeError(f"{path}: the variants {EXTENDED} need the extended pack (kaggle_phylstm_v2: usv_phylstm.npz "
                           f"with nu/group/sea and cummins_groups.npz beside it); nu in pack: {'nu' in d.files}, "
                           f"cummins_groups.npz: {gf}")
    cg = np.load(gf)
    if not (np.array_equal(cg["ids"], d["ids"]) and np.array_equal(cg["run_group"], d["group"])):
        raise RuntimeError(f"{gf} does not match {path} (ids / run_group)")
    return {"nu": d["nu"], "group": d["group"].astype(np.int64), "sea": d["sea"], "groups_file": gf,
            **{k: cg[k] for k in ("M", "D_lin", "G", "c_quad", "K")}}


SMOKE = "--smoke" in ARGV
OUT = ARGV[ARGV.index("--out") + 1] if "--out" in ARGV[:-1] else None     # optional output directory (local tests)


def out_dir():
    d = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "..", "..", "results", "sim", "phylstm")
    d = OUT or d
    d = os.path.join(d, "smoke") if SMOKE else d
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- PhyLSTM pieces
class Stack(nn.Module):
    """3 x [LSTM(units), ReLU], Dense(units), Dense(n_out), as LSTM_model in the official PhyLSTM2.py."""

    def __init__(self, n_in, n_out, units=100):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(n_in if i == 0 else units, units, batch_first=True) for i in range(3)])
        self.d1 = nn.Linear(units, units)
        self.d2 = nn.Linear(units, n_out)

    def forward(self, x, h_at=None):
        """Output (B, T, n_out); with h_at also the last LSTM layer's hidden state at step h_at (B, units)."""
        h = None
        for i, lstm in enumerate(self.lstms):
            o = lstm(x)[0]
            if h_at is not None and i == len(self.lstms) - 1:
                h = o[:, h_at]
            x = torch.relu(o)
        out = self.d2(self.d1(x))
        return out if h_at is None else (out, h)


def ddt(x, dt=DT):
    """4th-order finite-difference time derivative of x (B, T, C), one-sided 4th order at both ends."""
    c = 1.0 / (12 * dt)
    first = (-25 * x[:, 0] + 48 * x[:, 1] - 36 * x[:, 2] + 16 * x[:, 3] - 3 * x[:, 4]) * c
    second = (-3 * x[:, 0] - 10 * x[:, 1] + 18 * x[:, 2] - 6 * x[:, 3] + x[:, 4]) * c
    mid = (-x[:, 4:] + 8 * x[:, 3:-1] - 8 * x[:, 1:-3] + x[:, :-4]) * c
    penult = (3 * x[:, -1] + 10 * x[:, -2] - 18 * x[:, -3] + 6 * x[:, -4] - x[:, -5]) * c
    last = (25 * x[:, -1] - 48 * x[:, -2] + 36 * x[:, -3] - 16 * x[:, -4] + 3 * x[:, -5]) * c
    return torch.cat([first[:, None], second[:, None], mid, penult[:, None], last[:, None]], dim=1)


class Job:
    def __init__(self, S, P, rows, dev):
        self.dev = dev
        self.S = torch.tensor(S, device=dev)                                  # (N, T, 9) eta | eta_dot | f, SI units
        self.P = torch.tensor(P, device=dev)
        tr = self.S[torch.tensor(rows["train"], device=dev)]
        flat = tr.reshape(-1, 9)
        self.mu_eta = flat[:, :3].mean(0)
        self.sd = flat.std(0)                                                  # eta, eta_dot, f scales
        self.pscale = torch.tensor([1.0, 2.0, 5.0, 1.0], device=dev)           # is_80kg, U, kappa, visc -> ~[0, 1]
        self.ar = torch.arange(WIN, device=dev)
        self.flag = torch.zeros(WIN, device=dev)
        self.flag[CTX:] = 1.0
        self.train_rows = rows["train"]

    def setup_cummins(self, ext):
        """'cummins*': body velocities on the device and the group tensors of J_dyn (3 x 3 blocks, rows/cols 2-4 =
        heave/roll/pitch of M, D_lin, G and the FIR kernel K; c_quad of roll, pitch)."""
        dev, b = self.dev, slice(2, 5)
        self.NU = torch.tensor(ext["nu"], device=dev)                          # (N, T, 6) u, v, w, p, q, r
        self.group = torch.tensor(ext["group"], device=dev)                    # (N,) row of the group tensors
        t = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        self.cM, self.cD, self.cG = (t(ext[k][:, b, b]) for k in ("M", "D_lin", "G"))       # (G, 3, 3)
        self.cK = t(ext["K"][:, :CUMMINS_TAPS, b, b])                          # (G, L, 3, 3), K[:, 0] = lag 0
        self.cQ = t(ext["c_quad"][:, :2])                                      # (G, 2) roll, pitch
        tr = torch.tensor(self.train_rows, device=dev)
        self.sd_nu = self.NU[tr].reshape(-1, 6).std(0)                         # nu scales (w, p, q: 2-4)
        Mii = torch.diagonal(self.cM, dim1=1, dim2=2)[self.group[tr]]          # (n_train, 3)
        self.s_tau = (self.S[tr, :, 6:9] * Mii[:, None, :]).reshape(-1, 3).std(0)   # tau_w3..5 = f * M_ii

    def setup_sea(self, ext):
        """Sea-state targets (N, 5): Hs, Tp, U standardised over the train runs, cos and sin of beta."""
        sea = torch.tensor(ext["sea"], dtype=torch.float32, device=self.dev)  # Hs, Tp, beta_deg, U
        z = sea[:, [0, 1, 3]]
        tr = z[torch.tensor(self.train_rows, device=self.dev)]
        z = (z - tr.mean(0)) / tr.std(0).clamp_min(1e-6)
        b = torch.deg2rad(sea[:, 2])
        self.sea_t = torch.stack([z[:, 0], z[:, 1], torch.cos(b), torch.sin(b), z[:, 2]], dim=1)

    def nu_windows(self, runs, origins):
        """(B, WIN, 6) body velocities of the same windows as `windows`."""
        return self.NU[runs[:, None], (origins - CTX)[:, None] + self.ar]

    def diff_scales(self):
        """'sobolev': train std of the first and second time differences of normalised eta, per channel."""
        d1 = (self.S[torch.tensor(self.train_rows, device=self.dev), :, :3] / self.sd[:3]).diff(dim=1)
        s1 = d1.reshape(-1, 3).std(0)
        s2 = d1.diff(dim=1).reshape(-1, 3).std(0)
        self.s1, self.s2 = s1, s2
        return s1, s2

    def windows(self, runs, origins):
        """runs, origins: (B,) tensors -> W (B, WIN, 9) for times o - CTX .. o + HORIZON - 1."""
        return self.S[runs[:, None], (origins - CTX)[:, None] + self.ar]

    def net1_input(self, W):
        eta = (W[..., :3] - self.mu_eta) / self.sd[:3]
        eta = eta * (1.0 - self.flag)[None, :, None]                           # future eta hidden
        return torch.cat([eta, self.flag[None, :, None].expand(len(W), WIN, 1)], dim=-1)

    def unpack(self, Z):
        """Network-1 output (B, WIN, 9) in train-std units -> eta, eta_dot in SI units, r."""
        return Z[..., :3] * self.sd[:3] + self.mu_eta, Z[..., 3:6] * self.sd[3:6], Z[..., 6:]


def net1_out(job, net1, W, variant=None, with_h=False):
    """Network-1 output Z (B, WIN, 9) in train-std units. 'anchor': the eta of the future steps is an increment on the
    last observed normalised eta (input step CTX-1, which net1_input never masks). with_h: also the last LSTM layer's
    hidden state at the last past step CTX-1 (the sea-state head's input)."""
    x = job.net1_input(W)
    Z, h = net1(x, h_at=CTX - 1) if with_h else (net1(x), None)
    if variant == "anchor":
        fut = torch.cat([Z[:, CTX:, :3] + x[:, CTX - 1:CTX, :3], Z[:, CTX:, 3:]], dim=-1)
        Z = torch.cat([Z[:, :CTX], fut], dim=1)
    return (Z, h) if with_h else Z


def kin_rates(eta, nu):
    """eta_dot = J(eta) nu for (D, phi, theta) from nu (.., 6) = u, v, w, p, q, r (pack_kaggle.features)."""
    phi, th = eta[..., 1], eta[..., 2]
    u, v, w, p, q, r = nu.unbind(-1)
    return torch.stack([-torch.sin(th) * u + torch.cos(th) * torch.sin(phi) * v + torch.cos(th) * torch.cos(phi) * w,
                        p + torch.sin(phi) * torch.tan(th) * q + torch.cos(phi) * torch.tan(th) * r,
                        torch.cos(phi) * q - torch.sin(phi) * r], dim=-1)


def fir_batched(x, K):
    """mu (B, T, C) = sum_j K[b, j] x[b, t - j] with x = 0 before t = 0; x (B, T, C), K (B, L, C, C) per sample.
    One grouped conv1d (cummins_coef.fir_force with a kernel per sample)."""
    B, T, C = x.shape
    L = K.shape[1]
    w = K.flip(1).permute(0, 2, 3, 1).reshape(B * C, C, L)                    # (b, out) x in x taps, w[..., -1] = K[0]
    xx = nn.functional.pad(x.transpose(1, 2).reshape(1, B * C, T), (L - 1, 0))
    return nn.functional.conv1d(xx, w, groups=B).reshape(B, C, T).transpose(1, 2)


def cummins_terms(job, eta, nuh, W, Nw, runs):
    """J_d, J_kin, J_dyn of 'cummins' for predicted eta (B, WIN, 3) and (w, p, q) nuh (B, WIN, 3), SI units, against
    the true windows W (S channels) and Nw (nu). Zero J_kin and J_dyn when eta = W[..., :3] and nuh = Nw[..., 2:5]."""
    sd = job.sd
    nu = Nw[..., 2:5]
    Jd = (((eta - W[..., :3]) / sd[:3]) ** 2).mean() + (((nuh - nu) / job.sd_nu[2:5]) ** 2).mean()
    nu_hat = torch.cat([Nw[..., :2], nuh, Nw[..., 5:]], dim=-1)               # true u, v, r; predicted w, p, q
    kin = (ddt(eta) - kin_rates(eta, nu_hat)) - (ddt(W[..., :3]) - kin_rates(W[..., :3], Nw))
    Jkin = ((kin / sd[3:6]) ** 2).mean()
    g = job.group[runs]
    dnu, deta = nuh - nu, eta - W[..., :3]                                    # zero before the window (history = truth)
    quad = job.cQ[g][:, None, :] * (nuh[..., 1:].abs() * nuh[..., 1:] - nu[..., 1:].abs() * nu[..., 1:])
    dR = (torch.einsum("bij,btj->bti", job.cM[g], ddt(dnu)) + torch.einsum("bij,btj->bti", job.cD[g], dnu)
          + torch.einsum("bij,btj->bti", job.cG[g], deta) + fir_batched(dnu, job.cK[g])
          + torch.cat([torch.zeros_like(quad[..., :1]), quad], dim=-1))
    Jdyn = ((dR / job.s_tau) ** 2).mean()
    return Jd, Jkin, Jdyn


def losses(job, net1, net2, W, runs, variant, origins=None, head=None):
    """Training loss and its parts for a batch of windows W (B, WIN, 9) of runs/origins. Sea-state variants: the base
    variant's loss + SEA_W * MSE(head(h_{CTX-1}) - sea targets); `head` is None for the other variants."""
    if head is None:
        return base_losses(job, net2, net1_out(job, net1, W, variant), W, runs, variant, origins)
    base = BASE[variant]
    Z, h = net1_out(job, net1, W, base, with_h=True)
    loss, parts = base_losses(job, net2, Z, W, runs, base, origins)
    Jsea = SEA_W * ((head(h) - job.sea_t[runs]) ** 2).mean()
    parts["Jsea"] = Jsea.item()
    return loss + Jsea, parts


def base_losses(job, net2, Z, W, runs, variant, origins):
    if variant == "cummins":
        Jd, Jkin, Jdyn = cummins_terms(job, Z[..., :3] * job.sd[:3] + job.mu_eta, Z[..., 3:6] * job.sd_nu[2:5], W,
                                       job.nu_windows(runs, origins), runs)
        return Jd + Jkin + Jdyn, {"Jd": Jd.item(), "Jkin": Jkin.item(), "Jdyn": Jdyn.item()}
    eta, etad, _ = job.unpack(Z)
    sd = job.sd
    Jd = (((eta - W[..., :3]) / sd[:3]) ** 2).mean() + (((etad - W[..., 3:6]) / sd[3:6]) ** 2).mean()
    if variant in ("data", "anchor"):
        return Jd, {"Jd": Jd.item()}
    if variant == "sobolev":
        e1 = ((eta - W[..., :3]) / sd[:3]).diff(dim=1)                     # normalised error, first difference
        Js = SOBOLEV_LAM * (((e1 / job.s1) ** 2).mean() + ((e1.diff(dim=1) / job.s2) ** 2).mean())
        return Jd + Js, {"Jd": Jd.item(), "Js": Js.item()}
    Je = (((ddt(eta) - etad) / sd[3:6]) ** 2).mean()
    if variant == "je":
        return Jd + Je, {"Jd": Jd.item(), "Je": Je.item()}
    p = (job.P[runs] / job.pscale)[:, None, :].expand(len(W), WIN, 4)
    if variant == "phy_past":                  # Network 2 is causal: g at t < CTX only sees Z up to t
        g = net2(torch.cat([Z, p], dim=-1)[:, :CTX]) * sd[6:9]
        Jg = (((ddt(etad)[:, :CTX] + g - W[:, :CTX, 6:9]) / sd[6:9]) ** 2).mean()
    else:
        g = net2(torch.cat([Z, p], dim=-1)) * sd[6:9]
        Jg = (((ddt(etad) + g - W[..., 6:9]) / sd[6:9]) ** 2).mean()
    return Jd + Je + Jg, {"Jd": Jd.item(), "Je": Je.item(), "Jg": Jg.item()}


@torch.no_grad()
def forecast(job, net1, runs, origins, batch=512, variant=None):
    """eta forecast (B, HORIZON, 3) in SI units for the given runs/origins."""
    out = []
    for b in range(0, len(runs), batch):
        W = job.windows(runs[b:b + batch], origins[b:b + batch])
        out.append(job.unpack(net1_out(job, net1, W, variant))[0][:, CTX:])
    return torch.cat(out)


def to_report_units(x):
    """(..., 3) heave m, roll/pitch rad -> heave m, roll/pitch deg (the units of the LSTM baseline)."""
    return torch.cat([x[..., :1], torch.rad2deg(x[..., 1:])], dim=-1)


def result_tag(variant, seed, epochs):
    """Variant part of the result keys: 'phy', 'phy_s1', 'data_e45', 'phy_s1_e45'."""
    return variant + (f"_s{seed}" if seed != 0 else "") + (f"_e{epochs}" if epochs != DEFAULT_EPOCHS else "")


def parse_spec(spec, a):
    """'split:variant[:seed[:epochs]]' -> (split, variant, seed, epochs, tag); seed/epochs default to --seed/--epochs."""
    parts = spec.split(":")
    if not 2 <= len(parts) <= 4 or parts[0] not in SPLITS or parts[1] not in ALL_VARIANTS:
        raise ValueError(f"bad job spec {spec!r}: want split:variant[:seed[:epochs]], split in {SPLITS}, "
                         f"variant in {ALL_VARIANTS}")
    seed = int(parts[2]) if len(parts) > 2 else a.seed
    epochs = int(parts[3]) if len(parts) > 3 else a.epochs
    if seed < 0 or epochs < 1:
        raise ValueError(f"bad job spec {spec!r}: seed must be >= 0 and epochs >= 1")
    return parts[0], parts[1], seed, epochs, result_tag(parts[1], seed, epochs)


def run_job(S, P, rows, split, variant, a, dev, log, seed=None, epochs=None, tag=None, ext=None):
    """Train and score one job. seed/epochs default to a.seed/a.epochs and tag (the variant part of the result keys)
    to the variant, as before job specs carried them (experiments/phylstm/checks/rerun.py). --smoke: 2 epochs.
    ext: load_extended(...), needed by the EXTENDED variants only."""
    seed = a.seed if seed is None else seed
    epochs = a.epochs if epochs is None else epochs
    tag = variant if tag is None else tag
    n_ep = 2 if getattr(a, "smoke", False) else epochs
    name = f"{split}:{tag}"
    torch.manual_seed(seed); np.random.seed(seed)
    job = Job(S, P, rows, dev)
    if variant in EXTENDED and ext is None:
        raise RuntimeError(f"{name}: needs the extended pack (load_extended)")
    if variant == "sobolev":
        s1, s2 = job.diff_scales()
        log(f"  {name}: sobolev scales s1 {s1.tolist()}, s2 {s2.tolist()} (normalised eta units), lambda {SOBOLEV_LAM}")
    if BASE.get(variant, variant) == "cummins":
        job.setup_cummins(ext)
        log(f"  {name}: cummins, {CUMMINS_TAPS} FIR taps, s_tau {job.s_tau.tolist()}, sd nu_hpr {job.sd_nu[2:5].tolist()}")
    if variant in BASE:
        job.setup_sea(ext)
    net1 = Stack(4, 9).to(dev)
    net2 = Stack(9 + 4, 3).to(dev) if variant in WITH_NET2 else None
    head = None
    if variant in BASE:                         # own CPU RNG stream: net1/net2 init and origins as in the base variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10_000 + seed)
            head = nn.Linear(net1.d1.in_features, 5).to(dev)
    params = list(net1.parameters()) + (list(net2.parameters()) if net2 else []) \
        + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    T = S.shape[1]
    tr = torch.tensor(rows["train"], device=dev)
    rng = np.random.default_rng(seed)
    va_runs = np.repeat(rows["val"], a.val_origins)
    va_orig = rng.integers(CTX, T - HORIZON + 1, size=len(va_runs))
    va_runs, va_orig = torch.tensor(va_runs, device=dev), torch.tensor(va_orig, device=dev)
    best, best_ep, hist = np.inf, 0, []
    state = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
    for ep in range(n_ep):
        if (time.time() - T0) / 3600 > a.max_hours - 0.25:
            log(f"  {name} stopped after {ep} epochs: {a.max_hours} h limit")
            break
        for g_ in opt.param_groups:
            g_["lr"] = 1e-3 if ep < int(np.ceil(2 * n_ep / 3)) else 1e-4
        net1.train(); net2 and net2.train(); head and head.train()
        t0, tot, n = time.time(), {}, 0
        runs = tr.repeat_interleave(a.k)
        orig = torch.randint(CTX, T - HORIZON + 1, (len(runs),), device=dev)
        perm = torch.randperm(len(runs), device=dev)
        for b in range(0, len(perm), a.batch):
            k = perm[b:b + a.batch]
            W = job.windows(runs[k], orig[k])
            opt.zero_grad()
            loss, parts = losses(job, net1, net2, W, runs[k], variant, origins=orig[k], head=head)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for key, val in parts.items():
                tot[key] = tot.get(key, 0.0) + val * len(k)
            n += len(k)
        net1.eval()
        pred = forecast(job, net1, va_runs, va_orig, variant=variant)
        truth = job.windows(va_runs, va_orig)[:, CTX:, :3]
        val = (((pred - truth) / job.sd[:3]) ** 2).mean().item()
        rec = {"epoch": ep + 1, **{k_: v / n for k_, v in tot.items()}, "val_mse": val, "s": round(time.time() - t0, 1)}
        hist.append(rec)
        if val < best:
            best, best_ep = val, ep + 1
            state = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
        log(f"  {name} epoch {ep + 1}/{n_ep} " + " ".join(f"{k_} {v:.4f}" for k_, v in rec.items()
                                                       if k_ not in ("epoch", "s")) + f" ({rec['s']} s)")
    if best_ep == 0 or not np.isfinite(best):
        raise RuntimeError(f"{name}: validation never improved (best {best}), not scoring an untrained net")
    last = {k_: v.detach().clone() for k_, v in net1.state_dict().items()}
    wdir = os.path.join(out_dir(), "weights")                                # Network 1 + normalisation, for forecasts
    os.makedirs(wdir, exist_ok=True)                                          # later (experiments/landing)
    torch.save({"best": {k_: v.cpu() for k_, v in state.items()}, "last": {k_: v.cpu() for k_, v in last.items()},
                "mu_eta": job.mu_eta.cpu(), "sd": job.sd.cpu(), "variant": variant, "split": split, "tag": tag,
                "seed": seed, "epochs": n_ep, "best_epoch": best_ep}, os.path.join(wdir, f"{split}__{tag}.pt"))
    res = {}
    for suf, st in (("", state), ("_last", last)):                       # best-on-val (paper) and last epoch (baseline)
        net1.load_state_dict(st); net1.eval()
        mse, mae, mse_o = test_metrics(job, net1, rows["test"], S.shape[1], variant)
        res.update({f"{split}|{tag}{suf}|mse": mse, f"{split}|{tag}{suf}|mae": mae, f"{split}|{tag}{suf}|mse_o": mse_o,
                    f"{split}|{tag}{suf}|n_epochs": np.array([len(hist), n_ep])})        # trained, planned
        rmse = np.sqrt(mse.mean(axis=(0, 1)))
        log(f"  {name}{suf} ({'best epoch ' + str(best_ep) if not suf else 'epoch ' + str(len(hist))}, val "
            f"{best if not suf else hist[-1]['val_mse']:.4f}): test RMSE heave {rmse[0]:.4f} m, roll {rmse[1]:.3f} deg, "
            f"pitch {rmse[2]:.3f} deg")
    return res, {"best_epoch": best_ep, "best_val": best, "seed": seed, "epochs": epochs, "history": hist}


def test_metrics(job, net1, te, T, variant=None):
    """Forecast from every origin of the test runs; per-run mean e^2, |e| per lead and per-origin MSE (as
    run_sim_lstm_cv.score), errors in m and degrees."""
    o = np.arange(CTX, T - HORIZON + 1, STRIDE)
    R, O = len(te), len(o)
    mse = np.empty((R, HORIZON, 3), np.float32); mae = np.empty_like(mse); mse_o = np.empty((R, O, 3), np.float32)
    for r0 in range(0, R, 40):
        rr = te[r0:r0 + 40]
        runs = torch.tensor(np.repeat(rr, O), device=job.dev)
        orig = torch.tensor(np.tile(o, len(rr)), device=job.dev)
        e = to_report_units(forecast(job, net1, runs, orig, variant=variant)) - to_report_units(job.windows(runs, orig)[:, CTX:, :3])
        e = e.reshape(len(rr), O, HORIZON, 3).double().cpu().numpy()
        mse[r0:r0 + len(rr)] = (e ** 2).mean(axis=1)
        mae[r0:r0 + len(rr)] = np.abs(e).mean(axis=1)
        mse_o[r0:r0 + len(rr)] = (e ** 2).mean(axis=2)
    return mse, mae, mse_o


def worker(a, jobs, tag):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = find_data(a.data)
    d = np.load(path)
    S, P, ids = d["S"], d["P"], d["ids"]
    out, hist = {}, {}
    logf = open(os.path.join(out_dir(), f"log_{tag}.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"worker {tag}: device {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}, torch {torch.__version__}, "
        f"jobs {jobs}")
    ext = None
    if any(parse_spec(j, a)[1] in EXTENDED for j in jobs):
        ext = load_extended(d, path)                                          # raises before any training
    log(f"  data {path} ({'extended pack, ' + ext['groups_file'] if ext else 'original arrays only'})")
    t_start, took, failed = time.time(), {}, []
    for spec in jobs:
        split, variant, seed, epochs, rtag = parse_spec(spec, a)
        est = took.get((variant, epochs), took.get((variant, DEFAULT_EPOCHS), 0.0) * epochs / DEFAULT_EPOCHS) * 1.1
        if (time.time() - T0 + est) / 3600 > a.max_hours:
            log(f"  {spec} skipped: would end after the {a.max_hours} h limit")
            failed.append(spec)
            continue
        rows = {p: d[f"{split}_{p}"] for p in ("train", "val", "test")}
        if a.smoke:
            rows = {"train": rows["train"][:30], "val": rows["val"][:10], "test": rows["test"][:6]}
        t_job = time.time()
        try:
            res, h = run_job(S, P, rows, split, variant, a, dev, log, seed=seed, epochs=epochs, tag=rtag, ext=ext)
        except Exception as exc:                                              # keep the other jobs of this worker
            import traceback
            log(f"  {spec} FAILED: {exc!r}\n{traceback.format_exc()}")
            failed.append(spec)
            continue
        took[variant, epochs] = time.time() - t_job
        res[f"{split}|ids"] = ids[rows["test"]]
        out.update(res); hist[spec] = h
        np.savez_compressed(os.path.join(out_dir(), f"part_{tag}.npz"), **out)
        json.dump(hist, open(os.path.join(out_dir(), f"history_{tag}.json"), "w"), indent=1)
    log(f"worker {tag} done in {(time.time() - t_start) / 3600:.2f} h; failed or skipped: {failed or 'none'}")
    if failed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--k", type=int, default=64, help="random origins per training run per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-origins", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--out", default=None, help="output directory instead of /kaggle/working or results/sim/phylstm")
    a = ap.parse_known_args(ARGV)[0]
    global OUT, SMOKE
    OUT, SMOKE = a.out, a.smoke                                               # the parsed values decide out_dir()
    if a.smoke:
        a.k, a.val_origins = 4, 4                                             # and 2 epochs per job (run_job)
    jobs = a.jobs or [f"{s}:{v}" for v in VARIANTS for s in SPLITS]
    keys = [(sp[0], sp[4]) for sp in (parse_spec(j, a) for j in jobs)]      # fails here on a bad spec
    if len(set(keys)) < len(keys):
        raise ValueError(f"two jobs write the same result keys: {jobs}")
    if a.worker is not None:
        worker(a, jobs, a.worker)
        return
    for f in glob.glob(os.path.join(out_dir(), "part_*.npz")) + glob.glob(os.path.join(out_dir(), "history_*.json"))             + glob.glob(os.path.join(out_dir(), "log_*.txt")):
        os.remove(f)                                                          # no stale results in the merge
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0 and os.path.isdir("/kaggle/working") and not a.smoke:     # a Kaggle session without the GPU it
        raise SystemExit("no GPU in this Kaggle session (account not phone-verified?): stopping instead of "
                         "training on the CPU")                     # asked for would burn 12 h on the CPU
    print(f"data {find_data(a.data)}; {n_gpu} GPU(s); {len(jobs)} jobs; args {ARGV}", flush=True)
    codes = []
    if n_gpu > 1 and "__file__" not in globals():
        print("WARNING: no __file__, cannot start one process per GPU; running everything on GPU 0", flush=True)
    if n_gpu > 1 and "__file__" in globals():
        procs = []
        for i in range(n_gpu):
            mine = jobs[i::n_gpu]
            if not mine:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", str(i), "--jobs", *mine,
                   "--epochs", str(a.epochs), "--k", str(a.k), "--batch", str(a.batch), "--val-origins", str(a.val_origins),
                   "--seed", str(a.seed), "--max-hours", str(a.max_hours)] + (["--data", a.data] if a.data else []) \
                + (["--out", a.out] if a.out else []) + (["--smoke"] if a.smoke else [])
            procs.append(subprocess.Popen(cmd, env=env))
        codes = [p.wait() for p in procs]
        print(f"workers exited with {codes}", flush=True)
    else:
        try:
            worker(a, jobs, "0")
        except SystemExit as exc:
            codes = [exc.code]
    merged, hist = {}, {}
    for f in sorted(glob.glob(os.path.join(out_dir(), "part_*.npz"))):
        merged.update(dict(np.load(f)))
    for f in sorted(glob.glob(os.path.join(out_dir(), "history_*.json"))):
        hist.update(json.load(open(f)))
    name = "phylstm_metrics_smoke.npz" if a.smoke else "phylstm_metrics.npz"
    np.savez_compressed(os.path.join(out_dir(), name), **merged)
    json.dump(hist, open(os.path.join(out_dir(), name.replace("metrics", "history").replace(".npz", ".json")), "w"), indent=1)
    missing = [j for j, (split, tag) in zip(jobs, keys) if f"{split}|{tag}|mse" not in merged]
    print(f"merged {len(jobs) - len(missing)}/{len(jobs)} jobs -> {os.path.join(out_dir(), name)}; missing: {missing or 'none'}",
          flush=True)
    if missing or any(codes):
        sys.exit(1)                                                           # the kernel then ends as ERROR, not COMPLETE


if __name__ == "__main__":
    main()
