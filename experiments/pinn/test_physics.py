"""Tests for experiments/pinn/physics.py on a simulated run of data/sim/dataset_v1 (step 4.1).

  python -m pytest experiments/pinn/test_physics.py -q

"Done" criterion of step 4.1 (Ke_hoach_tung_buoc.md): the residual on the true simulated trajectory
is below 1 % of the wave-load amplitude (RMS residual / std of tau_wave, per DOF).
"""
import glob
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics import (VesselModel, ddt, interior, kinematic_residual, load_run, nu_from_eta,  # noqa: E402
                     relative_residual)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "sim", "dataset_v1", "runs")
N_ROWS = 12002          # header + 600 s at 20 Hz
SKIP = 600              # 30 s start-up transient
HEAVE, ROLL, PITCH, YAW = 2, 3, 4, 5


def complete_run(pattern):
    for csv in sorted(glob.glob(os.path.join(RUNS, pattern + ".csv"))):
        js = csv[:-4] + "_coef.json"
        if os.path.exists(js) and sum(1 for _ in open(csv)) == N_ROWS:
            return csv, js
    return None


@pytest.fixture(scope="module")
def run():
    # forward speed and quadratic roll/pitch damping exercise most terms of the model
    found = complete_run("otter_55kg_*_U1_k5_*") or complete_run("*")
    if found is None:
        pytest.skip("no complete run in data/sim/dataset_v1")
    torch.set_num_threads(2)
    return load_run(found[0]), VesselModel.from_json(found[1])


def test_full_residual_below_1_percent(run):
    d, m = run
    r = m.residual(d["nu"], d["eta"], d["tau_ext"], d["h"], "full")
    rel = relative_residual(r, d["tau_wave"], skip=SKIP)
    assert torch.all(rel[[HEAVE, ROLL, PITCH]] < 0.01), rel
    assert rel[YAW] < 0.02, rel


def test_eta_only_path_below_1_percent(run):
    """The PINN predicts eta only: nu from eta rates and the known surge/sway speeds."""
    d, m = run
    eta, h = d["eta"], d["h"]
    nu_i = interior(d["nu"])
    nu_rec = nu_from_eta(interior(eta), ddt(eta, h), nu_i[:, 0], nu_i[:, 1])
    r = m.residual(nu_rec, interior(eta), interior(d["tau_ext"]), h, "full")
    rel = relative_residual(r, interior(d["tau_wave"]), skip=SKIP)
    assert torch.all(rel[[HEAVE, ROLL, PITCH]] < 0.01), rel


def test_kinematics(run):
    d, _ = run
    k = kinematic_residual(d["eta"], d["nu"], d["h"])[SKIP:]
    scale = ddt(d["eta"], d["h"])[SKIP:].std(0)
    assert torch.all(k.pow(2).mean(0).sqrt()[2:] / scale[2:] < 0.01)


def test_memory_warmup_60s(run):
    """60 s of history (the forecast context) initialises the memory states to < 1 % of std(mu)."""
    d, m = run
    nu, h = d["nu"], d["h"]
    mu = m.memory_force(nu, m.memory_states(nu, h))
    k0, w, H = 6000, 1200, 400
    mu_w = m.memory_force(nu[k0 - w:k0 + H], m.memory_states(nu[k0 - w:k0 + H], h))[w:]
    err = (mu_w - mu[k0:k0 + H]).pow(2).mean(0).sqrt() / mu[SKIP:].std(0)
    assert torch.all(err < 0.01), err


def test_residual_is_differentiable(run):
    d, m = run
    sl = slice(6000, 6400)
    nu = d["nu"][sl].clone().requires_grad_(True)
    r = m.residual(nu, d["eta"][sl], d["tau_ext"][sl], d["h"], "full")
    r.pow(2).mean().backward()
    assert nu.grad is not None and torch.isfinite(nu.grad).all()


def test_linear_level_needs_the_quadratic_damping(run):
    """The fixture run has kappa_4 = kappa_5 = 5: a plain linearization drops kappa |p| p (zero slope)."""
    d, m = run
    U = m.coef["run"]["U_ref"]
    plain = relative_residual(m.residual(d["nu"], d["eta"], d["tau_ext"], d["h"], "linear", U=U), d["tau_wave"], skip=SKIP)
    quad = relative_residual(m.residual(d["nu"], d["eta"], d["tau_ext"], d["h"], "linear", U=U, quadratic=True),
                             d["tau_wave"], skip=SKIP)
    assert quad[ROLL] < plain[ROLL] and quad[PITCH] < plain[PITCH], (plain, quad)


def test_linearization_matches_full_for_small_perturbations(run):
    _, m = run
    U = 1.0
    nu0, F0, K = m.linearize(U)
    eta0 = torch.zeros(6, dtype=m.dtype)
    dnu = 1e-4 * torch.tensor([1.0, -0.5, 0.7, 0.3, -0.2, 0.4], dtype=m.dtype)
    exact = m.lhs_static(nu0 + dnu, eta0) - F0
    assert torch.allclose(exact, K @ dnu, rtol=1e-2, atol=1e-6)
