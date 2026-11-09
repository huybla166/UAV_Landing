"""Pack dataset_v2 for the PhyLSTM run on Kaggle (one compressed file + metadata).

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/pack_kaggle.py

Writes data/sim/dataset_v2/kaggle_phylstm/:
  usv_phylstm.npz        - S (3150, 2826, 9) float32 per run at 5 Hz, rows in cache meta order:
                             [D, phi, theta]                       eta, m and rad (sim_common.CSV_COLS)
                             [D_dot, phi_dot, theta_dot]           eta_dot from the kinematics eta_dot = J(eta) nu
                                                                   (5 Hz central differences miss it by 18-27 %)
                             [f_heave, f_roll, f_pitch]            tau_w3..5 / diag(MRB + MA_inf)_33,44,55 of the vessel,
                                                                   the mass-normalised wave excitation of PhyLSTM eq. (2)
                           P (3150, 4) float32 run parameters [is_80kg, U, kappa4/5, has_visc]
                           ids (3150,), and for each split of evaluation protocol (a) (cv0..cv5, ood_Hs, ood_beta)
                           '<split>_train' / '_val' / '_test' row indices (sim_common.cv_split / ood_split, QC failures
                           removed), so the Kaggle script uses exactly the splits of run_sim_lstm_cv.py
  dataset-metadata.json  - private Kaggle dataset <owner>/usv-dataset-v2-phylstm

  D:/tools/venv-gpu/Scripts/python.exe experiments/phylstm/pack_kaggle.py --v2 [owner]

Writes the EXTENDED pack data/sim/dataset_v2/kaggle_phylstm_v2/usv_phylstm.npz (the old pack is read, not changed):
  every key of the old pack, identical (asserted: S, P, ids and all split indices), plus per run at 5 Hz
  nu (3150, 2826, 6) float32       body velocities u, v, w, p, q, r (m/s, rad/s), cache columns
  psi (3150, 2826) float32         yaw angle (rad), not unwrapped
  group (3150,) int16              row of cummins_groups.npz (experiments/phylstm/cummins_coef.py, run it first)
  sea (3150, 4) float32            nominal Hs (m), Tp (s), beta_deg, U (m/s): labels for a training-only auxiliary task
  sea_real (3150, 2) float32       realised Hs_real (m) and U_mean (m/s) from the manifest
  sea_cols, sea_real_cols          column names
The directory also holds cummins_groups.npz, so one Kaggle dataset <owner>/usv-dataset-v2-phylstm-v2 carries both.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_common import CSV_COLS, K_FOLDS, SIM, cv_split, load, ood_split  # noqa: E402

OUT = os.path.join(SIM, "kaggle_phylstm")
SPLITS = [f"cv{k}" for k in range(K_FOLDS)] + ["ood_Hs", "ood_beta"]
ARGS = [a for a in sys.argv[1:] if a != "--v2"]
OWNER = ARGS[0] if ARGS else "wanghui1101"


def inertia(vessel):
    """diag(MRB + MA_inf) for heave, roll, pitch, identical in every coef.json of a vessel."""
    import pandas as pd
    man = pd.read_csv(os.path.join(SIM, "manifest.csv"))
    name = man[man.vessel == vessel].name.iloc[0]
    c = json.load(open(os.path.join(SIM, "runs", name + "_coef.json")))
    M = np.array(c["MRB"]) + np.array(c["MA_inf"])
    return np.diag(M)[2:5]


def features(meta, X, Mii):
    """S (n, T, 9) float32 = eta | eta_dot | f for the runs of meta (rows of X); elementwise, so chunks give the same
    numbers as the whole set."""
    c = {k: i for i, k in enumerate(CSV_COLS)}
    X = np.asarray(X, np.float64)
    phi, th = X[..., c["phi"]], X[..., c["theta"]]
    u, v, w, p, q, r = (X[..., c[k]] for k in ("u", "v", "w", "p", "q", "r"))
    eta = np.stack([X[..., c["D"]], phi, th], axis=-1)
    eta_dot = np.stack([-np.sin(th) * u + np.cos(th) * np.sin(phi) * v + np.cos(th) * np.cos(phi) * w,
                        p + np.sin(phi) * np.tan(th) * q + np.cos(phi) * np.tan(th) * r,
                        np.cos(phi) * q - np.sin(phi) * r], axis=-1)
    Minv = np.stack([1.0 / Mii[v_] for v_ in meta.vessel])                     # (N, 3)
    f = np.stack([X[..., c["tau_w3"]], X[..., c["tau_w4"]], X[..., c["tau_w5"]]], axis=-1) * Minv[:, None, :]
    return np.concatenate([eta, eta_dot, f], axis=-1).astype(np.float32)


def main():
    meta, X = load()
    Mii = {v_: inertia(v_) for v_ in meta.vessel.unique()}
    print({k: np.round(m, 3).tolist() for k, m in Mii.items()})
    S = features(meta, X, Mii)
    P = np.stack([(meta.vessel == "otter_80kg").to_numpy(float), meta.U.to_numpy(float), meta.kappa4.to_numpy(float),
                  (meta.dzeta3 > 0).to_numpy(float)], axis=1).astype(np.float32)
    ok = (meta.status == "ok").to_numpy()
    out = dict(S=S, P=P, ids=meta.id.to_numpy())
    for s in SPLITS:
        sp = cv_split(meta, int(s[2:])) if s.startswith("cv") else ood_split(meta, s[4:])
        for part, idx in sp.items():
            out[f"{s}_{part}"] = idx[ok[idx]]
        print(s, {p_: len(out[f"{s}_{p_}"]) for p_ in sp})
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "usv_phylstm.npz"), **out)
    json.dump({"title": "USV dataset v2 PhyLSTM", "id": f"{OWNER}/usv-dataset-v2-phylstm",
               "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(OUT, "dataset-metadata.json"), "w"), indent=1)
    print(f"S {S.shape}, {os.path.getsize(os.path.join(OUT, 'usv_phylstm.npz')) / 1e6:.0f} MB -> {OUT}")


def main_v2(chunk=150):
    """Extended pack (see the module docstring), built in chunks of runs (< 2 GB RAM)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cummins_coef import group_table
    out_dir = os.path.join(SIM, "kaggle_phylstm_v2")
    old = np.load(os.path.join(OUT, "usv_phylstm.npz"))
    meta, X = load()
    c = {k: i for i, k in enumerate(CSV_COLS)}
    Mii = {v_: inertia(v_) for v_ in meta.vessel.unique()}
    N, T = X.shape[:2]
    S = np.empty((N, T, 9), np.float32); nu = np.empty((N, T, 6), np.float32); psi = np.empty((N, T), np.float32)
    nu_cols = [c[k] for k in ("u", "v", "w", "p", "q", "r")]
    for b in range(0, N, chunk):
        Xb = np.asarray(X[b:b + chunk])
        S[b:b + chunk] = features(meta.iloc[b:b + chunk], Xb, Mii)
        nu[b:b + chunk] = Xb[..., nu_cols]
        psi[b:b + chunk] = Xb[..., c["psi"]]
    out = {k: old[k] for k in old.files}
    assert np.array_equal(S, out["S"]), "S differs from the old pack"
    P = np.stack([(meta.vessel == "otter_80kg").to_numpy(float), meta.U.to_numpy(float), meta.kappa4.to_numpy(float),
                  (meta.dzeta3 > 0).to_numpy(float)], axis=1).astype(np.float32)
    assert np.array_equal(P, out["P"]) and np.array_equal(meta.id.to_numpy(), out["ids"]), "P / ids differ"
    ok = (meta.status == "ok").to_numpy()
    for s in SPLITS:                                                         # same splits as the old pack
        sp = cv_split(meta, int(s[2:])) if s.startswith("cv") else ood_split(meta, s[4:])
        for part, idx in sp.items():
            assert np.array_equal(idx[ok[idx]], out[f"{s}_{part}"]), f"{s}_{part} differs"
    _, group = group_table(meta)
    cg = np.load(os.path.join(out_dir, "cummins_groups.npz"))
    assert np.array_equal(cg["run_group"], group) and np.array_equal(cg["ids"], out["ids"]), "rerun cummins_coef.py"
    out.update(nu=nu, psi=psi, group=group,
               sea=meta[["Hs", "Tp", "beta_deg", "U"]].to_numpy(np.float32),
               sea_cols=np.array(["Hs_m", "Tp_s", "beta_deg", "U_mps"]),
               sea_real=meta[["Hs_real", "U_mean"]].to_numpy(np.float32), sea_real_cols=np.array(["Hs_real_m", "U_mean_mps"]))
    del S
    path = os.path.join(out_dir, "usv_phylstm.npz")
    np.savez_compressed(path, **out)
    json.dump({"title": "USV dataset v2 PhyLSTM v2", "id": f"{OWNER}/usv-dataset-v2-phylstm-v2",
               "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(out_dir, "dataset-metadata.json"), "w"), indent=1)
    print(f"{len(out)} arrays, old keys identical; {os.path.getsize(path) / 1e6:.0f} MB -> {path}; "
          f"cummins_groups.npz {os.path.getsize(os.path.join(out_dir, 'cummins_groups.npz')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main_v2() if "--v2" in sys.argv[1:] else main()
