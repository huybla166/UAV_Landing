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
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_common import CSV_COLS, K_FOLDS, SIM, cv_split, load, ood_split  # noqa: E402

OUT = os.path.join(SIM, "kaggle_phylstm")
SPLITS = [f"cv{k}" for k in range(K_FOLDS)] + ["ood_Hs", "ood_beta"]
OWNER = sys.argv[1] if len(sys.argv) > 1 else "wanghui1101"


def inertia(vessel):
    """diag(MRB + MA_inf) for heave, roll, pitch, identical in every coef.json of a vessel."""
    import pandas as pd
    man = pd.read_csv(os.path.join(SIM, "manifest.csv"))
    name = man[man.vessel == vessel].name.iloc[0]
    c = json.load(open(os.path.join(SIM, "runs", name + "_coef.json")))
    M = np.array(c["MRB"]) + np.array(c["MA_inf"])
    return np.diag(M)[2:5]


def main():
    meta, X = load()
    c = {k: i for i, k in enumerate(CSV_COLS)}
    X = np.asarray(X, np.float64)
    phi, th = X[..., c["phi"]], X[..., c["theta"]]
    u, v, w, p, q, r = (X[..., c[k]] for k in ("u", "v", "w", "p", "q", "r"))
    eta = np.stack([X[..., c["D"]], phi, th], axis=-1)
    eta_dot = np.stack([-np.sin(th) * u + np.cos(th) * np.sin(phi) * v + np.cos(th) * np.cos(phi) * w,
                        p + np.sin(phi) * np.tan(th) * q + np.cos(phi) * np.tan(th) * r,
                        np.cos(phi) * q - np.sin(phi) * r], axis=-1)
    Mii = {v_: inertia(v_) for v_ in meta.vessel.unique()}
    print({k: np.round(m, 3).tolist() for k, m in Mii.items()})
    Minv = np.stack([1.0 / Mii[v_] for v_ in meta.vessel])                     # (N, 3)
    f = np.stack([X[..., c["tau_w3"]], X[..., c["tau_w4"]], X[..., c["tau_w5"]]], axis=-1) * Minv[:, None, :]
    S = np.concatenate([eta, eta_dot, f], axis=-1).astype(np.float32)
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


if __name__ == "__main__":
    main()
