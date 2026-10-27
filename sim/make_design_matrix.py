"""Build the simulation design matrix for the full dataset (Giai đoạn 2).

Writes sim/design_matrix.csv, one row per 600 s run of sim/run_one.m. Factors:
  vessel   otter_55kg, otter_80kg
  Hs       0.10 0.20 0.30 0.40 0.50 m            (Q5: Hs <= 0.5 m for a 2 m USV)
  Tp       3 levels spanning the JONSWAP validity band 1.25 <= w0*sqrt(Hs) <= 1.75
           (levels placed at 1.26 .. 1.74 so 2-decimal rounding of Tp stays inside the band)
  beta     0 45 90 135 180 deg (0 = following, 180 = head sea)
  U        0 1 2 m/s
  kappa    (k4, k5) = (0, 0) or (5, 5)            (Q4: nonlinear roll/pitch damping)
  rep      1 2 3                                  (wave realizations)
  visc     Q3 variant subset: extra viscous damping ratio dzeta (heave, pitch) = (0.2, 0.3) on
           top of BEM radiation damping, so the totals approach MSS otter.m (zeta 0.3 heave,
           0.4 pitch; CRAFT/USV/models/otter.m lines 214-216). Only rep 1, kappa (0, 0):
           every variant run has a baseline twin with the same sea.

Wave seed depends only on (Hs, Tp, beta, rep): the same sea is replayed for both vessels, all
speeds and both damping levels, so those factors can be compared pairwise.
Rows are shuffled with a fixed generator, so any prefix of the file is a spread-out sample.
Split (fixed before any training): test_ood_Hs (Hs = 0.40), test_ood_beta (beta = 45),
test_id (rep = 3), else train. Vessel transfer (55 -> 80 kg) is chosen at experiment time.
"""
import itertools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VESSELS = ["otter_55kg", "otter_80kg"]
HS = [0.10, 0.20, 0.30, 0.40, 0.50]
BETAS = [0, 45, 90, 135, 180]
SPEEDS = [0, 1, 2]
KAPPAS = [(0, 0), (5, 5)]
REPS = [1, 2, 3]
VISC = (0.2, 0.3)


def tp_levels(hs):
    w_lo, w_hi = 1.26 / np.sqrt(hs), 1.74 / np.sqrt(hs)
    return [round(float(t), 2) for t in np.linspace(2 * np.pi / w_hi, 2 * np.pi / w_lo, 3)]


rows = []
sea_ids = {}
for vessel, hs, beta, u, (k4, k5), rep in itertools.product(VESSELS, HS, BETAS, SPEEDS, KAPPAS, REPS):
    for tp in tp_levels(hs):
        sea = (hs, tp, beta, rep)
        sea_ids.setdefault(sea, len(sea_ids) + 1)
        seed = 1000 + sea_ids[sea]
        split = ("test_ood_Hs" if hs == 0.40 else "test_ood_beta" if beta == 45
                 else "test_id" if rep == 3 else "train")
        name = f"{vessel}_Hs{hs:.2f}_Tp{tp:.2f}_b{beta:03d}_U{u}_k{k4}_r{rep}"
        base = dict(vessel=vessel, Hs=hs, Tp=tp, beta_deg=beta, U=u, kappa4=k4, kappa5=k5,
                    rep=rep, seed=seed, split=split)
        rows.append(dict(base, dzeta3=0.0, dzeta5=0.0, name=name))
        if rep == 1 and k4 == 0:
            rows.append(dict(base, dzeta3=VISC[0], dzeta5=VISC[1], name=name + "_visc"))

df = pd.DataFrame(rows).sample(frac=1.0, random_state=20261026).reset_index(drop=True)
df.insert(0, "id", np.arange(1, len(df) + 1))
df = df[["id", "vessel", "Hs", "Tp", "beta_deg", "U", "kappa4", "kappa5", "rep", "seed", "dzeta3", "dzeta5", "split", "name"]]
out = os.path.join(HERE, "design_matrix.csv")
df.to_csv(out, index=False)

steep = (df.Hs / (9.81 * df.Tp ** 2 / (2 * np.pi))).max()
print(f"wrote {out}: {len(df)} runs, {df.seed.nunique()} distinct seas, max steepness Hs/lambda = 1/{1/steep:.0f}")
print(df.split.value_counts().to_string())
print(f"visc variant rows: {(df.dzeta3 > 0).sum()}")
print("\nfirst 20 rows:")
print(df.head(20)[["id", "vessel", "Hs", "Tp", "beta_deg", "U", "kappa4", "rep", "split"]].to_string(index=False))
