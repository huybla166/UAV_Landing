"""Add the operating-regime columns of analysis/sim/verify_dataset_v1.py (run_stats.csv) to a dataset manifest.

  DATASET=dataset_v2 python analysis/sim/add_regime.py
Columns added to data/sim/<DATASET>/manifest.csv: std_psi_deg, max_psi_deg, max_du, U_over_c, regime
('violent' = yaw std > 5 deg or max |u - U| > 1.5 m/s, else 'regular').
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.environ.get("DATASET", "dataset_v1")
V = os.path.join(ROOT, "analysis", "sim", "V" if DATASET == "dataset_v1" else "V_" + DATASET)
man_path = os.path.join(ROOT, "data", "sim", DATASET, "manifest.csv")
man = pd.read_csv(man_path)
st = pd.read_csv(os.path.join(V, "run_stats.csv"), index_col="id")
cols = ["std_psi_deg", "max_psi_deg", "max_du", "U_over_c"]
man = man.drop(columns=[c for c in cols + ["regime"] if c in man.columns])
man = man.join(st[cols], on="id")
man["regime"] = st.loc[man.id, "violent"].map({True: "violent", False: "regular"}).values
man.to_csv(man_path, index=False)
print(man.regime.value_counts().to_string())
print(man.groupby(["U", "beta_deg"]).regime.apply(lambda s: (s == "violent").mean()).unstack().round(2).to_string())
