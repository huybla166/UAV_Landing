"""Merge the per-worker QC files of a run_batch dataset into manifest.csv and summarise.

  python sim/batch_report.py data/sim/dataset_v0
"""
import glob
import os
import sys

import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "data/sim/dataset_v0"
design = pd.read_csv(os.path.join(out, "design_matrix.csv"))
qc_files = sorted(glob.glob(os.path.join(out, "qc", "qc_w*.csv")))
qc = (pd.concat([pd.read_csv(f).assign(engine="pytorch" if "gpu" in os.path.basename(f) else "octave")
                 for f in qc_files], ignore_index=True) if qc_files else pd.DataFrame(columns=["id"]))
qc = qc.drop_duplicates("id", keep="last")
man = design.merge(qc.drop(columns=["name"], errors="ignore"), on="id", how="left")
man["status"] = man["status"].fillna("not_run")
man.to_csv(os.path.join(out, "manifest.csv"), index=False)

done = man[man.status != "not_run"]
print(f"{out}: {len(done)} of {len(man)} runs attempted")
print(done.status.value_counts().to_string())
if "engine" in done:
    print(done.engine.value_counts().to_string())
if len(done):
    print(f"wall time per run: median {done.wall_s.median():.0f} s, max {done.wall_s.max():.0f} s")
    bad = done[done.status != "ok"]
    if len(bad):
        print("\nnot ok:")
        print(bad[["id", "name", "status", "message"]].to_string(index=False))
    cols = ["id", "vessel", "Hs", "Tp", "beta_deg", "U", "kappa4", "Hs_real", "U_mean",
            "std_heave_m", "std_roll_deg", "std_pitch_deg", "max_roll_deg", "status"]
    if len(done) <= 50:
        print("\n" + done.sort_values("id")[cols].round(3).to_string(index=False))
runs = glob.glob(os.path.join(out, "runs", "*.csv"))
size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out, "runs", "*"))) / 1e6
print(f"\nfiles: {len(runs)} run CSVs, {size:.0f} MB in {os.path.join(out, 'runs')}")
