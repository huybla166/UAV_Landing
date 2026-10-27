"""SHA-256 checksums of a simulated dataset, and the check of a reproduction against them.

  python sim/checksums.py make  data/sim/dataset_v2 [list.txt]
  python sim/checksums.py check sim/reproduce_v2/checksums_sha256.txt <runs_dir> [<original_runs_dir>]

`make` hashes runs/*.csv, design_matrix.csv, gpu_inputs/*.mat and inputs/** of a dataset folder and writes the list
(default <dataset>/checksums_sha256.txt; paths relative to the dataset folder, `sha256sum -c` format). The JSON files
carry a creation time, so they are compared by value with sim/compare_gpu_octave.py instead.
`check` reports, for every CSV in <runs_dir> that the list knows: identical bytes or not. With the original runs
folder it also prints, for files that differ, the largest |difference| / std of the original per column and the first
time at which a motion column differs by more than 1 % of its std (the divergence time of chaotic runs).
"""
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd

MOTION = ["D", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make(ds, out=None):
    files = sorted(glob.glob(os.path.join(ds, "runs", "*.csv")))
    files += [os.path.join(ds, "design_matrix.csv")]
    files += sorted(glob.glob(os.path.join(ds, "gpu_inputs", "*.mat")))
    files += sorted(glob.glob(os.path.join(ds, "inputs", "**", "*.*"), recursive=True))
    out = out or os.path.join(ds, "checksums_sha256.txt")
    with open(out, "w", newline="\n") as f:
        for p in files:
            f.write(f"{sha256(p)}  {os.path.relpath(p, ds).replace(os.sep, '/')}\n")
    print(f"{len(files)} files -> {out}")


def check(list_file, runs, original=None):
    ref = {}
    for line in open(list_file):
        digest, rel = line.rstrip("\n").split("  ", 1)
        ref[rel] = digest
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "*.csv"))):
        name = os.path.basename(p)
        if "runs/" + name not in ref:
            continue
        row = dict(run=name[:-4], identical=sha256(p) == ref["runs/" + name])
        if not row["identical"] and original and os.path.exists(os.path.join(original, name)):
            a, b = pd.read_csv(os.path.join(original, name)), pd.read_csv(p)
            rel_diff = (b - a).abs() / a.std().replace(0, 1.0)
            over = (rel_diff[MOTION] > 0.01).any(axis=1).to_numpy()
            row.update(max_rel=float(rel_diff.max().max()), worst=rel_diff.max().idxmax(),
                       diverges_at_s=float(a.t[over.argmax()]) if over.any() else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    n_id = int(df.identical.sum()) if len(df) else 0
    print(f"\n{n_id} of {len(df)} CSV files byte-identical to the list {list_file}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
