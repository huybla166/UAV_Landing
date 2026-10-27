"""Compare runs produced by sim/batch_sim_gpu.py with the same design rows from sim/run_one.m (Octave).

  python sim/compare_gpu_octave.py <gpu_runs_dir> <octave_runs_dir>

For every run name present in both: max |gpu - octave| per column, divided by the column's std
(Octave), and the coefficient JSONs compared value by value (except run.created / run.engine /
run.csv). The CSVs are written with 7 significant digits, so ~1e-6 relative is the floor.
"""
import json
import os
import sys

import numpy as np
import pandas as pd


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}.{k}" if pre else k))
    else:
        out[pre] = d
    return out


def main(gdir, odir):
    names = sorted(f[:-4] for f in os.listdir(gdir) if f.endswith(".csv") and os.path.exists(os.path.join(odir, f)))
    rows, json_bad = [], []
    for n in names:
        g = pd.read_csv(os.path.join(gdir, n + ".csv"))
        o = pd.read_csv(os.path.join(odir, n + ".csv"))
        if g.shape != o.shape:
            rows.append(dict(name=n, shape=f"{g.shape} vs {o.shape}"))
            continue
        sd = o.std().replace(0, 1.0)
        rel = ((g - o).abs().max() / sd)
        r = dict(name=n, worst_col=rel.idxmax(), worst_rel=rel.max())
        for c in ["zeta", "D", "phi", "theta", "psi", "u", "tauN"]:
            r[c] = rel[c]
        rows.append(r)
        gj = os.path.join(gdir, n + "_coef.json"); oj = os.path.join(odir, n + "_coef.json")
        if os.path.exists(gj) and os.path.exists(oj):
            a, b = flat(json.load(open(gj))), flat(json.load(open(oj)))
            skip = {"run.created", "run.engine", "run.csv"}
            for k in sorted((set(a) | set(b)) - skip):
                va, vb = a.get(k), b.get(k)
                if isinstance(va, (int, float, list)) and isinstance(vb, (int, float, list)):
                    try:
                        if not np.allclose(np.array(va, dtype=float), np.array(vb, dtype=float), rtol=1e-12, atol=1e-14, equal_nan=True):
                            json_bad.append((n, k))
                    except (ValueError, TypeError):
                        if va != vb:
                            json_bad.append((n, k))
                elif va != vb:
                    json_bad.append((n, k))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print(f"\n{len(names)} runs compared; worst max|diff|/std over all columns: {df.worst_rel.max():.2e}")
    print(f"JSON fields that differ: {len(json_bad)}", json_bad[:10])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
