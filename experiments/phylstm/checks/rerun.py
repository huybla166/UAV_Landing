import sys, json, types, numpy as np, torch
sys.path.insert(0, "D:/USV-landing-prediction/experiments/phylstm")
split, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.argv = ["x"]
import phylstm_kaggle as pk
a = types.SimpleNamespace(epochs=30, k=64, batch=256, val_origins=16, seed=seed, max_hours=11.0)
d = np.load("D:/USV-landing-prediction/data/sim/dataset_v2/kaggle_phylstm/usv_phylstm.npz")
S, P = d["S"], d["P"]
rows = {p: d[f"{split}_{p}"] for p in ("train", "val", "test")}
res, h = pk.run_job(S, P, rows, split, variant, a, "cuda", lambda m: print(m, flush=True))
tag = f"{split}_{variant}_s{seed}"
np.savez_compressed(f"rerun_{tag}.npz", **res)
json.dump(h, open(f"rerun_{tag}.json", "w"))
print("DONE", tag)
