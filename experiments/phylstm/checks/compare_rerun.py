import numpy as np, json, os
P = np.load("D:/USV-landing-prediction/results/sim/phylstm/phylstm_metrics.npz")
M = {"K_phy0": P["cv0|phy|mse"], "K_data0": P["cv0|data|mse"]}
for t, f in [("L_data0", "rerun_cv0_data_s0.npz"), ("L_data1", "rerun_cv0_data_s1.npz"), ("L_phy1", "rerun_cv0_phy_s1.npz")]:
    if os.path.exists(f):
        M[t] = np.load(f)[f"cv0|{t.split('_')[1][:-1]}|mse"]
        h = json.load(open(f.replace(".npz", ".json"))); print(t, "best_epoch", h["best_epoch"], "best_val", round(h["best_val"], 4))
def r(m, li=None): return np.sqrt(m.mean(axis=(0, 1))) if li is None else np.sqrt(m[:, li].mean(0))
for k, m in M.items(): print(f"{k:8s} RMSE all {r(m)}  0.2s {r(m,0)}  1s {r(m,4)}  2s {r(m,9)}")
pairs = [("K_phy0", "K_data0", "physics effect, seed 0 (Kaggle T4)"), ("L_phy1", "L_data1", "physics effect, seed 1 (local 3060)"),
         ("L_data0", "K_data0", "noise: same init, other GPU/sampling stream"), ("L_data1", "L_data0", "noise: data seed1 vs seed0 (local)")]
for a, b, lab in pairs:
    if a in M and b in M:
        print(f"{lab}: 100*(B-A)/B  all {100*(r(M[b])-r(M[a]))/r(M[b])}  0.2s {100*(r(M[b],0)-r(M[a],0))/r(M[b],0)}  1s {100*(r(M[b],4)-r(M[a],4))/r(M[b],4)}  2s {100*(r(M[b],9)-r(M[a],9))/r(M[b],9)}  5s {100*(r(M[b],24)-r(M[a],24))/r(M[b],24)}")
