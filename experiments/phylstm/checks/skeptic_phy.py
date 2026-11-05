import json, itertools
import numpy as np, pandas as pd

R = "D:/USV-landing-prediction/"
P = np.load(R + "results/sim/phylstm/phylstm_metrics.npz")
L = np.load(R + "results/sim/lstm_cv_metrics.npz")
H = json.load(open(R + "results/sim/phylstm/phylstm_history.json"))
meta = pd.read_csv(R + "data/sim/dataset_v2/cache_5Hz/meta.csv").set_index("id")
CV = [f"cv{k}" for k in range(6)]
SPL = CV + ["ood_Hs", "ood_beta"]
CH = ["heave", "roll", "pitch"]
rmse = lambda m: np.sqrt(m.mean(axis=(0, 1)))
np.set_printoptions(precision=4, suppress=True)

print("=== 1a. RMSE relative gain of phy over data (100*(data-phy)/data, %), per split; best and last epoch")
gain = {}
for s in SPL:
    g = 100 * (rmse(P[f"{s}|data|mse"]) - rmse(P[f"{s}|phy|mse"])) / rmse(P[f"{s}|data|mse"])
    gl = 100 * (rmse(P[f"{s}|data_last|mse"]) - rmse(P[f"{s}|phy_last|mse"])) / rmse(P[f"{s}|data_last|mse"])
    gain[s] = g
    print(f"{s:9s} best {g}  last {gl}")
G = np.array([gain[s] for s in CV]); print("6-fold mean", G.mean(0), "folds phy better:", (G > 0).sum(0), "/6")

print("\n=== 1b. paper-LSTM seed spread (same data, same arch, only seed differs): 100*(max-min)/mean RMSE and std/mean")
spread = {}
for s in SPL:
    r = np.array([rmse(L[f"{s}|seed{k}|mse"]) for k in range(3)])
    spread[s] = 100 * (r.max(0) - r.min(0)) / r.mean(0)
    pair = [100 * (r[i] - r[j]) / r[j] for i, j in itertools.permutations(range(3), 2)]
    print(f"{s:9s} range% {spread[s]}  sd% {100 * r.std(0, ddof=1) / r.mean(0)}  |phy-data|% {np.abs(gain[s])}")
S = np.array([spread[s] for s in SPL]); Gall = np.array([gain[s] for s in SPL])
print("mean seed range% over splits", S.mean(0), " mean |gain|%", np.abs(Gall).mean(0))
print("splits where |gain| > LSTM seed range:", (np.abs(Gall) > S).sum(0), "/8 per channel")

print("\n=== 1c. Same comparisons at lead 1 s and 2 s (index 4, 9)")
for li, t in [(4, 1), (9, 2), (24, 5), (49, 10)]:
    rows = []
    for s in SPL:
        d = np.sqrt(P[f"{s}|data|mse"][:, li].mean(0)); p = np.sqrt(P[f"{s}|phy|mse"][:, li].mean(0))
        r = np.array([np.sqrt(L[f"{s}|seed{k}|mse"][:, li].mean(0)) for k in range(3)])
        rows.append((100 * (d - p) / d, 100 * (r.max(0) - r.min(0)) / r.mean(0)))
    g = np.array([a for a, _ in rows]); sp = np.array([b for _, b in rows])
    print(f"lead {t:2d}s: gain% per split (rows=splits)\n{g}\n  mean gain {g.mean(0)}  mean LSTM seed range {sp.mean(0)}  splits gain>0 {(g > 0).sum(0)}/8  |gain|>range {(np.abs(g) > sp).sum(0)}/8")

print("\n=== 1d. run-level paired test is overconfident: fraction of runs where model A beats B (MSE over leads), and run-bootstrap z")
def paired(a, b):
    da = a.mean(1); db = b.mean(1)   # (R,3)
    d = db - da                       # >0: a better
    z = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))
    return (d > 0).mean(0), z
for s in ["cv0", "cv3", "ood_Hs"]:
    f, z = paired(P[f"{s}|phy|mse"], P[f"{s}|data|mse"])
    print(f"{s} phy vs data: frac runs phy better {f}, paired z {z}")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        f, z = paired(L[f"{s}|seed{i}|mse"], L[f"{s}|seed{j}|mse"])
        print(f"   LSTM seed{i} vs seed{j}: frac {f}, z {z}")

print("\n=== 1e/2. learning curves (normalized val MSE of 20-s forecast)")
for s in SPL:
    hp = H[f"{s}:phy"]; hd = H[f"{s}:data"]
    vp = np.array([e["val_mse"] for e in hp["history"]]); vd = np.array([e["val_mse"] for e in hd["history"]])
    n = min(len(vp), len(vd))
    below = (vp[:n] < vd[:n]).sum()
    sl_p = np.polyfit(np.arange(5), vp[-5:], 1)[0]; sl_d = np.polyfit(np.arange(5), vd[-5:], 1)[0]
    print(f"{s:9s} epochs {len(vp)}/{len(vd)} best_ep phy {hp['best_epoch']} data {hd['best_epoch']} | best val phy {hp['best_val']:.4f} data {hd['best_val']:.4f} "
          f"rel {100*(hd['best_val']-hp['best_val'])/hd['best_val']:.2f}% | phy<data at {below}/{n} epochs "
          f"(ep1-10 {(vp[:10] < vd[:10]).sum()}, 11-20 {(vp[10:20] < vd[10:20]).sum()}, 21-30 {(vp[20:n] < vd[20:n]).sum()}) | "
          f"slope last5 phy {sl_p:+.5f} data {sl_d:+.5f}/ep | val-noise sd last5 phy {np.std(vp[-5:]):.5f} data {np.std(vd[-5:]):.5f}")
    print(f"           val phy  {np.round(vp, 4)}\n           val data {np.round(vd, 4)}")
    print(f"           train Jd ep30 phy {hp['history'][-1]['Jd']:.4f} data {hd['history'][-1]['Jd']:.4f}  Je {hp['history'][-1].get('Je', 0):.4f} Jg {hp['history'][-1].get('Jg', 0):.4f}"
          f"  Jg ep1 {hp['history'][0]['Jg']:.4f}")

print("\n=== 3. condition breakdown, pooled runs. gain% = 100*(1 - sum mse_phy/sum mse_data) over all leads; also lead<=2s. "
      "noise = same for LSTM seed1 vs seed0 (and seed2 vs seed0)")
rows = []
for s in SPL:
    ids = P[f"{s}|ids"]
    for i, rid in enumerate(ids):
        rec = dict(split=s, id=rid, fold="ood" if s.startswith("ood") else "cv")
        for v in ["phy", "data"]:
            m = P[f"{s}|{v}|mse"][i]
            rec.update({f"{v}_all_{c}": m[:, k].mean() for k, c in enumerate(CH)})
            rec.update({f"{v}_s_{c}": m[:10, k].mean() for k, c in enumerate(CH)})
        for k_ in range(3):
            m = L[f"{s}|seed{k_}|mse"][i]
            rec.update({f"l{k_}_all_{c}": m[:, k].mean() for k, c in enumerate(CH)})
            rec.update({f"l{k_}_s_{c}": m[:10, k].mean() for k, c in enumerate(CH)})
        rows.append(rec)
df = pd.DataFrame(rows).join(meta[["vessel", "Hs", "beta_deg", "U", "regime"]], on="id")

def g(sub, a, b, w, c):
    return 100 * (1 - sub[f"{a}_{w}_{c}"].sum() / sub[f"{b}_{w}_{c}"].sum())

def table(by, sub=df, label=""):
    print(f"\n-- by {by} {label}")
    print(f"{by:>10s} {'n':>5s} | " + " | ".join(f"{c}: all  <=2s  wins  n01 n02" for c in CH))
    for key, grp in sub.groupby(by):
        parts = []
        for c in CH:
            wins = (grp[f"phy_all_{c}"] < grp[f"data_all_{c}"]).mean() * 100
            parts.append(f"{g(grp,'phy','data','all',c):+5.2f} {g(grp,'phy','data','s',c):+5.2f} {wins:4.0f}% "
                         f"{g(grp,'l1','l0','all',c):+5.2f} {g(grp,'l2','l0','all',c):+5.2f}")
        print(f"{str(key):>10s} {len(grp):5d} | " + " | ".join(parts))

table("split")
cvdf = df[df.fold == "cv"]
for by in ["U", "beta_deg", "regime", "vessel", "Hs"]:
    table(by, cvdf, "(6 CV folds pooled)")
for by in ["U", "regime"]:
    table(by, df[df.split == "ood_beta"], "(ood_beta)")
    table(by, df[df.split == "ood_Hs"], "(ood_Hs)")
cvdf2 = cvdf.copy(); cvdf2["U_beta"] = cvdf2.U.astype(str) + "_" + cvdf2.beta_deg.astype(str)
table("U_beta", cvdf2, "(CV)")
df.to_csv("D:/USV-landing-prediction/results/sim/phylstm/rerun/per_run.csv", index=False)

print("\n=== gain by lead (6-fold pooled, RMSE per lead), phy vs data and LSTM seed pairs")
for li in [0, 2, 4, 9, 14, 24, 49, 74, 99]:
    d = np.sqrt(np.concatenate([P[f"{s}|data|mse"][:, li] for s in CV]).mean(0))
    p = np.sqrt(np.concatenate([P[f"{s}|phy|mse"][:, li] for s in CV]).mean(0))
    l = [np.sqrt(np.concatenate([L[f"{s}|seed{k}|mse"][:, li] for s in CV]).mean(0)) for k in range(3)]
    l = np.array(l)
    print(f"lead {(li+1)*0.2:4.1f}s gain% {100*(d-p)/d}  LSTM seed range% {100*(l.max(0)-l.min(0))/l.mean(0)}")
