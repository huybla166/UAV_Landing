import numpy as np
R="D:/USV-landing-prediction/"
P=np.load(R+"results/sim/phylstm/phylstm_metrics.npz"); L=np.load(R+"results/sim/lstm_cv_metrics.npz")
d=np.load(R+"data/sim/dataset_v2/kaggle_phylstm/usv_phylstm.npz"); S=d["S"]; ids=d["ids"]
CV=[f"cv{k}" for k in range(6)]
o=np.arange(300, S.shape[1]-100+1, 5)
for name, spl in [("6-fold", CV), ("ood_Hs", ["ood_Hs"]), ("ood_beta", ["ood_beta"])]:
    rows=np.concatenate([d[f"{s}_test"] for s in spl])
    eta=S[rows][..., :3].astype(np.float64); eta[...,1:]=np.rad2deg(eta[...,1:])
    print(name)
    for li in [0,1,2,4,9]:
        h=li+1
        pers=np.sqrt(((eta[:, o-1+h]-eta[:, o-1])**2).mean(axis=(0,1)))     # last observed sample is o-1
        # linear extrapolation from last two samples
        lin=np.sqrt(((eta[:, o-1+h]-(eta[:, o-1]+h*(eta[:, o-1]-eta[:, o-2])))**2).mean(axis=(0,1)))
        f=lambda k: np.sqrt(np.concatenate([P[f"{s}|{k}|mse"][:,li] for s in spl]).mean(0))
        lstm=np.mean([np.sqrt(np.concatenate([L[f"{s}|seed{k}|mse"][:,li] for s in spl]).mean(0)) for k in range(3)],0)
        print(f"  lead {h*0.2:.1f}s persistence {pers} linear-extrap {lin} | data {f('data')} phy {f('phy')} LSTM {lstm}")
