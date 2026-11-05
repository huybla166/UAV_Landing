import numpy as np, json, csv
from scipy import stats
R='D:/USV-landing-prediction/'
P=np.load(R+'results/sim/phylstm/phylstm_metrics.npz'); L=np.load(R+'results/sim/lstm_cv_metrics.npz')
meta={int(r['id']):r for r in csv.DictReader(open(R+'data/sim/dataset_v2/cache_5Hz/meta.csv'))}
CH=['heave','roll','pitch']; CV=[f'cv{k}' for k in range(6)]; SP=CV+['ood_Hs','ood_beta']
MET=['RMSE','RMSE1','RMSE2','RMSE5','MAE']
def m(mse,mae):
    return {'RMSE':np.sqrt(mse.mean((0,1))),'RMSE1':np.sqrt(mse[:,4].mean(0)),'RMSE2':np.sqrt(mse[:,9].mean(0)),
            'RMSE5':np.sqrt(mse[:,24].mean(0)),'MAE':mae.mean((0,1))}
f=lambda x:f'{x:.3g}'
V={}
for s in SP:
    assert (P[s+'|ids']==L[s+'|ids']).all()
    for v in ['phy','data','phy_last','data_last']: V[s,v]=m(P[f'{s}|{v}|mse'],P[f'{s}|{v}|mae'])
    ls=[m(L[f'{s}|seed{d}|mse'],L[f'{s}|seed{d}|mae']) for d in range(3)]
    for d in range(3): V[s,f'l{d}']=ls[d]
    V[s,'lstm']={k:np.mean([x[k] for x in ls],0) for k in MET}
print('## 1-2. relative differences per split (%), 6-fold mean, std, t')
for a,b in [('phy','data'),('data','lstm'),('phy','lstm'),('phy_last','data_last')]:
    print(f'\n### ({a}-{b})/{b} in %')
    print('| ch | metric | '+' | '.join(SP)+' | #CV a<b | CV mean | CV std | t |')
    for c,ch in enumerate(CH):
        for k in MET:
            r={s:100*(V[s,a][k][c]-V[s,b][k][c])/V[s,b][k][c] for s in SP}
            cv=np.array([r[s] for s in CV]); mu,sd=cv.mean(),cv.std(ddof=1)
            print(f'| {ch} | {k} | '+' | '.join(f(r[s]) for s in SP)+f' | {(cv<0).sum()}/6 | {f(mu)} | {f(sd)} | {f(mu/(sd/np.sqrt(6)))} |')
print('\n## 3. run-level phy vs data (per-run MSE). frac = fraction phy better; p = Wilcoxon; seed-level = per-wave-seed averages')
print('| split | ch | lead | nRuns | frac runs | p runs | sign p | nSeeds | frac seeds | p seeds |')
for s in SP:
    ids=P[s+'|ids']; sd=np.array([int(meta[int(i)]['seed']) for i in ids]); us=np.unique(sd)
    for c,ch in enumerate(CH):
        for lead,sl in [('all',slice(None)),('1s',4),('2s',9)]:
            a=P[f'{s}|phy|mse'][:,sl,c]; b=P[f'{s}|data|mse'][:,sl,c]
            if a.ndim==2: a,b=a.mean(1),b.mean(1)
            d=a-b; fr=(d<0).mean(); p=stats.wilcoxon(a,b).pvalue; ps=stats.binomtest(int((d<0).sum()),len(d)).pvalue
            ds=np.array([d[sd==u].mean() for u in us])
            pss=stats.wilcoxon(ds).pvalue if len(us)>=6 else np.nan
            print(f'| {s} | {ch} | {lead} | {len(d)} | {f(fr)} | {p:.2g} | {ps:.2g} | {len(us)} | {f((ds<0).mean())} | {pss:.2g} |')
# pooled seeds across all 6 CV folds
print('\n### pooled over 6 CV folds, per-seed averages of (phy-data) per-run MSE')
print('| ch | lead | nSeeds | frac phy better | Wilcoxon p | sign p |')
for c,ch in enumerate(CH):
    for lead,sl in [('all',slice(None)),('1s',4),('2s',9)]:
        D=[]
        for s in CV:
            ids=P[s+'|ids']; sd=np.array([int(meta[int(i)]['seed']) for i in ids])
            a=P[f'{s}|phy|mse'][:,sl,c]; b=P[f'{s}|data|mse'][:,sl,c]
            if a.ndim==2: a,b=a.mean(1),b.mean(1)
            # relative per seed to scale-normalise
            D+=[(a[sd==u].mean()-b[sd==u].mean())/b[sd==u].mean() for u in np.unique(sd)]
        D=np.array(D); print(f'| {ch} | {lead} | {len(D)} | {f((D<0).mean())} | {stats.wilcoxon(D).pvalue:.2g} | {stats.binomtest(int((D<0).sum()),len(D)).pvalue:.2g} |')
print('\n## 4. LSTM 3-seed spread (max-min)/mean in %, vs |phy-data|/data in %')
print('| split | ch | RMSE spread | |phy-data| RMSE | RMSE1 spread | |phy-data| RMSE1 | pairwise seed |d| RMSE1 mean |')
for s in SP:
    for c,ch in enumerate(CH):
        out=[]
        for k in ['RMSE','RMSE1']:
            x=np.array([V[s,f'l{d}'][k][c] for d in range(3)])
            out+= [100*(x.max()-x.min())/x.mean(), 100*abs(V[s,'phy'][k][c]-V[s,'data'][k][c])/V[s,'data'][k][c]]
        x=np.array([V[s,f'l{d}']['RMSE1'][c] for d in range(3)])
        pw=np.mean([abs(x[i]-x[j])/x.mean() for i,j in [(0,1),(0,2),(1,2)]])*100
        print(f'| {s} | {ch} | '+' | '.join(f(o) for o in out)+f' | {f(pw)} |')
print('\n### CV-mean of LSTM spread vs phy-data and data-lstm (%)')
for c,ch in enumerate(CH):
    for k in MET:
        sp=np.mean([100*(max(V[s,f'l{d}'][k][c] for d in range(3))-min(V[s,f'l{d}'][k][c] for d in range(3)))/V[s,'lstm'][k][c] for s in CV])
        # implied single-seed std from 3 seeds, pooled over folds
        sdv=np.sqrt(np.mean([np.var([V[s,f'l{d}'][k][c] for d in range(3)],ddof=1)/V[s,'lstm'][k][c]**2 for s in CV]))*100
        pd=np.mean([100*(V[s,'phy'][k][c]-V[s,'data'][k][c])/V[s,'data'][k][c] for s in CV])
        dl=np.mean([100*(V[s,'data'][k][c]-V[s,'lstm'][k][c])/V[s,'lstm'][k][c] for s in CV])
        # noise of a phy-data difference of two single seeds: sqrt2*sd per fold; over 6 folds mean: sqrt2*sd/sqrt6
        print(f'{ch} {k}: LSTM spread {f(sp)}%, single-seed sd {f(sdv)}%, expected sd of 1-seed diff {f(np.sqrt(2)*sdv)}% (6-fold mean {f(np.sqrt(2)*sdv/np.sqrt(6))}%), phy-data {f(pd)}%, data-lstm {f(dl)}%')
h=json.load(open(R+'results/sim/phylstm/phylstm_history.json'))
print('\nbest epochs:',{k:(v['best_epoch'],len(v['history'])) for k,v in h.items()})
