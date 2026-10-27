import sys, time, os
sys.path.insert(0, r'D:\USV-landing-prediction\sim')
import torch, numpy as np, pandas as pd
import batch_sim_gpu as m
m.torch = torch
S = r'C:\Users\huybl\AppData\Local\Temp\claude\D--USV-landing-prediction\328cea5c-0bb6-4b38-a98b-b4e3be358938\scratchpad\gpu_test'
dm = pd.read_csv(r'D:\USV-landing-prediction\sim\design_matrix.csv')
g = dm[(dm.id <= 40) & (dm.vessel == 'otter_55kg') & (dm.dzeta3 == 0)]
dev = sys.argv[1]
V = m.load_vessel(os.path.join(S, 'gpu_inputs', 'vessel_otter_55kg_dz0_0.mat'), dev)
W, ang, gg = m.load_waves(os.path.join(S, 'gpu_inputs'), 'otter_55kg')
for B in [len(g), 800]:
    rows = pd.concat([g] * (B // len(g) + 1)).iloc[:B]
    m.T_FINAL = 10.0
    b = m.Batch(V, W, ang, gg, rows, dev)
    t0 = time.time(); b.run(); 
    if dev == 'cuda': torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{dev} B={B}: {dt/201*1e3:.1f} ms/step -> full 12001 steps {dt/201*12001/60:.1f} min', flush=True)
