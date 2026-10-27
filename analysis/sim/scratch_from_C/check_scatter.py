import sys, os, math
sys.path.insert(0, r'D:\USV-landing-prediction\sim')
import torch, numpy as np, pandas as pd
import batch_sim_gpu as m
m.torch = torch
S = r'C:\Users\huybl\AppData\Local\Temp\claude\D--USV-landing-prediction\328cea5c-0bb6-4b38-a98b-b4e3be358938\scratchpad\gpu_test'
dm = pd.read_csv(r'D:\USV-landing-prediction\sim\design_matrix.csv')
for ves in ['otter_55kg', 'otter_80kg']:
    g = dm[(dm.id <= 40) & (dm.vessel == ves)]
    V = m.load_vessel(os.path.join(S, 'gpu_inputs', f'vessel_{ves}_dz0_0.mat'), 'cuda')
    W, ang, gg = m.load_waves(os.path.join(S, 'gpu_inputs'), ves)
    b = m.Batch(V, W, ang, gg, g, 'cuda')
    torch.manual_seed(0); worst = 0
    for k in range(200):
        B = len(g)
        t = float(600 * torch.rand(1)); pn = 100 * torch.randn(B, device='cuda', dtype=torch.float64)
        pe = 100 * torch.randn_like(pn); psi = 2 * math.pi * torch.rand_like(pn) - math.pi
        ta, za = b.wave(t, pn, pe, psi, 'gather'); tb, zb = b.wave(t, pn, pe, psi, 'scatter')
        worst = max(worst, float(((ta - tb).abs() / ta.abs().clamp(min=1e-9)).max()))
    print(f'{ves}: scatter vs gather max rel diff {worst:.2e}')
