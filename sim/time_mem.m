% TIME_MEM  Cost per call of hydroVesselMem and its parts (Octave), to plan the batch.
mss_setup;
[v, cfg] = build_model('otter_55kg.mat', 2*pi/1.9, false);
v.kappa_4 = 5; v.kappa_5 = 5; v.kappa_6 = 0;
v = build_memory(v, cfg, false);
x = [1; 0.01; 0.02; 0.05; 0.04; 0.01; zeros(6,1); zeros(v.memory.n, 1)];
N = 400;
t = tic; for i = 1:N, xd = hydroVesselMem(x, zeros(6,1), v, 0, 0); end; fprintf('hydroVesselMem  %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, [~, C] = rbody(v.main.m, v.main.k44, v.main.k55, v.main.k66, x(4:6), [0 0 0]'); end; fprintf('  rbody         %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, C = m2c(v.MAinf, x(1:6)); end; fprintf('  m2c           %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, X = XuuITTC(1, 1025, 2, 1.08, 0.165, []); end; fprintf('  XuuITTC       %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, c = crossFlowDrag(v.main.Lpp, v.main.B, v.main.T, x(1:6), 'Hoerner'); end; fprintf('  crossFlowDrag %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, J = eulerang(0.05, 0.04, 0.01); end; fprintf('  eulerang      %.3f ms\n', 1e3*toc(t)/N);
t = tic; for i = 1:N, y = v.memory.Ar * x(13:end) + v.memory.Br * x(1:6); end; fprintf('  memory matvec %.3f ms\n', 1e3*toc(t)/N);
wm = regular_wave_model(v, 3, 0.1);
cfg2 = hydroVesselConfig('otter_55kg.mat', true); env = cfg2.environment; env.Hs = 0.2; env.w0 = 3.3; env.beta_wave = pi/2;
env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100; env.numDirections = 24; env.maxFreq = 9.5;
rng(1); [~, wmod] = waveInitialization(v, env);
t = tic; for i = 1:N, tw = waveLoads(i*0.05, wmod, [1; 2], 1, 0.01, pi/2, 'position'); end; fprintf('waveLoads       %.3f ms\n', 1e3*toc(t)/N);
