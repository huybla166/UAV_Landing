% CHECK_FAST_WAVES  waveLoads with the stacked RAO table (waveModel.RAOstack) must equal the
% per-DOF loop; reports the speed-up.
mss_setup;
[v, cfg] = build_model('otter_55kg.mat', 2*pi/1.9, false);
env = cfg.environment; env.Hs = 0.3; env.w0 = 2*pi/2.5; env.beta_wave = 2.4;
env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100; env.numDirections = 24;
env.maxFreq = 9.5; env.Vc = 0; env.betaVc = 0;
rng(7); [~, wm] = waveInitialization(v, env);
wf = wm; wf.RAOstack = cat(3, wm.RAO_re{:}, wm.RAO_im{:});
rng(1); worst = 0;
for n = 1:500
    t = 600 * rand; pos = 50 * randn(2, 1); psi = 2 * pi * rand;
    [a, za] = waveLoads(t, wm, pos, 1, psi, env.beta_wave, 'position');
    [b, zb] = waveLoads(t, wf, pos, 1, psi, env.beta_wave, 'position');
    worst = max([worst; abs(a - b) ./ max(abs(a), 1e-9); abs(za - zb)]);
end
N = 400;
tt = tic; for i = 1:N, a = waveLoads(i*0.05, wm, [1; 2], 1, 0.01, 2.4, 'position'); end; ta = toc(tt) / N;
tt = tic; for i = 1:N, b = waveLoads(i*0.05, wf, [1; 2], 1, 0.01, 2.4, 'position'); end; tb = toc(tt) / N;
fprintf('waveLoads stacked vs loop: max rel diff %.2e | %.3f ms -> %.3f ms per call\n', worst, 1e3*ta, 1e3*tb);
