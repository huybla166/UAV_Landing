% CHECK_FAST_DYNAMICS  hydroVesselMemFast must equal hydroVesselMem (random states, both vessels,
% with/without current), and report the speed-up.
mss_setup;
rng(3);
for vf = {'otter_55kg.mat', 'otter_80kg.mat'}
    [v, cfg] = build_model(vf{1}, 2*pi/1.9, false);
    v.kappa_4 = 5; v.kappa_5 = 5; v.kappa_6 = 0.5;
    v = build_memory(v, cfg, false);
    v = prepare_fast(v);
    worst = 0;
    for n = 1:300
        x = [randn(6,1) .* [1.5; 0.3; 0.3; 0.5; 0.5; 0.3]; randn(6,1) .* [5; 5; 0.2; 0.3; 0.2; 1]; 0.1 * randn(v.memory.n, 1)];
        tau = randn(6, 1) * 20;
        if mod(n, 2), Vc = 0.4; else, Vc = 0; end
        a = hydroVesselMem(x, tau, v, Vc, 0.7);
        b = hydroVesselMemFast(x, tau, v, Vc, 0.7);
        worst = max(worst, max(abs(a - b) ./ max(abs(a), 1e-6)));
    end
    N = 400; x = [1; 0.01; 0.02; 0.05; 0.04; 0.01; zeros(6,1); zeros(v.memory.n,1)];
    t = tic; for i = 1:N, a = hydroVesselMem(x, zeros(6,1), v, 0, 0); end; ta = toc(t) / N;
    t = tic; for i = 1:N, b = hydroVesselMemFast(x, zeros(6,1), v, 0, 0); end; tb = toc(t) / N;
    fprintf('%s: max rel diff %.2e over 300 random states | %.3f ms -> %.3f ms per call (x%.1f)\n', ...
        vf{1}, worst, 1e3*ta, 1e3*tb, ta/tb);
end
