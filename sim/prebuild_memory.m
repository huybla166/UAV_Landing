% PREBUILD_MEMORY  Fit and cache the fluid-memory model for every vessel / viscous-damping setting
% in the design matrix, before parallel workers start (they then only load the caches).
mss_setup;
settings = {[0 0], [0.2 0.3]};      % (dzeta heave, dzeta pitch): baseline, Q3 variant
for vf = {'otter_55kg.mat', 'otter_80kg.mat'}
    for s = 1:numel(settings)
        [v, cfg] = build_model(vf{1}, 2*pi/2, false);
        cfg.damping.delta_zeta_345([1 3]) = settings{s};
        t0 = tic;
        v = build_memory(v, cfg, true);
        fprintf('%s dz35 = [%g %g]: %d states, Dv diag = [%s], %.0f s\n', vf{1}, settings{s}, v.memory.n, ...
            num2str(diag(v.Dv)', '%.1f '), toc(t0));
    end
end
