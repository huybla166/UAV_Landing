function prep_gpu_inputs(designCsv, outDir, firstRow, lastRow, writeJson, skipDone)
% PREP_GPU_INPUTS  Everything sim_gpu/batch_sim.py needs to run design-matrix rows on the GPU,
% computed with exactly the code path of run_one.m (no simulation here):
%
%   <outDir>/gpu_inputs/vessel_<vessel>_dz<dz3>_<dz5>.mat   model constants (hydroVesselMemFast)
%   <outDir>/gpu_inputs/waves_<vessel>.mat                  wave model per seed (waveInitialization
%                                                           after rng(seed), as run_one.m)
%   <outDir>/runs/<name>_coef.json                          same export_coef.m output as run_one.m,
%                                                           plus run.engine = 'pytorch'
% writeJson = false skips the JSON files (inputs only); skipDone (default true) skips rows whose
% CSV already exists in <outDir>/runs.

if nargin < 3 || isempty(firstRow), firstRow = 1; end
if nargin < 5 || isempty(writeJson), writeJson = true; end
if nargin < 6 || isempty(skipDone), skipDone = true; end
D = read_design_local(designCsv);
if nargin < 4 || isempty(lastRow), lastRow = max(D.id); end
gpuDir = fullfile(outDir, 'gpu_inputs'); runDir = fullfile(outDir, 'runs');
if ~exist(gpuDir, 'dir'), mkdir(gpuDir); end
if ~exist(runDir, 'dir'), mkdir(runDir); end
T_final = 600; h = 0.025; h_log = 0.05;      % dataset_v2: integration step 0.025 s, CSV at 20 Hz

models = containers.Map();      % key vessel|Tp|dz -> struct(vessel, cfg, info)
waves = struct();               % per vessel: seeds and stacked wave arrays
savedVessel = containers.Map();
rows = find(D.id >= firstRow & D.id <= lastRow)';
t0 = tic;
for j = rows
    if skipDone && exist(fullfile(runDir, [D.name{j} '.csv']), 'file') == 2
        continue;                    % already simulated (Octave run_one.m): keep its JSON
    end
    vname = D.vessel{j}; matFile = [vname '.mat'];
    Tp = D.Tp(j); dz35 = [D.dzeta3(j), D.dzeta5(j)];
    key = sprintf('%s|%.4f|%g_%g', vname, Tp, dz35);
    if ~isKey(models, key)
        [vessel, cfg, info] = build_model(matFile, 2 * pi / Tp, false);
        cfg.damping.delta_zeta_345([1 3]) = dz35;       % as run_one.m with opts.delta_zeta35
        vessel = build_memory(vessel, cfg, false);
        vessel = prepare_fast(vessel);
        models(key) = struct('vessel', vessel, 'cfg', cfg, 'info', info);
    end
    Mdl = models(key);
    vessel = Mdl.vessel; cfg = Mdl.cfg; info = Mdl.info;
    k456 = [D.kappa4(j), D.kappa5(j), 0];
    vessel.kappa_4 = k456(1); vessel.kappa_5 = k456(2); vessel.kappa_6 = k456(3);
    cfg.damping.nonlinear_456 = k456;

    % ------------------------------------------------ model constants (once per vessel and damping)
    vkey = sprintf('%s_dz%g_%g', vname, dz35);
    if ~isKey(savedVessel, vkey)
        f = vessel.fast; mem = vessel.memory; hc = cfg.control.heading;
        S = struct('MRB', vessel.MRB, 'MAinf', vessel.MAinf, 'Minv_inf', vessel.Minv_inf, ...
            'Dv', vessel.Dv, 'Dref', vessel.Dref(:), 'G', vessel.G, 'Ar', mem.Ar, 'Br', mem.Br, ...
            'Cr', mem.Cr, 'Dr', mem.Dr, 'm', f.m, 'Ig', f.Ig, 'M11', f.M11, 'M12', f.M12, ...
            'M21', f.M21, 'M22', f.M22, 'xL', f.xL, 'kcf', f.kcf, 'Xuu_const', f.Xuu_const, ...
            'L', f.L, 'hc_r_max', hc.r_max, 'hc_wn', hc.wn, 'hc_zeta', hc.zeta);
        save('-v7', fullfile(gpuDir, ['vessel_' vkey '.mat']), '-struct', 'S');
        savedVessel(vkey) = true;
    end

    % ------------------------------------------------ wave model for this seed (as run_one.m)
    env = cfg.environment;
    env.Hs = D.Hs(j); env.w0 = 2 * pi / Tp; env.beta_wave = D.beta_deg(j) * pi / 180;
    env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100;
    env.numDirections = 24; env.Vc = 0; env.betaVc = 0;
    w = vessel.forceRAO.w(:); maxFreq = max(w(w < 10)); env.maxFreq = maxFreq;
    rng(D.seed(j));
    [env, wm] = waveInitialization(vessel, env);
    if ~isfield(waves, vname), waves.(vname) = struct('seeds', [], 'W', {{}}); end
    if ~any(waves.(vname).seeds == D.seed(j))
        waves.(vname).seeds(end+1) = D.seed(j);
        waves.(vname).W{end+1} = struct('Omega', wm.Omega(:), 'mu', wm.mu(:), 'Amp', wm.Amp, ...
            'phases', wm.randomPhases, 'RAOstack', cat(3, wm.RAO_re{:}, wm.RAO_im{:}), ...
            'raoAngles', wm.raoAngles(:), 'g', wm.g);
    end

    % ------------------------------------------------ coefficient JSON (same fields as run_one.m)
    if writeJson
        name = D.name{j};
        csvFile = fullfile(runDir, [name '.csv']);
        jsonFile = fullfile(runDir, [name '_coef.json']);
        run = struct('matFile', matFile, 'matPath', info.matPath, 'Hs', D.Hs(j), 'Tp', Tp, ...
            'beta_deg', D.beta_deg(j), 'U_ref', D.U(j), 'seed', D.seed(j), 'T_final', T_final, 'h', h, ...
            'RAO_dt', h, 'psi_ref_deg', 0, 'csv', csvFile, ...
            'aperiodic_T_requested', info.aperiodic_T_requested, 'aperiodic_T_used', info.aperiodic_T_used, ...
            'delta_zeta', info.delta_zeta, 'kappa_456', cfg.damping.nonlinear_456, 'control', 'autopilot', ...
            'memory', true, 'delta_zeta_345_used', cfg.damping.delta_zeta_345, 'maxFreq', maxFreq, ...
            'wavePhase', 'position', 'waveImpl', 'fast', 'waveFrame', 'seakeeping', 'T_lp', 20, ...
            'waveHold', 'rk4', 'h_log', h_log, 'memory_fit_version', vessel.memory.fitVersion, ...
            'engine', 'pytorch (sim/batch_sim_gpu.py)', ...
            'created', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        export_coef(jsonFile, vessel, env, run);
    end
    if mod(numel(find(rows <= j)), 100) == 0
        fprintf('  %d / %d rows, %.0f s\n', numel(find(rows <= j)), numel(rows), toc(t0));
    end
end

% ------------------------------------------------ stacked wave arrays per vessel
if isempty(fieldnames(waves)), fprintf('prep rows %d..%d: nothing to do\n', firstRow, lastRow); return; end
vn = fieldnames(waves);
for i = 1:numel(vn)
    Wc = waves.(vn{i}).W; n = numel(Wc);
    S = struct();
    S.seeds = waves.(vn{i}).seeds(:);
    S.Omega = cell2mat(cellfun(@(s) s.Omega', Wc(:), 'UniformOutput', false));     % n x Nf
    S.mu = cell2mat(cellfun(@(s) s.mu', Wc(:), 'UniformOutput', false));           % n x Nd
    S.Amp = cat(3, cellfun(@(s) s.Amp, Wc, 'UniformOutput', false){:});            % Nf x Nd x n
    S.phases = cat(3, cellfun(@(s) s.phases, Wc, 'UniformOutput', false){:});      % Nf x Nd x n
    S.RAOstack = cat(4, cellfun(@(s) s.RAOstack, Wc, 'UniformOutput', false){:});  % Nf x (Nh+1) x 12 x n
    S.raoAngles = Wc{1}.raoAngles; S.g = Wc{1}.g;
    for k = 2:n
        if ~isequal(Wc{k}.raoAngles, S.raoAngles), error('prep_gpu_inputs:angles', 'RAO heading grids differ'); end
    end
    out = fullfile(gpuDir, sprintf('waves_%s_rows%d_%d.mat', vn{i}, firstRow, lastRow));
    save('-v7', out, '-struct', 'S');
    fprintf('wrote %s (%d seeds)\n', out, n);
end
fprintf('prep rows %d..%d done in %.0f s\n', firstRow, lastRow, toc(t0));
end

function D = read_design_local(f)
fid = fopen(f, 'r');
fgetl(fid);
C = textscan(fid, '%f %s %f %f %f %f %f %f %f %f %f %f %s %s', 'Delimiter', ',');
fclose(fid);
D = struct('id', C{1}, 'vessel', {C{2}}, 'Hs', C{3}, 'Tp', C{4}, 'beta_deg', C{5}, 'U', C{6}, ...
           'kappa4', C{7}, 'kappa5', C{8}, 'rep', C{9}, 'seed', C{10}, 'dzeta3', C{11}, 'dzeta5', C{12}, ...
           'split', {C{13}}, 'name', {C{14}});
end
