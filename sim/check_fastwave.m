% CHECK_FASTWAVE  Verify sim/waveLoads.m against MSS waveForceRAO and quantify the
% effect of (a) the phase formulation and (b) the default 5 rad/s spectrum cut.
%   octave-cli --eval "cd('D:/USV-landing-prediction/sim'); check_fastwave"
mss_setup;
here = fileparts(mfilename('fullpath'));
outDir = fullfile(here, '..', 'data', 'sim', 'check_fastwave');
if ~exist(outDir, 'dir'), mkdir(outDir); end
vf = 'otter_55kg.mat';
S = load(resolveHydroVesselFile(vf), 'vessel'); vessel = S.vessel;
fw = vessel.forceRAO.w(:); wmax = max(fw(fw < 10));
cfg = hydroVesselConfig(vf, true);
env = cfg.environment; env.Hs = 0.15; env.w0 = 2*pi/1.9; env.beta_wave = pi/2;
env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100;
env.numDirections = 24; env.maxFreq = wmax;
rng(7); [env, wm] = waveInitialization(vessel, env);

% ---------------------------------------------------------------- (1) function level
fprintf('(1) Function level, 500 random (t, U, psi, beta):\n');
rng(11); e1 = 0; e2 = 0; e3 = 0;
for n = 1:500
    t = 600 * rand; U = 1.0 * rand; psi = 2*pi*rand - pi; bw = 2*pi*rand;
    [a, za] = waveForceRAO(t, wm, U, psi, bw);
    [b, zb] = waveLoads(t, wm, [0; 0], U, psi, bw, 'mss');
    e1 = max(e1, max(abs(a - b) ./ max(abs(a), 1)));
    e1 = max(e1, abs(za - zb));
    % constant U, psi: position phase along the straight track equals the MSS phase
    NE = U * t * [cos(psi); sin(psi)];
    [c, zc] = waveLoads(t, wm, NE, U, psi, bw, 'position');
    e2 = max(e2, max(abs(a - c) ./ max(abs(a), 1)));
    e3 = max(e3, abs(za - zc));
end
fprintf('    waveLoads(''mss'') vs MSS waveForceRAO      : max rel. diff %.2e\n', e1);
fprintf('    waveLoads(''position'') on straight track     : max rel. diff forces %.2e, zeta %.2e m\n', e2, e3);
fprintf('    (U <= 1 m/s and w <= %.2f rad/s keep W_e > 0, so the two phase forms must agree)\n', wmax);

% ---------------------------------------------------------------- (2) system level
args = {vf, 0.15, 1.9, 90, 1.0, 7, 600, 0.05, outDir};
runs = { 'ref_msslib',   struct('waveImpl','msslib','wavePhase','mss',      'maxFreq', wmax); ...
         'fast_mss',     struct('waveImpl','fast',  'wavePhase','mss',      'maxFreq', wmax); ...
         'fast_position',struct('waveImpl','fast',  'wavePhase','position', 'maxFreq', wmax); ...
         'fast_pos_cut5',struct('waveImpl','fast',  'wavePhase','position', 'maxFreq', 5) };
X = cell(1, 4); T = zeros(1, 4);
for r = 1:4
    o = runs{r, 2}; o.name = runs{r, 1}; o.verbose = false;
    t0 = tic; run_one(args{:}, o); T(r) = toc(t0);
    X{r} = dlmread(fullfile(outDir, [o.name '.csv']), ',', 1, 0);
end
fprintf('\n(2) 600 s runs, Otter 55 kg, Hs 0.15 m, Tp 1.9 s, beam sea, U 1 m/s\n');
fprintf('    wall time [s]: MSS %.0f | fast/mss %.0f | fast/position %.0f  -> speed-up x%.1f\n', T(1), T(2), T(3), T(1)/T(2));
k0 = find(X{1}(:,1) >= 30, 1);
cols = [2 11 12 13 14]; names = {'zeta','heave','roll','pitch','yaw'};
fprintf('    fast/mss vs MSS, max|diff|/std over the run:');
for j = 1:numel(cols)
    a = X{1}(k0:end, cols(j)); b = X{2}(k0:end, cols(j));
    fprintf('  %s %.1e', names{j}, max(abs(a - b)) / std(a));
end
fprintf('\n');

% ---------------------------------------------------------------- (3) physics checks
fs = 20; segs = {[30 300], [300 600]};
fprintf('\n(3) Mean frequency of zeta (rad/s), first vs second half of the run:\n');
for r = [1 3]
    z = X{r}(:, 2); t = X{r}(:, 1);
    wm_seg = zeros(1, 2);
    for q = 1:2
        m = t >= segs{q}(1) & t < segs{q}(2);
        zz = z(m) - mean(z(m)); n = numel(zz);
        Z = abs(fft(zz)).^2; f = (0:n-1)' * (2*pi*fs/n); h2 = 2:floor(n/2);
        wm_seg(q) = sum(f(h2) .* Z(h2)) / sum(Z(h2));
    end
    fprintf('    %-14s 30-300 s: %.2f   300-600 s: %.2f\n', runs{r,1}, wm_seg(1), wm_seg(2));
end
fprintf('    (spectral peak of the sea: w0 = %.2f rad/s; the encountered mean should not drift with time)\n', env.w0);

fprintf('\n(4) Spectrum cut at 5 rad/s vs full (position phase), std after 30 s:\n');
sc = [1 180/pi 180/pi]; nm = {'heave [m]','roll [deg]','pitch [deg]'}; cc = [11 12 13];
for j = 1:3
    a = std(X{3}(k0:end, cc(j))) * sc(j); b = std(X{4}(k0:end, cc(j))) * sc(j);
    fprintf('    %-12s full %.4f  cut %.4f  ratio %.2f\n', nm{j}, a, b, a/b);
end
fprintf('    Hs realized  full %.3f  cut %.3f\n', 4*std(X{3}(k0:end,2)), 4*std(X{4}(k0:end,2)));
fprintf('    psi range (deg), position run: %.1f .. %.1f\n', min(X{3}(:,14))*180/pi, max(X{3}(:,14))*180/pi);
