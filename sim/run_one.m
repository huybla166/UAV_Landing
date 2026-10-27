function out = run_one(matFile, Hs, Tp, beta_deg, U_ref, seed, T_final, h, outDir, opts)
% RUN_ONE  Headless wave -> wave force -> 6-DOF USV motion simulation (MSS 2.0).
%
%   out = run_one(matFile, Hs, Tp, beta_deg, U_ref, seed, T_final, h, outDir)
%
%   matFile  : vessel data file, e.g. 'testShip.mat' or 'otter_55kg.mat'
%              (resolved with resolveHydroVesselFile, i.e. from MSS/HYDRO/vessels_*)
%   Hs       : significant wave height (m)
%   Tp       : peak period (s)            -> w0 = 2*pi/Tp
%   beta_deg : wave direction (deg), 0 = following sea, 180 = head sea
%   U_ref    : target forward speed (m/s), held by a PI surge controller
%   seed     : random seed for the wave phases
%   T_final  : simulation length (s)
%   h        : time step (s), 0.05 = 20 Hz
%   outDir   : output directory (created if missing)
%   opts     : optional struct: spectrumType ('JONSWAP'), spreadingFlag (1),
%              numFreqIntervals (100), numDirections (24), psi_ref_deg (0),
%              name (auto), verbose (true)
%
%   Writes  <outDir>/<name>.csv        22 columns, header in first line:
%             t, zeta, tau_w1..tau_w6, N, E, D, phi, theta, psi,
%             u, v, w, p, q, r, tauX, tauN
%           <outDir>/<name>_coef.json  model matrices and run metadata
%
%   Replicates the loop of SIMhydroVessel.m without the GUI:
%     hydroVesselConfig -> load vessel -> waveInitialization
%     -> computeManeuveringModel -> [headingAutopilot + waveForceRAO + rk4(@hydroVessel)]
%
%   Units: positions m, angles rad in the CSV (convert to deg downstream).

if nargin < 10, opts = struct(); end
opts = setDefault(opts, 'spectrumType', 'JONSWAP');
opts = setDefault(opts, 'spreadingFlag', 1);
opts = setDefault(opts, 'numFreqIntervals', 100);
opts = setDefault(opts, 'numDirections', 24);
opts = setDefault(opts, 'psi_ref_deg', 0);
opts = setDefault(opts, 'verbose', true);
opts = setDefault(opts, 'RAO_dt', h);          % update wave forces every step
opts = setDefault(opts, 'wavePhase', 'position'); % 'position' (physical) or 'mss' (Omega_e*t, as MSS)
opts = setDefault(opts, 'waveImpl', 'fast');       % 'fast' = sim/waveLoads.m, 'msslib' = MSS waveForceRAO (needs wavePhase 'mss')
opts = setDefault(opts, 'maxFreq', []);        % [] = highest finite RAO frequency of the vessel
opts = setDefault(opts, 'control', 'autopilot');  % 'autopilot' (speed PI + heading PID) or 'none' (tau_ctrl = 0)
opts = setDefault(opts, 'kappa456', []);       % [] = from hydroVesselConfig; else [k4 k5 k6] nonlinear damping
opts = setDefault(opts, 'waveFrame', 'seakeeping'); % 'seakeeping': wave loads at the low-pass (mean) position/heading
opts = setDefault(opts, 'T_lp', 20);           % low-pass time constant (s) of the seakeeping frame
opts = setDefault(opts, 'memory', true);       % true: Cummins fluid memory (hydroVesselMem); false: MSS constant matrices
opts = setDefault(opts, 'delta_zeta35', []);   % [] = config (0 0); else extra viscous damping ratio in heave, pitch
opts = setDefault(opts, 'dynImpl', 'fast');    % 'fast' = hydroVesselMemFast.m, 'reference' = hydroVesselMem.m (same model)
opts = setDefault(opts, 'h_log', h);           % log period (multiple of h): h = 0.025 with h_log = 0.05 keeps the 20 Hz CSV
opts = setDefault(opts, 'waveHold', 'rk4');    % 'rk4': wave load at t, t+h/2, t+h inside each RK4 step (sim/rk4u.m)
                                               % 'zoh': held over the step as in MSS (delays the load by h/2)
[~, base, ~] = fileparts(matFile);
opts = setDefault(opts, 'name', sprintf('%s_Hs%.2f_Tp%.1f_b%03d_U%.1f_s%d', ...
    base, Hs, Tp, round(beta_deg), U_ref, seed));

if ~exist(outDir, 'dir'), mkdir(outDir); end
clear headingAutopilot;                        % persistent integrator/reference states

% ---------------------------------------------------------------- vessel + constant M, D, G at w0
[vessel, cfg, info] = build_model(matFile, 2 * pi / Tp, opts.verbose);
matPath = info.matPath; aper = info.aperiodic_T_requested; aperUsed = info.aperiodic_T_used; dz = info.delta_zeta;
if ~isempty(opts.kappa456)
    vessel.kappa_4 = opts.kappa456(1); vessel.kappa_5 = opts.kappa456(2); vessel.kappa_6 = opts.kappa456(3);
    cfg.damping.nonlinear_456 = opts.kappa456;
end
if opts.memory
    if ~isempty(opts.delta_zeta35)      % Q3 variant: extra viscous heave/pitch damping
        cfg.damping.delta_zeta_345([1 3]) = opts.delta_zeta35;
    end
    vessel = build_memory(vessel, cfg, opts.verbose);   % cached per vessel and damping setting
    vessel = prepare_fast(vessel);
    dyn = @hydroVesselMemFast;     % = hydroVesselMem.m, inlined (sim/check_fast_dynamics.m)
    if strcmp(opts.dynImpl, 'reference'), dyn = @hydroVesselMem; end
    nMem = vessel.memory.n;
    Msys = vessel.MRB + vessel.MAinf;
else
    dyn = @hydroVessel;
    nMem = 0;
    Msys = vessel.M;
end

% ---------------------------------------------------------------- waves
env = cfg.environment;
env.Hs = Hs;
env.w0 = 2 * pi / Tp;
env.beta_wave = beta_deg * pi / 180;
env.spectrumType = opts.spectrumType;
env.spreadingFlag = opts.spreadingFlag;
env.numFreqIntervals = opts.numFreqIntervals;
env.numDirections = opts.numDirections;
env.Vc = 0; env.betaVc = 0;
% MSS truncates the wave spectrum at 5 rad/s by default. A 2 m USV has heave/roll/pitch
% resonances near 6 rad/s, so keep the spectrum up to the highest finite RAO frequency.
if isempty(opts.maxFreq)
    w = vessel.forceRAO.w(:);
    opts.maxFreq = max(w(w < 10));
end
env.maxFreq = opts.maxFreq;

setSeed(seed);
[env, waveModel] = waveInitialization(vessel, env);   % spectrum, random phases, RAO tables
waveModel.RAOstack = cat(3, waveModel.RAO_re{:}, waveModel.RAO_im{:});   % fast path in waveLoads.m

% ---------------------------------------------------------------- surge force to hold U_ref
x = [U_ref; 0; 0; 0; 0; 0; zeros(6, 1); zeros(nMem, 1)];
xd0 = dyn(x, zeros(6, 1), vessel, 0, 0);
tau_ss = -Msys * xd0(1:6);                     % generalized force that cancels the drift at U_ref
tauX0 = tau_ss(1);
Kp_u = Msys(1, 1) / 10;                        % 10 s speed-loop time constant
Ki_u = Kp_u / 10;                              % trial batch: Kp/50 left the mean speed 10-15 % low at U = 2
u_int = 0;

% ---------------------------------------------------------------- loop
t = 0:h:T_final;
n = numel(t);
kLog = max(1, round(opts.h_log / h));
if abs(kLog * h - opts.h_log) > 1e-12, error('run_one:h_log', 'h_log must be a multiple of h'); end
nLog = floor((n - 1) / kLog) + 1;
log = zeros(nLog, 22);
tau_wave1 = zeros(6, 1); zeta = 0; nextRAO = 0;
if strcmp(opts.waveImpl, 'msslib') && ~strcmp(opts.wavePhase, 'mss')
    error('run_one:waveImpl', 'waveImpl ''msslib'' (MSS waveForceRAO) only supports wavePhase ''mss''.');
end
useRK4u = strcmp(opts.waveHold, 'rk4');
if useRK4u && (~strcmp(opts.waveImpl, 'fast') || ~strcmp(opts.waveFrame, 'seakeeping') || abs(opts.RAO_dt - h) > 1e-12)
    error('run_one:waveHold', 'waveHold ''rk4'' needs waveImpl ''fast'', waveFrame ''seakeeping'' and RAO_dt = h.');
end
if useRK4u
    [tau_wave1, zeta] = waveLoads(0, waveModel, x(7:8), U_ref, x(12), env.beta_wave, opts.wavePhase);
end
psi_ref = opts.psi_ref_deg * pi / 180;
hc = cfg.control.heading;
% Seakeeping frame (Fossen's unified model): 1st-order wave loads are evaluated at the
% mean (low-pass) track and heading, not at the wave-frequency oscillating position.
% Using the instantaneous position lets O(a^2) drift Doppler-shift the encounter
% frequency (see analysis/sim/check_wave_phase.md). At constant U, psi it equals MSS.
vbar = U_ref * [cos(x(12)); sin(x(12))];   % low-pass NED velocity
posS = x(7:8);  psiS = x(12);

for i = 1:n
    nu = x(1:6); eta = x(7:12);

    % speed hold (surge) + heading autopilot (yaw)
    if strcmp(opts.control, 'none')
        tauX = 0; tauN = 0;
    else
        e_u = U_ref - nu(1);
        u_int = u_int + h * e_u;
        tauX = tauX0 + Kp_u * e_u + Ki_u * u_int;
        tauN = headingAutopilot(eta(6), nu(6), psi_ref, hc.r_max, Msys(6, 6), hc.wn, hc.zeta, h);
    end
    tau = [tauX; 0; 0; 0; 0; tauN];

    % 1st-order wave loads from force RAOs (Fossen 2027, eq. 10.96)
    if ~useRK4u && t(i) >= nextRAO
        Uabs = sqrt(nu(1)^2 + nu(2)^2);
        if strcmp(opts.waveFrame, 'seakeeping')
            wpos = posS; wpsi = psiS;
        else
            wpos = eta(1:2); wpsi = eta(6);
        end
        if strcmp(opts.waveImpl, 'msslib')
            [tau_wave1, zeta] = waveForceRAO(t(i), waveModel, Uabs, wpsi, env.beta_wave);
        else
            [tau_wave1, zeta] = waveLoads(t(i), waveModel, wpos, Uabs, wpsi, env.beta_wave, opts.wavePhase);
        end
        nextRAO = nextRAO + opts.RAO_dt;
    end

    if mod(i - 1, kLog) == 0
        log((i - 1) / kLog + 1, :) = [t(i), zeta, tau_wave1', eta', nu', tauX, tauN];
    end

    % advance the seakeeping frame (first-order low-pass of NED velocity and heading)
    vNED = [cos(eta(6)) * nu(1) - sin(eta(6)) * nu(2); sin(eta(6)) * nu(1) + cos(eta(6)) * nu(2)];
    vbar = vbar + (h / opts.T_lp) * (vNED - vbar);
    posS0 = posS; psiS0 = psiS;
    posS = posS + h * vbar;
    psiS = psiS + (h / opts.T_lp) * (mod(eta(6) - psiS + pi, 2*pi) - pi);

    if useRK4u
        % wave load at mid-step and end of step (frame interpolated linearly); the end value is
        % the next sample's load, so each step costs two waveLoads calls
        Uabs = sqrt(nu(1)^2 + nu(2)^2);
        tau_m = waveLoads(t(i) + h/2, waveModel, 0.5 * (posS0 + posS), Uabs, 0.5 * (psiS0 + psiS), env.beta_wave, opts.wavePhase);
        [tau_1, zeta_1] = waveLoads(t(i) + h, waveModel, posS, Uabs, psiS, env.beta_wave, opts.wavePhase);
        x = rk4u(dyn, h, x, tau + tau_wave1, tau + tau_m, tau + tau_1, vessel, env.Vc, env.betaVc);
        tau_wave1 = tau_1; zeta = zeta_1;
    else
        x = rk4(dyn, h, x, tau + tau_wave1, vessel, env.Vc, env.betaVc);
    end

    if any(~isfinite(x))
        error('run_one:diverged', 'State diverged at t = %.2f s', t(i));
    end
end

% ---------------------------------------------------------------- outputs
csvFile  = fullfile(outDir, [opts.name '.csv']);
jsonFile = fullfile(outDir, [opts.name '_coef.json']);
header = 't,zeta,tau_w1,tau_w2,tau_w3,tau_w4,tau_w5,tau_w6,N,E,D,phi,theta,psi,u,v,w,p,q,r,tauX,tauN';
fid = fopen(csvFile, 'w'); fprintf(fid, '%s\n', header); fclose(fid);
dlmwrite(csvFile, log, '-append', 'precision', '%.7g');

run = struct('matFile', matFile, 'matPath', matPath, 'Hs', Hs, 'Tp', Tp, ...
    'beta_deg', beta_deg, 'U_ref', U_ref, 'seed', seed, 'T_final', T_final, 'h', h, ...
    'RAO_dt', opts.RAO_dt, 'psi_ref_deg', opts.psi_ref_deg, 'csv', csvFile, ...
    'aperiodic_T_requested', aper, 'aperiodic_T_used', aperUsed, 'delta_zeta', dz, ...
    'kappa_456', cfg.damping.nonlinear_456, 'control', opts.control, 'memory', opts.memory, ...
    'delta_zeta_345_used', cfg.damping.delta_zeta_345, 'maxFreq', opts.maxFreq, 'wavePhase', opts.wavePhase, 'waveImpl', opts.waveImpl, ...
    'waveFrame', opts.waveFrame, 'T_lp', opts.T_lp, 'waveHold', opts.waveHold, 'h_log', opts.h_log, ...
    'created', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
export_coef(jsonFile, vessel, env, run);

% summary after a 30 s transient
k0 = find(log(:, 1) >= min(30, T_final / 2), 1);
out = struct();
out.csv = csvFile; out.json = jsonFile; out.name = opts.name;
out.std_heave_m   = std(log(k0:end, 11));
out.std_roll_deg  = std(log(k0:end, 12)) * 180 / pi;
out.std_pitch_deg = std(log(k0:end, 13)) * 180 / pi;
out.max_roll_deg  = max(abs(log(k0:end, 12))) * 180 / pi;
out.mean_speed    = mean(log(k0:end, 15));
out.Hs_realized   = 4 * std(log(k0:end, 2));          % 4*std(zeta) should be close to Hs
if opts.verbose
    fprintf('%s\n  Hs_in %.2f  Hs_real %.2f | std heave %.3f m, roll %.2f deg, pitch %.2f deg | max roll %.2f deg | U %.2f m/s\n', ...
        opts.name, Hs, out.Hs_realized, out.std_heave_m, out.std_roll_deg, out.std_pitch_deg, out.max_roll_deg, out.mean_speed);
end
end

% ---------------------------------------------------------------- helpers
function s = setDefault(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end

function setSeed(seed)
% MSS uses rng(); Octave < 7 lacks it, fall back to the legacy generators.
if exist('rng', 'file') == 2 || exist('rng', 'builtin') == 5
    rng(seed);
else
    rand('state', seed); randn('state', seed);
end
end
