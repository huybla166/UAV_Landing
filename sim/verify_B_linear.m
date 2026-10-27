function verify_B_linear(caseName, matFile, Hs, Tp, beta_deg, U_ref, control, kappa456, seed, T_final, outDir)
% VERIFY_B_LINEAR  Step 1.4 (verification B): how much of the simulated motion is the
% linear "sum of sines" of the base paper, and how much is dynamic nonlinearity.
%
% 1. Run the full model (run_one; reuses <outDir>/B_<case>.csv if it already exists).
% 2. Drive two LINEAR models with exactly the logged forcing (zero-order hold per step,
%    as in run_one), same h, same initial state:
%      L0  zero-speed constant ODE    M eta'' + D eta' + G eta = tau
%      LU  numerical linearization of hydroVessel (12 states, incl. Coriolis/Munk,
%          ITTC/cross-flow, kinematics eta' = J(eta) nu) about steady forward speed U_ref:
%          dx' = A dx + [M^-1; 0] dtau,  dtau = tau_w + tau_ctrl - tau0
%    The response of either to a sum-of-sines force is itself a sum of sines (eq. 5 of the
%    base paper); the remaining difference is dynamic nonlinearity.
% 3. Per DOF after 60 s: std, correlation, R2 = 1 - var(nl - lin)/var(nl).
%    Yaw is reported but not meaningful when the autopilot moment is replayed open loop.
% 4. Motion spectrum predicted component by component (psi = 0, vessel at origin; valid
%    for U = 0): coherent sum over directions at each frequency (this realization) and the
%    incoherent sum (expected spectrum), versus the Welch spectrum of the simulation.

if ischar(kappa456) || isempty(kappa456), kappa456 = []; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
h = 0.05;
csv = fullfile(outDir, ['B_' caseName '.csv']);
if exist(csv, 'file') ~= 2
    opts = struct('name', ['B_' caseName], 'control', control, 'kappa456', kappa456, 'verbose', false);
    run_one(matFile, Hs, Tp, beta_deg, U_ref, seed, T_final, h, outDir, opts);
end
X = dlmread(csv, ',', 1, 0);
t = X(:, 1); n = numel(t);
tau_w = X(:, 3:8); eta_nl = X(:, 9:14); nu_nl = X(:, 15:20); tauX = X(:, 21); tauN = X(:, 22);

[vessel, ~, ~] = build_model(matFile, 2*pi/Tp, false);
if ~isempty(kappa456)
    vessel.kappa_4 = kappa456(1); vessel.kappa_5 = kappa456(2); vessel.kappa_6 = kappa456(3);
end
kap = [vessel.kappa_4, vessel.kappa_5, vessel.kappa_6];
M = vessel.M; D = vessel.D; G = vessel.G; Minv = inv(M);

% ------------------------------------------------------------ L0: zero-speed constant ODE
f0 = @(x, tau) [Minv * (tau - D * x(1:6) - G * x(7:12)); x(1:6)];
x = [nu_nl(1, :)'; eta_nl(1, :)']; x(1) = 0;          % no forward speed in L0
eta_L0 = zeros(n, 6);
for i = 1:n
    eta_L0(i, :) = x(7:12)';
    tau = tau_w(i, :)' + [0; 0; 0; 0; 0; tauN(i)];
    k1 = f0(x, tau); k2 = f0(x + h/2*k1, tau); k3 = f0(x + h/2*k2, tau); k4 = f0(x + h*k3, tau);
    x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

% ------------------------------------------------------------ LU: linearization about U_ref
x0 = [U_ref; zeros(11, 1)];
xd0 = hydroVessel(x0, zeros(6, 1), vessel, 0, 0);
tau0 = -M * xd0(1:6);                                   % force holding the steady state
A = zeros(12); e = 1e-6;
for q = 1:12
    dx = zeros(12, 1); dx(q) = e;
    A(:, q) = (hydroVessel(x0 + dx, tau0, vessel, 0, 0) - hydroVessel(x0 - dx, tau0, vessel, 0, 0)) / (2*e);
end
Bm = [Minv; zeros(6)];
fU = @(dx, dtau) A * dx + Bm * dtau;
dx = [nu_nl(1, :)'; eta_nl(1, :)'] - x0;
eta_LU = zeros(n, 6);
for i = 1:n
    eta_LU(i, :) = dx(7:12)';
    dtau = tau_w(i, :)' + [tauX(i); 0; 0; 0; 0; tauN(i)] - tau0;
    k1 = fU(dx, dtau); k2 = fU(dx + h/2*k1, dtau); k3 = fU(dx + h/2*k2, dtau); k4 = fU(dx + h*k3, dtau);
    dx = dx + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end
eta_LU(:, 1) = eta_LU(:, 1) + U_ref * t;                % back to absolute N (not used)

% ------------------------------------------------------------ metrics
m = t >= 60;
names = {'heave', 'roll', 'pitch', 'yaw'}; cols = [3 4 5 6];
fid = fopen(fullfile(outDir, ['B_' caseName '_metrics.csv']), 'w');
fprintf(fid, 'case,matFile,Hs,Tp,beta_deg,U_ref,control,kappa4,kappa5,kappa6,dof,std_nl,std_L0,std_LU,corr_L0,corr_LU,R2_L0,R2_LU\n');
for j = 1:4
    a = eta_nl(m, cols(j)); a = a - mean(a);
    b0 = eta_L0(m, cols(j)); b0 = b0 - mean(b0);
    bU = eta_LU(m, cols(j)); bU = bU - mean(bU);
    r0 = sum(a .* b0) / sqrt(sum(a.^2) * sum(b0.^2));
    rU = sum(a .* bU) / sqrt(sum(a.^2) * sum(bU.^2));
    R0 = 1 - var(a - b0) / var(a);  RU = 1 - var(a - bU) / var(a);
    fprintf(fid, '%s,%s,%g,%g,%g,%g,%s,%g,%g,%g,%s,%.6g,%.6g,%.6g,%.6f,%.6f,%.6f,%.6f\n', caseName, matFile, ...
        Hs, Tp, beta_deg, U_ref, control, kap(1), kap(2), kap(3), names{j}, std(a), std(b0), std(bU), r0, rU, R0, RU);
    fprintf('%-12s %-6s std nl %.4g | L0 %.4g R2 %8.4f | LU %.4g R2 %8.4f\n', caseName, names{j}, std(a), std(b0), R0, std(bU), RU);
end
fclose(fid);
% time series of the three models (heave m, roll/pitch rad), for figures
fid = fopen(fullfile(outDir, ['B_' caseName '_series.csv']), 'w');
fprintf(fid, 't,heave_nl,heave_L0,heave_LU,roll_nl,roll_L0,roll_LU,pitch_nl,pitch_L0,pitch_LU\n');
S = [t, eta_nl(:,3), eta_L0(:,3), eta_LU(:,3), eta_nl(:,4), eta_L0(:,4), eta_LU(:,4), eta_nl(:,5), eta_L0(:,5), eta_LU(:,5)];
fprintf(fid, '%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n', S');
fclose(fid);

% ------------------------------------------------------------ spectra (component prediction)
cfg = hydroVesselConfig(matFile, true);
env = cfg.environment; env.Hs = Hs; env.w0 = 2*pi/Tp; env.beta_wave = beta_deg*pi/180;
env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100; env.numDirections = 24;
w = vessel.forceRAO.w(:); env.maxFreq = max(w(w < 10));
rng(seed); [~, wm] = waveInitialization(vessel, env);
Om = wm.Omega(:); Nf = numel(Om); Nd = numel(wm.mu);
ang = wm.raoAngles(:);
varCoh = zeros(Nf, 6); varExp = zeros(Nf, 6);
for i = 1:Nf
    H = inv(-Om(i)^2 * M + 1i * Om(i) * D + G);
    rsum = zeros(6, 1);
    for jd = 1:Nd
        b = mod(env.beta_wave + wm.mu(jd), 2*pi);
        F = zeros(6, 1);
        for d = 1:6
            F(d) = interp1(ang, wm.RAO_re{d}(i, :), b) + 1i * interp1(ang, wm.RAO_im{d}(i, :), b);
        end
        r = H * F * wm.Amp(i, jd);
        rsum = rsum + r * exp(1i * wm.randomPhases(i, jd));
        varExp(i, :) = varExp(i, :) + (abs(r').^2) / 2;
    end
    varCoh(i, :) = (abs(rsum').^2) / 2;
end
dOm = gradient(Om);
fs = 1 / h; L = 4096; hop = L / 2; win = 0.5 - 0.5 * cos(2*pi*(0:L-1)'/L);
yy = eta_nl(m, 3:5); yy = yy - mean(yy); nseg = floor((size(yy,1) - L) / hop) + 1;
P = zeros(L/2 + 1, 3);
for s = 1:nseg
    seg = yy((s-1)*hop + (1:L), :) .* win;
    Fs = fft(seg);
    P = P + abs(Fs(1:L/2+1, :)).^2;
end
P = P / (nseg * sum(win.^2) * fs) * 2 / (2*pi);          % one-sided, per rad/s
wW = 2*pi*(0:L/2)' * fs / L;
fid = fopen(fullfile(outDir, ['B_' caseName '_spectra.csv']), 'w');
fprintf(fid, 'kind,w,heave,roll,pitch\n');
for i = 1:Nf, fprintf(fid, 'pred_coherent,%.6g,%.6g,%.6g,%.6g\n', Om(i), varCoh(i,3)/dOm(i), varCoh(i,4)/dOm(i), varCoh(i,5)/dOm(i)); end
for i = 1:Nf, fprintf(fid, 'pred_expected,%.6g,%.6g,%.6g,%.6g\n', Om(i), varExp(i,3)/dOm(i), varExp(i,4)/dOm(i), varExp(i,5)/dOm(i)); end
for i = 1:numel(wW), fprintf(fid, 'sim,%.6g,%.6g,%.6g,%.6g\n', wW(i), P(i, 1), P(i, 2), P(i, 3)); end
fclose(fid);
fprintf('%s: std sim (after 60 s) heave %.4g roll %.4g pitch %.4g | component prediction coherent %.4g %.4g %.4g | expected %.4g %.4g %.4g\n', ...
    caseName, std(yy(:,1)), std(yy(:,2)), std(yy(:,3)), sqrt(sum(varCoh(:,3))), sqrt(sum(varCoh(:,4))), sqrt(sum(varCoh(:,5))), ...
    sqrt(sum(varExp(:,3))), sqrt(sum(varExp(:,4))), sqrt(sum(varExp(:,5))));
end
