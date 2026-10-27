function verify_A_regular(matFile, w0_model, a, betas_deg, omegas, h, T_sim, outCsv)
% VERIFY_A_REGULAR  Step 1.3 (verification A): steady-state response in regular waves.
%
% For each (beta, w) the full nonlinear time-domain model (hydroVessel + RK4 + waveLoads,
% controllers off, U = 0, small wave amplitude) is simulated and the complex amplitude of
% heave/roll/pitch/yaw is fitted on the last half of the run. It is compared with
%
%   A1  H_const(w) F(w,beta),  H_const = (-w^2 M + i w D + G)^-1     (the same constant
%       M, D, G that hydroVessel integrates and that the PINN residual will use)
%   A2  H_fd(w) F(w,beta),     H_fd = (-w^2 (MRB + A(w)) + i w (B(w) + Bv) + C(w))^-1
%       (frequency-dependent potential-flow coefficients from Capytaine + the same Bv)
%   A3  Capytaine motion RAO (potential flow only, no viscous Bv), for reference
%
% Writes one row per (beta, w, DOF) to outCsv.

if nargin < 1 || isempty(matFile),   matFile = 'otter_55kg.mat'; end
if nargin < 2 || isempty(w0_model),  w0_model = 2*pi/1.9; end
if nargin < 3 || isempty(a),         a = 0.002; end   % small: O(a^2) Coriolis drift negligible
if nargin < 4 || isempty(betas_deg), betas_deg = [180 135 90]; end
if nargin < 5 || isempty(omegas),    omegas = [2.0 3.0 4.0 5.0 5.5 6.0 6.5 7.0 7.5 8.0 9.0]; end
if nargin < 6 || isempty(h),         h = 0.01; end
if nargin < 7 || isempty(T_sim),     T_sim = 60; end
if nargin < 8 || isempty(outCsv)
    outCsv = fullfile(fileparts(mfilename('fullpath')), '..', 'analysis', 'sim', 'A_regular_wave.csv');
end

[vessel, ~, info] = build_model(matFile, w0_model, true);
vessel.kappa_4 = 0; vessel.kappa_5 = 0; vessel.kappa_6 = 0;   % linear check
M = vessel.M; D = vessel.D; G = vessel.G;
Bv = vessel.powerBased.Bv;
fr = vessel.freqs(:);
dofs = [3 4 5 6]; dofName = {'surge','sway','heave','roll','pitch','yaw'};

fid = fopen(outCsv, 'w');
fprintf(fid, 'vessel,w0_model,a,beta_deg,w,dof,sim_amp,sim_phase,const_amp,const_phase,fd_amp,fd_phase,capy_amp,capy_phase,fit_rms_rel\n');

t = (0:h:T_sim)';
n = numel(t);
fitMask = t >= T_sim / 2;
for b = betas_deg(:)'
    beta = b * pi / 180;
    for w = omegas(:)'
        wm = regular_wave_model(vessel, w, a);
        % ------------------------------------------------ time domain
        x = zeros(12, 1);
        Y = zeros(n, 12);
        for i = 1:n
            Y(i, :) = x';
            % seakeeping frame: at U = 0 the mean position/heading is the origin / psi = 0
            [tw, ~] = waveLoads(t(i), wm, [0; 0], 0, 0, beta, 'position');
            x = rk4(@hydroVessel, h, x, tw, vessel, 0, 0);
        end
        % ------------------------------------------------ force phasor at psi = 0
        [F_re] = arrayfun(@(k) interp1(wm.raoAngles, wm.RAO_re{k}, mod(beta, 2*pi)), 1:6);
        [F_im] = arrayfun(@(k) interp1(wm.raoAngles, wm.RAO_im{k}, mod(beta, 2*pi)), 1:6);
        F = a * (F_re + 1i * F_im).';
        % ------------------------------------------------ A1: same constant model
        Pc = (-w^2 * M + 1i * w * D + G) \ F;
        % ------------------------------------------------ A2: frequency-dependent coefficients
        Aw = interpMat(vessel.A, fr, w); Bw = interpMat(vessel.B, fr, w); Cw = interpMat(vessel.C, fr, w);
        Pf = (-w^2 * (vessel.MRB + Aw) + 1i * w * (Bw + Bv) + Cw) \ F;
        % ------------------------------------------------ A3: Capytaine motion RAO
        mr = vessel.motionRAO; mw = mr.w(:);
        bi = interp1([vessel.headings(:); 2*pi], [(1:numel(vessel.headings))'; 1], mod(beta, 2*pi), 'nearest');
        for d = dofs
            % fit x = c0 + c1 t + A cos wt + B sin wt   ->  phasor X = A - iB
            yy = Y(fitMask, 6 + d); tt = t(fitMask);
            Phi = [ones(size(tt)), tt, cos(w * tt), sin(w * tt)];
            coef = Phi \ yy;
            X = coef(3) - 1i * coef(4);
            res = yy - Phi * coef;
            fitRel = sqrt(mean(res.^2)) / max(abs(X), eps);
            capA = interp1(mw, mr.amp{d}(:, bi, 1), w) * a;
            capP = interp1(mw, unwrap(mr.phase{d}(:, bi, 1)), w);
            fprintf(fid, '%s,%.6g,%.6g,%g,%.6g,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.3e\n', ...
                matFile, w0_model, a, b, w, dofName{d}, abs(X), angle(X), abs(Pc(d)), angle(Pc(d)), ...
                abs(Pf(d)), angle(Pf(d)), capA, capP, fitRel);
        end
        fprintf('beta %3g  w %.2f  roll sim/const %.4f  pitch %.4f  heave %.4f\n', b, w, ...
            ampRatio(Y, fitMask, t, w, 10, Pc(4)), ampRatio(Y, fitMask, t, w, 11, Pc(5)), ampRatio(Y, fitMask, t, w, 9, Pc(3)));
    end
end
fclose(fid);
fprintf('wrote %s\n', outCsv);
fprintf('model built at w0 = %.3f rad/s; aperiodic T used %s\n', info.w0, mat2str(info.aperiodic_T_used, 4));
end

function Mw = interpMat(Arr, fr, w)
Mw = zeros(6);
for r = 1:6
    for c = 1:6
        Mw(r, c) = interp1(fr, squeeze(Arr(r, c, :)), w, 'linear');
    end
end
end

function r = ampRatio(Y, mask, t, w, col, P)
yy = Y(mask, col); tt = t(mask);
coef = [ones(size(tt)), tt, cos(w*tt), sin(w*tt)] \ yy;
r = abs(coef(3) - 1i*coef(4)) / max(abs(P), eps);
end
