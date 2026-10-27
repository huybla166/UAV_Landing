function verify_E_memory(matFile, a, betas_deg, omegas, h, T_sim, outCsv, waveHold)
% VERIFY_E_MEMORY  Regular-wave check of the Cummins fluid-memory model (hydroVesselMemFast).
%
% For each (beta, w): simulate the full time-domain model (controllers off, U = 0, kappa = 0,
% small amplitude a) at the dataset time step h, fit the steady complex amplitude of
% heave/roll/pitch on the last half, and compare with
%
%   E1  H_fit(w) F    H_fit = (-w^2 (MRB + A_inf) + i w (K_fit(iw) + Dv) + G)^-1,
%                     K_fit(s) = Cr (sI - Ar)^-1 Br + Dr   (the identified state-space memory:
%                     sim vs E1 tests only the time integration)
%   E2  H_fd(w) F     H_fd = (-w^2 (MRB + A(w)) + i w (B(w) + Dv) + G)^-1   (BEM coefficients:
%                     sim vs E2 tests fit + residualization + integration together)
%   E3  H_const(w) F  H_const = (-w^2 M + i w D + G)^-1, the MSS constant model at
%                     w0 = 2 pi / 1.9 used before (shows what the memory model changes)
%
% Writes one row per (beta, w, DOF) to outCsv.

if nargin < 1 || isempty(matFile),   matFile = 'otter_55kg.mat'; end
if nargin < 2 || isempty(a),         a = 0.002; end
if nargin < 3 || isempty(betas_deg), betas_deg = [180 135 90]; end
if nargin < 4 || isempty(omegas),    omegas = [2.0 3.0 4.0 5.0 5.5 6.0 6.5 7.0 7.5 8.0 9.0]; end
if nargin < 5 || isempty(h),         h = 0.05; end
if nargin < 6 || isempty(T_sim),     T_sim = 60; end
if nargin < 8 || isempty(waveHold),  waveHold = 'rk4'; end   % as run_one.m; 'zoh' = MSS rk4.m
[~, base] = fileparts(matFile);
if nargin < 7 || isempty(outCsv)
    outCsv = fullfile(fileparts(mfilename('fullpath')), '..', 'analysis', 'sim', ...
        sprintf('E_memory_%s_h%g_%s.csv', base, h, waveHold));
end

[vessel, cfg] = build_model(matFile, 2*pi/1.9, false);
vessel.kappa_4 = 0; vessel.kappa_5 = 0; vessel.kappa_6 = 0;   % linear check
vessel = build_memory(vessel, cfg, false);
vessel = prepare_fast(vessel);
mem = vessel.memory; nMem = mem.n;
MRB = vessel.MRB; Ainf = vessel.MAinf; Dv = vessel.Dv; G = vessel.G;
fr = vessel.freqs(:);
dofs = [3 4 5]; dofName = {'surge','sway','heave','roll','pitch','yaw'};

fid = fopen(outCsv, 'w');
fprintf(fid, 'vessel,h,waveHold,a,beta_deg,w,dof,sim_amp,sim_phase,fit_amp,fit_phase,fd_amp,fd_phase,const_amp,const_phase,fit_rms_rel\n');
t = (0:h:T_sim)';
n = numel(t);
fitMask = t >= T_sim / 2;
for b = betas_deg(:)'
    beta = b * pi / 180;
    for w = omegas(:)'
        wm = regular_wave_model(vessel, w, a);
        x = zeros(12 + nMem, 1);
        Y = zeros(n, 6);
        for i = 1:n
            Y(i, :) = x(7:12)';
            tw = waveLoads(t(i), wm, [0; 0], 0, 0, beta, 'position');
            if strcmp(waveHold, 'rk4')
                twm = waveLoads(t(i) + h/2, wm, [0; 0], 0, 0, beta, 'position');
                tw1 = waveLoads(t(i) + h, wm, [0; 0], 0, 0, beta, 'position');
                x = rk4u(@hydroVesselMemFast, h, x, tw, twm, tw1, vessel, 0, 0);
            else
                x = rk4(@hydroVesselMemFast, h, x, tw, vessel, 0, 0);
            end
        end
        F_re = arrayfun(@(k) interp1(wm.raoAngles, wm.RAO_re{k}, mod(beta, 2*pi)), 1:6);
        F_im = arrayfun(@(k) interp1(wm.raoAngles, wm.RAO_im{k}, mod(beta, 2*pi)), 1:6);
        F = a * (F_re + 1i * F_im).';
        s = 1i * w;
        Kfit = mem.Cr * ((s * eye(nMem) - mem.Ar) \ mem.Br) + mem.Dr;
        P1 = (-w^2 * (MRB + Ainf) + s * (Kfit + Dv) + G) \ F;
        Aw = interpMat(vessel.A, fr, w); Bw = interpMat(vessel.B, fr, w);
        P2 = (-w^2 * (MRB + Aw) + s * (Bw + Dv) + G) \ F;
        P3 = (-w^2 * vessel.M + s * vessel.D + G) \ F;
        for d = dofs
            yy = Y(fitMask, d); tt = t(fitMask);
            Phi = [ones(size(tt)), tt, cos(w * tt), sin(w * tt)];
            coef = Phi \ yy;
            X = coef(3) - 1i * coef(4);
            fitRel = sqrt(mean((yy - Phi * coef).^2)) / max(abs(X), eps);
            fprintf(fid, '%s,%g,%s,%g,%g,%.6g,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.3e\n', ...
                base, h, waveHold, a, b, w, dofName{d}, abs(X), angle(X), abs(P1(d)), angle(P1(d)), ...
                abs(P2(d)), angle(P2(d)), abs(P3(d)), angle(P3(d)), fitRel);
        end
        fprintf('beta %3g w %.2f | sim/BEM heave %.3f roll %.3f pitch %.3f\n', b, w, ...
            abs(phasor(Y, fitMask, t, w, 3)) / abs(P2(3)), abs(phasor(Y, fitMask, t, w, 4)) / abs(P2(4)), ...
            abs(phasor(Y, fitMask, t, w, 5)) / abs(P2(5)));
    end
end
fclose(fid);
fprintf('wrote %s\n', outCsv);
end

function Mw = interpMat(Arr, fr, w)
Mw = zeros(6);
for r = 1:6
    for c = 1:6
        Mw(r, c) = interp1(fr, squeeze(Arr(r, c, :)), w, 'linear');
    end
end
end

function X = phasor(Y, mask, t, w, col)
yy = Y(mask, col); tt = t(mask);
coef = [ones(size(tt)), tt, cos(w*tt), sin(w*tt)] \ yy;
X = coef(3) - 1i * coef(4);
end
