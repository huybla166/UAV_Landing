function [vessel, cfg, info] = build_model(matFile, w0, verbose)
% BUILD_MODEL  Load an MSS vessel file and build the constant-coefficient 6-DOF model
% used by hydroVessel.m (MSS 2.0): M = MRB + A_eq, D = B_eq + Bv, G, kappa_4..6.
%
%   [vessel, cfg, info] = build_model(matFile, w0)
%   w0 : spectral peak frequency (rad/s) used to weight A(w), B(w) (power-based method)
%
% MSS rejects aperiodic damping time constants T_i > M(i,i)/B_eq(i,i) (they would need
% negative viscous damping) and B_eq depends on w0. For sweeps over Tp we clamp T_i to
% 95 % of that limit and report requested and used values in INFO.

if nargin < 3, verbose = true; end
named = {'supply.mat','s175.mat','tanker.mat','semisub.mat','fpso.mat','testship.mat','lauv_marie.mat'};
useGeneric = ~any(strcmpi(matFile, named));
cfg = hydroVesselConfig(matFile, useGeneric);

matPath = resolveHydroVesselFile(matFile);
S = load(matPath, 'vessel');
vessel = S.vessel;
vessel.matFile = matFile;
vessel.kappa_4 = cfg.damping.nonlinear_456(1);
vessel.kappa_5 = cfg.damping.nonlinear_456(2);
vessel.kappa_6 = cfg.damping.nonlinear_456(3);

if isfield(cfg.damping, 'T_126')
    aper = cfg.damping.T_126;   dz = cfg.damping.delta_zeta_345;  aperIdx = [1 2 6];
else
    aper = cfg.damping.T_1236;  dz = cfg.damping.delta_zeta_45;   aperIdx = [1 2 3 6];
end
probe = computeManeuveringModel(vessel, w0, 1e-3 * ones(size(aper)), dz, 0);  % always feasible
aperUsed = aper;
for k = 1:numel(aperIdx)
    i = aperIdx(k);
    Tmax = probe.M(i, i) / probe.powerBased.B_eq(i, i);
    if aper(k) > Tmax
        aperUsed(k) = 0.95 * Tmax;
    end
end
if any(aperUsed ~= aper) && verbose
    fprintf('  damping time constants clamped: %s -> %s s\n', mat2str(aper, 4), mat2str(aperUsed, 4));
end
vessel = computeManeuveringModel(vessel, w0, aperUsed, dz, 0);

info = struct('matPath', matPath, 'w0', w0, 'aperiodic_T_requested', aper, ...
    'aperiodic_T_used', aperUsed, 'delta_zeta', dz, 'kappa_456', cfg.damping.nonlinear_456);
end
