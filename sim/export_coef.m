function export_coef(jsonFile, vessel, env, run)
% EXPORT_COEF  Write the constant 6-DOF model used by hydroVessel.m to JSON.
%   The PINN uses these as the "answer key" (forward problem) or as the
%   target to recover (inverse problem).
%
%   Equations (hydroVessel.m, MSS 2.0):
%     eta_dot = J(eta) nu
%     nu_dot  = Minv * ( tau_ext + tau_crossflow - (CRB + CA + D_nl) nu_r - G eta )
%     D_nl(4,4) = D(4,4) (1 + kappa_4 |p|), same for (5,5), (6,6)

s = struct();
s.schema   = 'usv-landing-prediction/coef v2';
s.source   = 'MSS 2.0 hydroVessel + computeManeuveringModel (power-based constant matrices)';
s.run      = run;

s.MRB  = vessel.MRB;
s.G    = vessel.G;      % restoring (6x6)
s.kappa = [vessel.kappa_4, vessel.kappa_5, vessel.kappa_6];
if isfield(vessel, 'memory')
    % model actually integrated: hydroVesselMem.m (Cummins fluid memory)
    s.model = 'cummins_memory';
    s.source = 'MSS 2.0 vessel data (Capytaine) + Cummins fluid memory identified with MSS FDI (sim/build_memory.m)';
    s.M_inf = vessel.MRB + vessel.MAinf;   % inertia used in the equations
    s.MA_inf = vessel.MAinf;
    s.Dv = vessel.Dv;                      % constant viscous / maneuvering damping
    s.Dref = vessel.Dref;                  % quadratic damping scale: kappa_k * Dref_k * |nu_k|
    s.wn_bem = vessel.wn;                  % BEM natural frequencies (rad/s), entries 3..5
    s.memory = struct('n', vessel.memory.n, 'Ar', vessel.memory.Ar, 'Br', vessel.memory.Br, ...
        'Cr', vessel.memory.Cr, 'Dr', vessel.memory.Dr, 'report', {vessel.memory.report});
    % constant-coefficient approximation at w0 (MSS power-based), kept as a reference for the PINN
    s.constant_model = struct('M', vessel.M, 'D', vessel.D, 'MA', vessel.MA, 'w0', env.w0);
    Mnat = vessel.MRB + vessel.MAinf;
else
    s.model = 'constant_power_based';
    s.M    = vessel.M;      % MRB + MA (6x6)
    s.MA   = vessel.MA;
    s.D    = vessel.D;      % linear damping (6x6), B_eq + Bv
    Mnat = vessel.M;
end

main = vessel.main;
keep = {'name','m','rho','g','Lpp','B','T','nabla','k44','k55','k66','C_B','GM_T','GM_L'};
s.main = struct();
for i = 1:numel(keep)
    if isfield(main, keep{i})
        s.main.(keep{i}) = main.(keep{i});
    end
end

% Natural periods (sanity checks): BEM fixed point when the memory model is used, else the
% diagonal of the constant model
s.natural_period_s = struct();
dofs = {'heave', 3; 'roll', 4; 'pitch', 5};
for i = 1:size(dofs, 1)
    k = dofs{i, 2};
    if isfield(vessel, 'wn') && isfinite(vessel.wn(k))
        s.natural_period_s.(dofs{i, 1}) = 2 * pi / vessel.wn(k);
    elseif vessel.G(k, k) > 0
        s.natural_period_s.(dofs{i, 1}) = 2 * pi * sqrt(Mnat(k, k) / vessel.G(k, k));
    end
end

s.environment = struct('Hs', env.Hs, 'w0', env.w0, 'Tp', 2*pi/env.w0, ...
    'beta_wave_deg', env.beta_wave * 180 / pi, 'spectrumType', env.spectrumType, ...
    'spreadingFlag', env.spreadingFlag, 'numFreqIntervals', env.numFreqIntervals, ...
    'numDirections', env.numDirections, 'Vc', env.Vc);

try
    txt = jsonencode(s, 'PrettyPrint', true);
catch
    txt = jsonencode(s);
end
fid = fopen(jsonFile, 'w');
if fid < 0
    error('export_coef:open', 'Cannot write %s', jsonFile);
end
fprintf(fid, '%s\n', txt);
fclose(fid);
end
