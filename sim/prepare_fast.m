function vessel = prepare_fast(vessel)
% PREPARE_FAST  Precompute the constants used by hydroVesselMemFast.m (same model as
% hydroVesselMem.m): inertia/added-mass blocks for the Coriolis terms, the Hoerner cross-flow
% coefficient and strip positions, and the ITTC wetted surface.
c = mssConstants();
f.rho_cf = c.rho_water;                       % crossFlowDrag uses mssConstants().rho_water
m = vessel.main.m;
f.m = m;
f.Ig = m * diag([vessel.main.k44^2, vessel.main.k55^2, vessel.main.k66^2]);
MA = 0.5 * (vessel.MAinf + vessel.MAinf');     % m2c symmetrizes
f.M11 = MA(1:3,1:3); f.M12 = MA(1:3,4:6); f.M21 = f.M12'; f.M22 = MA(4:6,4:6);
L = vessel.main.Lpp; B = vessel.main.B; T = vessel.main.T;
nStrips = 20; dx = L / nStrips;
f.xL = (-L/2 + ((1:nStrips)' - 0.5) * dx);
f.kcf = 0.5 * f.rho_cf * T * Hoerner(B, T) * dx;
if isfield(vessel.main, 'C_B') && vessel.main.C_B > 0, C_B = vessel.main.C_B; else, C_B = 0.65; end
f.Xuu_const = -0.5 * vessel.main.rho * (1.025 * L * (C_B * B + 1.7 * T)) * (1 + 0.1);   % times Cf(Re)
f.L = L;
f.Dv = vessel.Dv;
f.Dv_diag = diag(vessel.Dv);
f.Dv_offdiag = any(any(vessel.Dv - diag(diag(vessel.Dv))));
vessel.fast = f;
end
