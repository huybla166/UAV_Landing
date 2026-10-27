function xdot = hydroVesselMem(x, tau_ext, vessel, Vc, betaVc)
% HYDROVESSELMEM  MSS hydroVessel.m with Cummins fluid memory (unified seakeeping/maneuvering
% form, Fossen ch. 5-6): infinite-frequency added mass for inertia and Coriolis, radiation
% damping through the identified state-space memory model (build_memory.m), constant viscous /
% maneuvering damping Dv, quadratic damping kappa_k * Dref_k * |nu_k| for k = 4, 5, 6.
%
%   x = [nu (6); eta (6); xr (n)],  xdot = [nu_dot; eta_dot; xr_dot]
%   nu_dot = nu_c_dot + (MRB + A_inf)^-1 ( tau_ext + tau_crossflow
%            - (CRB + CA(A_inf) + D_nl) nu_r - Cr xr - G eta )
%   xr_dot = Ar xr + Br nu_r
% Everything else (current, ITTC surge resistance with exponential blending, cross-flow drag,
% kinematics) is taken unchanged from MSS 2.0 hydroVessel.m (2026-10-02).

nu = x(1:6);
eta = x(7:12);
xr = x(13:end);

v_c = [ Vc * cos(betaVc - eta(6)); Vc * sin(betaVc - eta(6)); 0 ];
nu_c = [v_c; zeros(3,1)];
nu_c_dot = [-Smtrx(nu(4:6)) * v_c; zeros(3,1)];
nu_r = nu - nu_c;

[~, CRB] = rbody(vessel.main.m, vessel.main.k44, vessel.main.k55, vessel.main.k66, nu(4:6), [0 0 0]');
CA = m2c(vessel.MAinf, nu_r);

% Surge: ITTC quadratic resistance blended with the linear damping (as hydroVessel.m)
if isfield(vessel.main,'C_B') && vessel.main.C_B > 0
    C_B = vessel.main.C_B;
else
    C_B = [];
end
Xuu = XuuITTC(nu_r(1), vessel.main.rho, vessel.main.Lpp, vessel.main.B, vessel.main.T, C_B);
k_u = 3;

D_nl = vessel.Dv;
D_nl(1,1) = vessel.Dv(1,1) * exp(-k_u * abs(nu_r(1))) - Xuu * abs(nu_r(1));
D_nl(4,4) = vessel.Dv(4,4) + vessel.kappa_4 * vessel.Dref(4) * abs(nu(4));
D_nl(5,5) = vessel.Dv(5,5) + vessel.kappa_5 * vessel.Dref(5) * abs(nu(5));
D_nl(6,6) = vessel.Dv(6,6) * (1 + vessel.kappa_6 * abs(nu(6)));

tau_crossflow = crossFlowDrag(vessel.main.Lpp, vessel.main.B, vessel.main.T, nu_r, 'Hoerner');

mu = vessel.memory.Cr * xr + vessel.memory.Dr * nu_r;
nudot = nu_c_dot + vessel.Minv_inf * (tau_ext + tau_crossflow - (CRB + CA + D_nl) * nu_r ...
        - mu - vessel.G * eta);
J = eulerang(eta(4), eta(5), eta(6));
etadot = J * nu;
xrdot = vessel.memory.Ar * xr + vessel.memory.Br * nu_r;

xdot = [nudot; etadot; xrdot];
end
