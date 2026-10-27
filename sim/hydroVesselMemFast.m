function xdot = hydroVesselMemFast(x, tau_ext, vessel, Vc, betaVc)
% HYDROVESSELMEMFAST  Same equations as hydroVesselMem.m, written out for speed in Octave:
% rbody/m2c/eulerang/crossFlowDrag/XuuITTC inlined, constants from prepare_fast.m.
% Checked against hydroVesselMem.m by sim/check_fast_dynamics.m.
f = vessel.fast;
nu = x(1:6); eta = x(7:12); xr = x(13:end);

if Vc ~= 0
    v_c = [Vc * cos(betaVc - eta(6)); Vc * sin(betaVc - eta(6)); 0];
    w = nu(4:6);
    nu_c_dot = [-[w(2)*v_c(3) - w(3)*v_c(2); w(3)*v_c(1) - w(1)*v_c(3); w(1)*v_c(2) - w(2)*v_c(1)]; 0; 0; 0];
    nu_r = nu - [v_c; 0; 0; 0];
else
    nu_c_dot = zeros(6, 1);
    nu_r = nu;
end
v1 = nu_r(1:3); v2 = nu_r(4:6); w = nu(4:6);

% rigid-body Coriolis about the CG: [m S(w) v1 ; -S(Ig w) v2]  (S(a) b = a x b)
Igw = f.Ig * w;
crb = [f.m * [w(2)*v1(3) - w(3)*v1(2); w(3)*v1(1) - w(1)*v1(3); w(1)*v1(2) - w(2)*v1(1)];
       -[Igw(2)*v2(3) - Igw(3)*v2(2); Igw(3)*v2(1) - Igw(1)*v2(3); Igw(1)*v2(2) - Igw(2)*v2(1)]];
% added-mass Coriolis m2c(A_inf, nu_r): [-a1 x v2 ; -a1 x v1 - a2 x v2]
a1 = f.M11 * v1 + f.M12 * v2; a2 = f.M21 * v1 + f.M22 * v2;
a1v2 = [a1(2)*v2(3) - a1(3)*v2(2); a1(3)*v2(1) - a1(1)*v2(3); a1(1)*v2(2) - a1(2)*v2(1)];
a1v1 = [a1(2)*v1(3) - a1(3)*v1(2); a1(3)*v1(1) - a1(1)*v1(3); a1(1)*v1(2) - a1(2)*v1(1)];
a2v2 = [a2(2)*v2(3) - a2(3)*v2(2); a2(3)*v2(1) - a2(1)*v2(3); a2(1)*v2(2) - a2(2)*v2(1)];
ca = [-a1v2; -a1v1 - a2v2];

% damping: Dv nu_r with the nonlinear diagonal terms of hydroVesselMem.m
u = nu_r(1);
Re = max(f.L * abs(u) / 1e-6, 1e5);
Xuu = f.Xuu_const * 0.075 / (log10(Re) - 2)^2;
if f.Dv_offdiag, d = f.Dv * nu_r; else, d = f.Dv_diag .* nu_r; end
d(1) = (f.Dv_diag(1) * exp(-3 * abs(u)) - Xuu * abs(u)) * u;
d(4) = d(4) + vessel.kappa_4 * vessel.Dref(4) * abs(nu(4)) * nu_r(4);
d(5) = d(5) + vessel.kappa_5 * vessel.Dref(5) * abs(nu(5)) * nu_r(5);
d(6) = d(6) + f.Dv_diag(6) * vessel.kappa_6 * abs(nu(6)) * nu_r(6);

% cross-flow drag, Hoerner, 20 strips (crossFlowDrag.m)
Uh = nu_r(2) + f.xL * nu_r(6); Uv = nu_r(3) + f.xL * nu_r(5);
Qh = abs(Uh) .* Uh; Qv = abs(Uv) .* Uv;
tcf = -f.kcf * [0; sum(Qh); sum(Qv); 0; f.xL' * Qv; f.xL' * Qh];

mu = vessel.memory.Cr * xr + vessel.memory.Dr * nu_r;
nudot = nu_c_dot + vessel.Minv_inf * (tau_ext + tcf - crb - ca - d - mu - vessel.G * eta);

% kinematics eta_dot = J(eta) nu (Rzyx, Tzyx)
cphi = cos(eta(4)); sphi = sin(eta(4)); cth = cos(eta(5)); sth = sin(eta(5)); cpsi = cos(eta(6)); spsi = sin(eta(6));
R = [cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*cphi*sth;
     spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi;
     -sth,      cth*sphi,                  cth*cphi];
Tm = [1, sphi*sth/cth, cphi*sth/cth; 0, cphi, -sphi; 0, sphi/cth, cphi/cth];
etadot = [R * nu(1:3); Tm * nu(4:6)];

xdot = [nudot; etadot; vessel.memory.Ar * xr + vessel.memory.Br * nu_r];
end
