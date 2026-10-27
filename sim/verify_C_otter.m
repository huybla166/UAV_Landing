function verify_C_otter(outCsv)
% VERIFY_C_OTTER  Step 1.5 (verification C): Otter coefficients from MSS-Capytaine
% (BEM, approximate hull) versus the MSS sea-trial-calibrated model CRAFT/USV/models/otter.m.
%
% For 55 kg (otter.m mP = 0) and 80 kg (mP = 25 kg at CO, matching the CG in the
% MSS-Capytaine otter_80kg config) it reports, for heave/roll/pitch:
%   - restoring C_kk, added-mass ratio A_kk / MRB_kk, natural frequency, damping ratio;
%   - BEM values at the self-consistent natural frequency (fixed point of
%     w = sqrt(C_kk / (MRB_kk + A_kk(w)))) with B_kk(w_n) + Bv_kk;
%   - the constant model used for the dataset (build_model at w0 = 2*pi/1.9), via modalFromMDG;
%   - otter.m linearized at rest (M returned by otter, G and D by central differences),
%     via modalFromMDG.
% Mesh variants otter_55kg_coarse / _fine are included when present.

if nargin < 1 || isempty(outCsv)
    outCsv = fullfile(fileparts(mfilename('fullpath')), '..', 'analysis', 'sim', 'C_otter_coeffs.csv');
end
w0_data = 2*pi/1.9;
cases = {'otter_55kg.mat', 0; 'otter_80kg.mat', 25; 'otter_55kg_coarse.mat', 0; 'otter_55kg_fine.mat', 0};
dofN = {'heave', 'roll', 'pitch'}; K = [3 4 5];
fid = fopen(outCsv, 'w');
fprintf(fid, 'source,mesh,mass_kg,dof,C_kk,MRB_kk,A_kk,A_over_MRB,w_n,T_n,zeta\n');

for c = 1:size(cases, 1)
    f = cases{c, 1}; mP = cases{c, 2};
    try
        p = resolveHydroVesselFile(f);
    catch
        fprintf('skip %s (not found)\n', f); continue
    end
    S = load(p, 'vessel'); v = S.vessel;
    mesh = 'base'; if ~isempty(strfind(f, 'coarse')), mesh = 'coarse'; elseif ~isempty(strfind(f, 'fine')), mesh = 'fine'; end
    mass = v.main.m;
    fr = v.freqs(:); fin = fr > 0 & fr < 10;
    % viscous Bv as in the dataset (delta_zeta via computeManeuveringModel at w0_data)
    [vm, ~, info] = build_model(f, w0_data, false);
    Bv = vm.powerBased.Bv;
    fprintf('\n=== %s (%s mesh, %.0f kg)  Bv diag(3:5) = %s\n', f, mesh, mass, mat2str(diag(Bv(3:5,3:5))', 4));
    % ---------------------------------------------- BEM, frequency dependent, self-consistent w_n
    for j = 1:3
        k = K(j);
        Akk = squeeze(v.A(k, k, :)); Bkk = squeeze(v.B(k, k, :)); Ckk = squeeze(v.C(k, k, :));
        Cc = interp1(fr(fin), Ckk(fin), 3);           % C is frequency independent
        w = 6;
        for it = 1:100
            wn = sqrt(Cc / (v.MRB(k, k) + interp1(fr(fin), Akk(fin), w)));
            if abs(wn - w) < 1e-10, break; end
            w = 0.5 * w + 0.5 * wn;
        end
        Aw = interp1(fr(fin), Akk(fin), wn); Bw = interp1(fr(fin), Bkk(fin), wn);
        zeta = (Bw + Bv(k, k)) / (2 * sqrt(Cc * (v.MRB(k, k) + Aw)));
        fprintf('  BEM  %-5s C %9.2f  A/MRB %.3f  w_n %.3f rad/s  T_n %.3f s  zeta %.3f (rad. %.3f + visc. %.3f)\n', ...
            dofN{j}, Cc, Aw / v.MRB(k, k), wn, 2*pi/wn, zeta, Bw / (2*sqrt(Cc*(v.MRB(k,k)+Aw))), ...
            Bv(k,k) / (2*sqrt(Cc*(v.MRB(k,k)+Aw))));
        fprintf(fid, 'BEM_freq_dependent,%s,%g,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n', mesh, mass, dofN{j}, ...
            Cc, v.MRB(k, k), Aw, Aw / v.MRB(k, k), wn, 2*pi/wn, zeta);
    end
    % ---------------------------------------------- constant model of the dataset
    [~, ~, ~, w3, w4, w5, z3, z4, z5] = modalFromMDG(vm.M, vm.D, vm.G, 0);
    wc = [w3 w4 w5]; zc = [z3 z4 z5];
    for j = 1:3
        k = K(j);
        fprintf('  CONST(w0=%.2f) %-5s A/MRB %.3f  w_n %.3f  T_n %.3f s  zeta %.3f\n', w0_data, dofN{j}, ...
            vm.MA(k, k) / vm.MRB(k, k), wc(j), 2*pi/wc(j), zc(j));
        fprintf(fid, 'constant_model_w0_%.2f,%s,%g,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n', w0_data, mesh, mass, dofN{j}, ...
            vm.G(k, k), vm.MRB(k, k), vm.MA(k, k), vm.MA(k, k) / vm.MRB(k, k), wc(j), 2*pi/wc(j), zc(j));
    end
    % ---------------------------------------------- otter.m (only for the base meshes)
    if strcmp(mesh, 'base')
        rbP = [0; 0; 0];
        x0 = zeros(12, 1); n0 = [0; 0];
        [~, ~, Mo] = otter(x0, n0, mP, rbP, 0, 0);
        fx = @(x) otter(x, n0, mP, rbP, 0, 0);
        Jn = zeros(6, 6); Je = zeros(6, 6); e = 1e-5;
        for q = 1:6
            dx = zeros(12, 1); dx(q) = e;      xp = fx(x0 + dx); xm = fx(x0 - dx); Jn(:, q) = (xp(1:6) - xm(1:6)) / (2*e);
            dx = zeros(12, 1); dx(6 + q) = e;  xp = fx(x0 + dx); xm = fx(x0 - dx); Je(:, q) = (xp(1:6) - xm(1:6)) / (2*e);
        end
        Go = -Mo * Je; Do = -Mo * Jn;
        [~, ~, ~, w3, w4, w5, z3, z4, z5] = modalFromMDG(Mo, Do, Go, 0);
        wo = [w3 w4 w5]; zo = [z3 z4 z5];
        nabla = (55 + mP) / 1025;
        for j = 1:3
            k = K(j);
            % otter.m: MRB is about CO; added mass is diagonal "best practice"
            fprintf('  otter.m %-5s C %9.2f  M_kk %.3f  w_n %.3f  T_n %.3f s  zeta %.3f\n', dofN{j}, Go(k, k), Mo(k, k), wo(j), 2*pi/wo(j), zo(j));
            fprintf(fid, 'otter_m,%s,%g,%s,%.6g,NaN,NaN,NaN,%.6g,%.6g,%.6g\n', mesh, 55 + mP, dofN{j}, Go(k, k), wo(j), 2*pi/wo(j), zo(j));
        end
        fprintf('  otter.m  GM_T ~ %.3f m, GM_L ~ %.3f m (G44, G55 / (rho g nabla));  MSS-Capytaine GM_T %.3f, GM_L %.3f, draft %.3f m\n', ...
            Go(4,4) / (1025*9.81*nabla), Go(5,5) / (1025*9.81*nabla), v.main.GM_T, v.main.GM_L, v.main.T);
    end
end
fclose(fid);
fprintf('\nwrote %s\n', outCsv);
end
