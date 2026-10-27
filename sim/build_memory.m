function vessel = build_memory(vessel, cfg, verbose)
% BUILD_MEMORY  Add a Cummins fluid-memory model to an MSS vessel built by build_model.
%
% Cummins equation (zero speed), Fossen (2021/2027) ch. 5, Perez & Fossen (2009):
%   (MRB + A_inf) nu_dot + mu + Dv nu + G eta = tau,   mu = int_0^t K(t - s) nu(s) ds
%   K(jw) = B(w) + jw (A(w) - A_inf)
% Each non-negligible K_ij(jw) is fitted by a stable rational transfer function with the MSS
% FDI routine ident_retardation_FD (frequency-domain LS, iterative quasi-linear, order
% increased until R2 of both A and B >= 0.99), realised in state space and stacked:
%   xr_dot = Ar xr + Br nu,   mu = Cr xr.
%
% Added fields: vessel.memory.{Ar,Br,Cr,n,report}, vessel.MAinf, vessel.Minv_inf, vessel.Dv,
% vessel.Dref (6x1, scale of the quadratic damping), vessel.wn (BEM natural freqs 3..5).
% Dv (viscous/maneuvering, constant): surge/sway/yaw from the aperiodic time constants T_126
% with the zero-frequency added mass, Dv_ii = (MRB + A(0))_ii / T_i (radiation damping vanishes
% at w -> 0); heave/roll/pitch: 2 * delta_zeta_k * sqrt((MRB + A(w_n))_kk * G_kk).
% Quadratic damping (kappa) acts on Dref_k = B_kk(w_n) + Dv_kk, i.e. the same magnitude as
% MSS hydroVessel's D_kk (1 + kappa_k |nu_k|), but the linear radiation part now has memory.
%
% Results are cached in sim/cache/<vessel>_memory_v2_dz<dz3>_<dz4>_<dz5>.mat, one file per vessel and
% viscous damping setting, so batch workers that mix settings never overwrite each other's cache.
% Build every cache once before a parallel batch (sim/prebuild_memory.m).

if nargin < 3, verbose = true; end
pkg load control
pkg load signal
cacheDir = fullfile(fileparts(mfilename('fullpath')), 'cache');
if ~exist(cacheDir, 'dir'), mkdir(cacheDir); end
[~, base] = fileparts(vessel.matFile);
T_126 = cfg.damping.T_126; dz = cfg.damping.delta_zeta_345;
cacheFile = fullfile(cacheDir, sprintf('%s_memory_v3_dz%g_%g_%g.mat', base, dz(1), dz(2), dz(3)));
key = [T_126(:); dz(:); 3];           % 3 = fit version (passive diagonals, bounded beyond the band)

if exist(cacheFile, 'file') == 2
    C = load(cacheFile);
    if isequal(C.key, key)
        vessel.memory = C.memory; vessel.MAinf = C.MAinf; vessel.Minv_inf = C.Minv_inf;
        vessel.Dv = C.Dv; vessel.Dref = C.Dref; vessel.wn = C.wn;
        if verbose, fprintf('  fluid memory: loaded %s (%d states)\n', cacheFile, C.memory.n); end
        return
    end
end

fr = vessel.freqs(:);
fin = find(fr > 0 & fr < 10);
W = fr(fin);
Ainf = vessel.A(:, :, end);           % MSS convention: freqs(end) = 10 is infinity
A0 = vessel.A(:, :, 1);               % freqs(1) = 0
MRB = vessel.MRB;
G = vessel.G;

% ---------------------------------------------------------------- natural freqs, Dv, Dref
wn = NaN(6, 1);
for k = 3:5
    Akk = squeeze(vessel.A(k, k, fin)); w = 6;
    for it = 1:200
        w_new = sqrt(G(k, k) / (MRB(k, k) + interp1(W, Akk, w, 'linear', 'extrap')));
        if abs(w_new - w) < 1e-10, break; end
        w = 0.5 * (w + w_new);
    end
    wn(k) = w;
end
Dv = zeros(6);
for kk = 1:3
    i = [1 2 6]; i = i(kk);
    Dv(i, i) = (MRB(i, i) + A0(i, i)) / T_126(kk);
end
for kk = 1:3
    k = kk + 2;
    Ak = interp1(W, squeeze(vessel.A(k, k, fin)), wn(k), 'linear', 'extrap');
    Dv(k, k) = 2 * dz(kk) * sqrt((MRB(k, k) + Ak) * G(k, k));
end
Dref = zeros(6, 1);
for k = 3:5
    Dref(k) = interp1(W, squeeze(vessel.B(k, k, fin)), wn(k), 'linear', 'extrap') + Dv(k, k);
end
Dref(6) = Dv(6, 6);

% ---------------------------------------------------------------- identify K_ij
% Couplings below 1 % of sqrt(max B_ii * max B_jj) are numerical noise (e.g. longitudinal-lateral
% terms of a port-starboard symmetric hull, ~0.1-0.2 %) and are not modelled.
% Modes faster than pmax = 3 * max(W) lie far outside the data and would make RK4 (h = 0.05 s)
% unstable; they are residualized (treated as instantaneous): their static gain -r/p becomes a
% constant feedthrough Dr, which preserves K(0) = 0 and the response inside the data range.
pmax = 3 * max(W);
Bscale = zeros(6, 1);
for k = 1:6, Bscale(k) = max(abs(squeeze(vessel.B(k, k, fin)))); end
% v2: only passive diagonal fits are accepted (Re K_ii(jw) >= 0 on 0-60 rad/s). The unconstrained
% v1 fit was free above max(W) and gave Re K_55 < 0 at 13-30 rad/s (non-passive; for otter_80kg an
% unstable linear vessel model), which matters at forward speed in head seas where the encounter
% frequency reaches 28 rad/s. (Anchoring the fit with synthetic high-frequency points was tried
% and rejected: the weighted least squares of fit_siso_fresp let the anchors dominate the band.)
w_pass = linspace(0.05, 60, 1200)';
blocksA = {}; blocksB = {}; blocksC = {}; report = {}; Dr = zeros(6);
for i = 1:6
    for j = i:6
        Bij = squeeze(vessel.B(i, j, fin)); Aij = squeeze(vessel.A(i, j, fin));
        if max(abs(Bij)) < 1e-2 * sqrt(Bscale(i) * Bscale(j)), continue; end
        Kw = Bij + 1i * W .* (Aij - Ainf(i, j));
        tol = 1e-3 * max(abs(Bij));
        best = []; bestScore = -Inf;
        for ord = 2:12
            [Num, Den] = fit_pair(W, Kw, ord);
            [R2A, R2B] = fit_quality(polyval(Num, 1i * W) ./ polyval(Den, 1i * W), W, Aij, Bij, Ainf(i, j));
            % a diagonal kernel must be passive: Re K_ii(jw) >= 0 on 0-60 rad/s (within tolerance);
            % every kernel must stay bounded beyond the data: |K_ij(jw)| above max(W) may not exceed
            % 1.1 x its maximum inside the band (the fit is free there and can grow or resonate)
            Kp = polyval(Num, 1i * w_pass) ./ polyval(Den, 1i * w_pass);
            bounded = max(abs(Kp(w_pass > max(W)))) <= 1.1 * max(abs(Kw));
            passive = bounded && ((i ~= j) || min(real(Kp)) >= -tol);
            if passive && min(R2A, R2B) > bestScore
                best = struct('Num', Num, 'Den', Den, 'ord', ord, 'R2A', R2A, 'R2B', R2B); bestScore = min(R2A, R2B);
            end
            if passive && R2A >= 0.99 && R2B >= 0.99, break; end
        end
        if isempty(best), error('build_memory: no passive fit for K%d%d', i, j); end
        % modal split: slow poles kept as states, fast poles residualized into a feedthrough
        p = roots(best.Den);
        r = zeros(size(p));
        dDen = polyder(best.Den);
        for q = 1:numel(p), r(q) = polyval(best.Num, p(q)) / polyval(dDen, p(q)); end
        fast = abs(p) > pmax;
        dfast = real(sum(-r(fast) ./ p(fast)));
        ps = p(~fast); rs = r(~fast);
        den_s = real(poly(ps));
        num_s = zeros(1, numel(ps));
        for q = 1:numel(ps), num_s = num_s + rs(q) * poly(ps([1:q-1, q+1:end])); end
        num_s = real(num_s);
        Kh = polyval(num_s, 1i * W) ./ polyval(den_s, 1i * W) + dfast;
        [R2Ar, R2Br] = fit_quality(Kh, W, Aij, Bij, Ainf(i, j));
        if i == j
            minRe = min(real(polyval(num_s, 1i * w_pass) ./ polyval(den_s, 1i * w_pass) + dfast));
            if minRe < -tol, error('build_memory: K%d%d not passive after residualization (%.3g)', i, j, minRe); end
        end
        [a, b, c, ~] = tf2ss(num_s, den_s);
        % mu_i += K_ij * nu_j  and, by symmetry of the zero-speed kernel, mu_j += K_ij * nu_i
        targets = [i j; j i];
        if i == j, targets = [i j]; end
        for t = 1:size(targets, 1)
            blocksA{end+1} = a;
            bb = zeros(size(a, 1), 6); bb(:, targets(t, 2)) = b; blocksB{end+1} = bb;
            cc = zeros(6, size(a, 1)); cc(targets(t, 1), :) = c; blocksC{end+1} = cc;
            Dr(targets(t, 1), targets(t, 2)) = Dr(targets(t, 1), targets(t, 2)) + dfast;
        end
        report{end+1} = sprintf(['K%d%d: order %d -> %d states (%d fast poles residualized), ' ...
            'R2A %.4f, R2B %.4f (before reduction %.4f, %.4f), fastest kept pole %.1f rad/s'], ...
            i, j, best.ord, numel(ps), sum(fast), R2Ar, R2Br, best.R2A, best.R2B, max([abs(ps); 0]));
        if verbose, fprintf('  %s\n', report{end}); end
    end
end
Ar = blkdiag(blocksA{:});
Br = vertcat(blocksB{:});
Cr = horzcat(blocksC{:});
ev = eig(Ar);
if any(real(ev) >= 0), error('build_memory: unstable memory model'); end
MAinf = Ainf; Minv_inf = inv(MRB + Ainf);
% passivity of the full kernel (Hermitian part of K(jw) >= 0) and stability of the linear system
% d/dt [eta; nu; xr] with Dv, G (the three position integrators have eigenvalue 0 exactly)
n = size(Ar, 1); minEigK = Inf;
for w = [w_pass(1:20:end); 9.5; 13; 20]'
    Kw = Cr * ((1i * w * eye(n) - Ar) \ Br) + Dr;
    minEigK = min(minEigK, min(eig(0.5 * (Kw + Kw'))));
end
Alti = zeros(12 + n);
Alti(1:6, 7:12) = eye(6);
Alti(7:12, 1:6) = -Minv_inf * G; Alti(7:12, 7:12) = -Minv_inf * (Dv + Dr); Alti(7:12, 13:end) = -Minv_inf * Cr;
Alti(13:end, 7:12) = Br; Alti(13:end, 13:end) = Ar;
evL = eig(Alti);
maxReL = max(real(evL(abs(evL) > 1e-9)));
if maxReL > 1e-6, error('build_memory: linear vessel model unstable (max Re eig %.3g)', maxReL); end
report{end+1} = sprintf('passivity: min eig Re K(jw) on 0-60 rad/s = %.3g (scale max B = %.3g); linear system max Re eig = %.3g', ...
    minEigK, max(Bscale), maxReL);
if verbose, fprintf('  %s\n', report{end}); end
memory = struct('Ar', Ar, 'Br', Br, 'Cr', Cr, 'Dr', Dr, 'n', size(Ar, 1), 'report', {report}, ...
    'maxPole', max(abs(ev)), 'pmax', pmax, 'minEigReK', minEigK, 'maxReEigLinear', maxReL, 'fitVersion', 3);
save(cacheFile, 'key', 'memory', 'MAinf', 'Minv_inf', 'Dv', 'Dref', 'wn');
vessel.memory = memory; vessel.MAinf = MAinf; vessel.Minv_inf = Minv_inf;
vessel.Dv = Dv; vessel.Dref = Dref; vessel.wn = wn;
if verbose
    fprintf('  fluid memory: %d states, fastest pole %.1f rad/s; saved %s\n', memory.n, memory.maxPole, cacheFile);
end
end

function [R2A, R2B] = fit_quality(Kh, W, Aij, Bij, Ainf_ij)
Bh = real(Kh); Ah = imag(Kh) ./ W + Ainf_ij;
R2B = 1 - sum((Bij - Bh).^2) / sum((Bij - mean(Bij)).^2);
R2A = 1 - sum((Aij - Ah).^2) / sum((Aij - mean(Aij)).^2);
end

function [Num, Den] = fit_pair(W, Kw, ord)
% MSS FDI fit (ident_retardation_FD core: K(s) = s * P(s)/Q(s), deg P = ord - 2, deg Q = ord),
% then any unstable pole is reflected into the left half plane and the numerator re-fitted.
Scale = max(abs(Kw));
Fresp = (Kw / Scale) ./ (1i * W);
[NumP, Den] = fit_siso_fresp(W, Fresp, [], ord - 2, ord, 2, 20);
r = roots(Den);
if any(real(r) >= 0)
    r(real(r) >= 0) = -abs(real(r(real(r) >= 0))) + 1i * imag(r(real(r) >= 0));
    Den = real(poly(r)) * Den(1);
    Q = polyval(Den, 1i * W);
    V = zeros(numel(W), ord - 1);
    for k = 1:ord - 1, V(:, k) = (1i * W).^(ord - 1 - k); end   % P(s) coefficients, highest first
    y = Fresp .* Q;
    NumP = real([real(V); imag(V)] \ [real(y); imag(y)]).';
end
Num = Scale * [NumP 0];
end
