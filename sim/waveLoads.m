function [tau_wave1, waveElevation] = waveLoads(t, waveModel, posNE, U, psi, beta_wave, phaseMode)
% WAVELOADS  1st-order wave elevation and 6-DOF wave forces from force RAOs for a
% surface vessel (same RAO model as MSS waveForceRAO, Fossen 2027 eqs. 10.83, 10.96),
% with two choices for the wave-component phase:
%
%   phaseMode = 'position' (default, used for the dataset)
%       phase_ij = w_i t - k_i (N cos(chi_j) + E sin(chi_j)) + eps_ij,
%       k_i = w_i^2 / g (deep water), chi_j = beta_wave + mu_j (propagation direction,
%       measured like a heading; beta_wave = 0 travels along psi = 0, i.e. following sea).
%       The encounter frequency follows from the vessel's actual track, so it stays
%       correct when speed or heading vary in time.
%
%   phaseMode = 'mss'  (reproduces MSS waveForceRAO exactly; for verification only)
%       phase_ij = |w_i - k_i U cos(beta_rel + mu_j)| t + eps_ij.
%       Exact only for constant U and psi: a change dW in encounter frequency shifts
%       the phase by dW * t, so long runs with yaw/speed oscillations get spurious
%       instantaneous frequencies W_e + t dW_e/dt.
%
% Inputs: t (s); waveModel from MSS waveInitialization; posNE = [N; E] (m) of the CG;
%         U (m/s) used only in 'mss' mode; psi (rad); beta_wave (rad).
% The directional RAO interpolation is linear in direction on the uniform RAO heading
% grid (identical to interp1 'linear', done by indexing for speed).

if nargin < 7 || isempty(phaseMode), phaseMode = 'position'; end
if ~strcmp(waveModel.raoType, 'force')
    error('waveLoads:InvalidWaveModel', 'waveModel must be initialized with force RAOs.');
end

Omega = waveModel.Omega;          % Nf x 1
mu    = waveModel.mu(:);          % Nd x 1
Amp   = waveModel.Amp;            % Nf x Nd
k     = Omega.^2 / waveModel.g;   % Nf x 1

% ------------------------------------------------ directional RAO (linear in direction)
ang = waveModel.raoAngles(:);                     % 0 .. 2*pi, closed
dAng = ang(2) - ang(1);
if any(abs(diff(ang) - dAng) > 1e-9)
    error('waveLoads:NonUniformHeadings', 'RAO heading grid must be uniform.');
end
beta_rel = beta_wave - psi;
beta_RAO = mod(beta_rel + mu, 2*pi);              % Nd x 1
pos = beta_RAO / dAng;
i0 = min(max(floor(pos) + 1, 1), numel(ang) - 1); % left node
wgt = (pos - (i0 - 1))';                          % 1 x Nd
i0 = i0';

% ------------------------------------------------ component phases
switch phaseMode
    case 'position'
        chi = (beta_wave + mu)';                  % 1 x Nd
        phase = Omega * t - k * (posNE(1) * cos(chi) + posNE(2) * sin(chi)) + waveModel.randomPhases;
    case 'mss'
        Omega_e = abs(Omega - k * U .* cos(beta_RAO'));
        phase = Omega_e * t + waveModel.randomPhases;
    otherwise
        error('waveLoads:phaseMode', 'phaseMode must be ''position'' or ''mss''.');
end
c = Amp .* cos(phase);
s = Amp .* sin(phase);

waveElevation = sum(c(:));
if isfield(waveModel, 'RAOstack')
    % all 6 DOFs at once: RAOstack = cat(3, RAO_re{:}, RAO_im{:}), Nf x (Nh+1) x 12
    S = waveModel.RAOstack;
    w1 = 1 - wgt;
    P = S(:, i0, :) .* w1 + S(:, i0 + 1, :) .* wgt;            % Nf x Nd x 12
    P = reshape(P, [], 12);
    tau_wave1 = (c(:)' * P(:, 1:6) - s(:)' * P(:, 7:12))';
    return
end
tau_wave1 = zeros(6, 1);
for DOF = 1:6
    R = waveModel.RAO_re{DOF};  I = waveModel.RAO_im{DOF};     % Nf x (Nh+1)
    re = R(:, i0) .* (1 - wgt) + R(:, i0 + 1) .* wgt;
    im = I(:, i0) .* (1 - wgt) + I(:, i0 + 1) .* wgt;
    tau_wave1(DOF) = sum(sum(re .* c - im .* s));
end
end
