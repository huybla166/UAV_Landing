function wm = regular_wave_model(vessel, w, a)
% REGULAR_WAVE_MODEL  A waveModel (same fields waveLoads.m reads) for one regular,
% long-crested wave of frequency w (rad/s) and amplitude a (m). The force RAO tables
% are interpolated in frequency exactly as MSS waveInitialization does (re/im parts,
% zero-speed table), so waveLoads gives tau = Re[a * RAO_F(w, beta) * exp(i*phase)].

rao = vessel.forceRAO;
freqs = rao.w(:);
nH = numel(vessel.headings);
wm.raoType = 'force';
wm.Omega = w;
wm.mu = 0;
wm.Amp = a;
wm.S_M = NaN;
wm.g = vessel.main.g;
wm.randomPhases = 0;
wm.raoAngles = [vessel.headings(:); 2*pi];
wm.raoReferenceDepth = 0;
wm.RAO_re = cell(1, 6);
wm.RAO_im = cell(1, 6);
for DOF = 1:6
    re = zeros(1, nH); im = zeros(1, nH);
    for k = 1:nH
        A = rao.amp{DOF}(:, k, 1); P = rao.phase{DOF}(:, k, 1);
        re(k) = interp1(freqs, A .* cos(P), w, 'linear', 'extrap');
        im(k) = interp1(freqs, A .* sin(P), w, 'linear', 'extrap');
    end
    wm.RAO_re{DOF} = [re, re(1)];
    wm.RAO_im{DOF} = [im, im(1)];
end
end
