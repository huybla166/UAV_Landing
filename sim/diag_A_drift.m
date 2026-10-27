% DIAG_A_DRIFT  Why does verification A fail at beta = 180 deg, w = 6 rad/s?
% Same regular wave, two ways to place the wave phase:
%   (a) instantaneous position N(t), E(t)             (current waveLoads 'position')
%   (b) fixed seakeeping-frame position (0, 0)        (vessel at rest on average)
mss_setup;
[vessel, ~, ~] = build_model('otter_55kg.mat', 2*pi/1.9, false);
vessel.kappa_4 = 0; vessel.kappa_5 = 0; vessel.kappa_6 = 0;
w = 6.0; a = 0.01; beta = pi; h = 0.01; T = 60;
wm = regular_wave_model(vessel, w, a);
t = (0:h:T)'; n = numel(t);
F = zeros(6,1);
for k = 1:6
    F(k) = a * (interp1(wm.raoAngles, wm.RAO_re{k}, beta) + 1i * interp1(wm.raoAngles, wm.RAO_im{k}, beta));
end
P = (-w^2 * vessel.M + 1i * w * vessel.D + vessel.G) \ F;
for mode = 1:2
    x = zeros(12,1); Y = zeros(n, 12);
    for i = 1:n
        Y(i,:) = x';
        if mode == 1, pos = x(7:8); else, pos = [0; 0]; end
        tw = waveLoads(t(i), wm, pos, 0, 0, beta, 'position');
        x = rk4(@hydroVessel, h, x, tw, vessel, 0, 0);
    end
    m = t >= 30; tt = t(m);
    Phi = [ones(size(tt)), tt, cos(w*tt), sin(w*tt)];
    c3 = Phi \ Y(m, 9);  c5 = Phi \ Y(m, 11);
    r3 = abs(c3(3) - 1i*c3(4)) / abs(P(3));  r5 = abs(c5(3) - 1i*c5(4)) / abs(P(5));
    res3 = sqrt(mean((Y(m,9) - Phi*c3).^2)) / abs(P(3));
    zc = find(diff(sign(Y(m,9) - mean(Y(m,9)))) ~= 0);
    w_obs = pi * (numel(zc) - 1) / (tt(zc(end)) - tt(zc(1)));
    labels = {'(a) instantaneous position', '(b) seakeeping frame     '};
    fprintf('%s: N(30)=%.4f m  N(60)=%.4f m  mean u(30-60)=%.4f m/s | heave ratio %.3f pitch ratio %.3f | fit residual %.2f | observed heave freq %.3f rad/s (wave %.2f)\n', ...
        labels{mode}, Y(find(t>=30,1),7), Y(end,7), mean(Y(m,1)), r3, r5, res3, w_obs, w);
end
