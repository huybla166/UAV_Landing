% TIME_PARTS  Wall-clock time of each stage of run_one (Octave/MATLAB).
t_all = tic;
t = tic; mss_setup; fprintf('mss_setup (addpath)          %6.2f s\n', toc(t));

matFile = 'otter_55kg.mat';
t = tic;
cfg = hydroVesselConfig(matFile, true);
S = load(resolveHydroVesselFile(matFile), 'vessel'); vessel = S.vessel;
vessel.kappa_4 = 0; vessel.kappa_5 = 0; vessel.kappa_6 = 0;
fprintf('config + load                %6.2f s\n', toc(t));

env = cfg.environment; env.Hs = 0.15; env.w0 = 2*pi/1.9; env.beta_wave = pi/2;
env.spectrumType = 'JONSWAP'; env.spreadingFlag = 1; env.numFreqIntervals = 100; env.numDirections = 24;
env.Vc = 0; env.betaVc = 0;
rng(1);
t = tic; [env, waveModel] = waveInitialization(vessel, env); fprintf('waveInitialization           %6.2f s\n', toc(t));
t = tic; vessel = computeManeuveringModel(vessel, env.w0, [5 0.5 1], [0 0.2 0], 0); fprintf('computeManeuveringModel      %6.2f s\n', toc(t));

N = 200; h = 0.05; x = zeros(12,1);
t = tic; for i = 1:N, [tw, z] = waveForceRAO(i*h, waveModel, 0, 0, env.beta_wave); end
dt_w = toc(t) / N; fprintf('waveForceRAO   per call      %6.2f ms\n', 1e3 * dt_w);
t = tic; for i = 1:N, xd = hydroVessel(x, tw, vessel, 0, 0); end
dt_h = toc(t) / N; fprintf('hydroVessel    per call      %6.2f ms\n', 1e3 * dt_h);
t = tic; for i = 1:N, c = crossFlowDrag(vessel.main.Lpp, vessel.main.B, vessel.main.T, x(1:6), 'Hoerner'); end
dt_c = toc(t) / N; fprintf('  crossFlowDrag per call      %6.2f ms\n', 1e3 * dt_c);
clear headingAutopilot;
t = tic; for i = 1:N, tauN = headingAutopilot(0, 0, 0, 0.035, vessel.M(6,6), 1, 1, h); end
dt_a = toc(t) / N; fprintf('headingAutopilot per call    %6.2f ms\n', 1e3 * dt_a);

steps600 = 600 / h;
fprintf('\nEstimated loop time for 600 s at h = %.2f s (RAO every step): %.0f s\n', h, steps600 * (dt_w + 4*dt_h + dt_a));
fprintf('  of which wave forces %.0f s, dynamics (4x RK4) %.0f s\n', steps600*dt_w, steps600*4*dt_h);

log = randn(steps600, 22);
t = tic; f = fullfile(tempdir, 'w.csv'); dlmwrite(f, log, 'precision', '%.7g'); fprintf('dlmwrite 12000x22            %6.2f s\n', toc(t));
fprintf('TOTAL this script            %6.2f s\n', toc(t_all));
