% PROFILE_RUN  Where does run_one spend its time? (Octave/MATLAB)
%   octave-cli --eval "cd('D:/USV-landing-prediction/sim'); profile_run"
mss_setup;
outDir = fullfile(tempdir, 'mss_profile');
profile clear; profile on;
run_one('otter_55kg.mat', 0.15, 1.9, 90, 0, 1, 15, 0.05, outDir, struct('verbose', false));
profile off;
p = profile('info');
ft = p.FunctionTable;
[~, idx] = sort([ft.TotalTime], 'descend');
fprintf('%-34s %10s %10s %8s\n', 'function', 'total[s]', 'self[s]', 'calls');
for k = idx(1:min(25, numel(idx)))
    ch = ft(k).Children;
    childTime = 0;
    if ~isempty(ch), childTime = sum([ft(ch).TotalTime]); end
    fprintf('%-34s %10.2f %10.2f %8d\n', ft(k).FunctionName, ft(k).TotalTime, ft(k).TotalTime - childTime, ft(k).NumCalls);
end
