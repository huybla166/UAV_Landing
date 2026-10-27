function mssRoot = mss_setup(mssRoot)
% MSS_SETUP  Add the MSS toolbox (and this sim folder) to the path, headless.
%   mss_setup()            uses the MSS_ROOT environment variable or D:\tools\MSS
%   mss_setup('C:\x\MSS')  uses an explicit directory
%
% Works in GNU Octave and MATLAB. No GUI, no graphics toolkit required.

if nargin < 1 || isempty(mssRoot)
    mssRoot = getenv('MSS_ROOT');
    if isempty(mssRoot)
        mssRoot = 'D:\tools\MSS';
    end
end
if exist(fullfile(mssRoot, 'CRAFT', 'hydroVessel.m'), 'file') ~= 2
    error('mss_setup:notFound', 'MSS not found at %s (expected CRAFT/hydroVessel.m).', mssRoot);
end

addpath(genpath(mssRoot));
addpath(fileparts(mfilename('fullpath')));   % this sim/ folder

if exist('OCTAVE_VERSION', 'builtin')
    % Keep Octave quiet about MATLAB-only syntax inside MSS; we do not use the GUI.
    warning('off', 'Octave:shadowed-function');
    warning('off', 'Octave:language-extension');
end

fprintf('MSS path set: %s\n', mssRoot);
end
