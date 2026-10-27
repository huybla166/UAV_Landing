function x = rk4u(dyn, h, x, u0, um, u1, varargin)
% RK4U  Classical RK4 step with a time-varying input: u0 = u(t), um = u(t + h/2), u1 = u(t + h).
%   x = rk4u(@f, h, x, u0, um, u1, p1, p2, ...) integrates x' = f(x, u, p1, p2, ...).
% MSS rk4.m holds the input constant over the step (zero-order hold), which delays a wave load
% of frequency w by h/2 (phase -w h/2; sim/verify_E_memory.m measured it to 0.1 deg).
k1 = dyn(x, u0, varargin{:});
k2 = dyn(x + 0.5 * h * k1, um, varargin{:});
k3 = dyn(x + 0.5 * h * k2, um, varargin{:});
k4 = dyn(x + h * k3, u1, varargin{:});
x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end
