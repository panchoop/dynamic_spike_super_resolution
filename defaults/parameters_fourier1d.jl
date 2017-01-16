# Static parameters
f_c = 6
x_max = 0.01
v_max = 0.1
pix_size = 0.5 * x_max/f_c
num_x = 10*f_c
filter_x =  ones(2*f_c+1);
model_static = Fourier1d(f_c, x_max, filter_x, num_x)
# Dynamic parameters
K = 2
tau = 1.0/(K*30)
v_max = 0.05
num_v = 10*K
model_dynamic = DynamicFourier1d(model_static, v_max, tau, K, num_v)
# Model with "all frequencies"
filter_z = ones(2*K+1);
filter2d = filter_x * filter_z'
n_approx_x = 10*f_c
model_allfreqs = Fourier2d(x_max, x_max/tau, filter2d, num_x, num_v)
# Particles
test_case = (dx, dv) -> three_points_1d(x_max, dx, dv)
# Parameter range
vec_dx = linspace(0.0001, 0.004, 10)
vec_dv = linspace(0.00, 0.03, 10)
# Number of trials
iter_mc = 1
# Threshold to accept/reject reconstruction
threshold = 1e-4
threshold_weight = 1e-1
# Relative noise level in measure generation
noise_level_x, noise_level_v, noise_level_weights = 0.0, 0.0, 0.0
# Do static case?
do_static = true
do_allfreqs = true
