# Static parameters
f_c = 6
x_max = 0.01
pix_size = 0.5 * x_max/f_c
model_static = Fourier1d(f_c, x_max)
# Dynamic parameters
K = 5
tau = 1.0/(K*30)
v_max = 0.05
num_v = 10
model_dynamic = DynamicFourier1d(model_static, v_max, tau, K, num_v)
# Particles
test_case = (dx, dv) -> three_points_1d(x_max, dx, dv)
# Parameter range
vec_dx = linspace(0.0001, 0.004, 20)
vec_dv = linspace(0.0, 0.03, 20)
# Number of trials
iter_mc = 1
# Do static case?
do_static = true
# Threshold to accept/reject reconstruction
threshold = 1e-4
threshold_weight = 1e-3
# Relative noise level in measure generation
noise_level_x, noise_level_v, noise_level_weights = 0.1, 0.1, 0.1
