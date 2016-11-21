# Static parameters
f_c = 10
x_max = 0.01
filter = x -> exp(-(x/x_max)^2/2*f_c^2)
filter_grad = x -> -x/x_max*f_c^2 * filter(x)
pix_size = 0.5 * x_max/f_c
model_static = Conv1d(filter, filter_grad, -x_max:pix_size:2*x_max, 0:0.5*pix_size:x_max, x_max)
# Dynamic parameters
K = 1
tau = 1.0/(K*30)
v_max = 0.05
num_v = 10
model_dynamic = DynamicConv1d(model_static, v_max, tau, K, num_v)
# Particles
test_case = (dx, dv) -> three_points_1d(x_max, dx, dv)
# Parameter range
vec_dx = linspace(0.0001, 0.004, 5)
vec_dv = linspace(0.0, 0.03, 5)
# Number of trials
iter_mc = 5
# Do static case?
do_static = true
# Threshold to accept/reject reconstruction
threshold = 1e-4
threshold_weight = 1e-3
# Relative noise level in measure generation
noise_level_x, noise_level_v, noise_level_weights = 0.1, 0.1, 0.1
