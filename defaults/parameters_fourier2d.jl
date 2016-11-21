# Static parameters
f_min = 20
f_c_x = 5
f_c_z = 30
x_max = 0.01
filter_x =  ones(2*f_c_x+1);
filter_z = ones(2*f_c_x+1);
#filter_z = [zeros(f_c_z+f_min); ones(f_c_z-f_min+1)]
filter2d = filter_x * filter_z'
n_approx_x = 10*f_c_x
n_approx_z = 10*f_c_x
model_static = Fourier2d(x_max, filter2d, n_approx_x, n_approx_z)
# Dynamic parameters
K = 2
tau = 1.0/(K*30)
v_max = 0.05
num_v = 10
model_dynamic = DynamicFourier2d(model_static, v_max, tau, K, num_v)
# Particles
test_case = (dx, dv) -> three_points_2d(x_max, dx, dv)
# Parameter range
vec_dx = linspace(0.0001, 0.004, 10)
vec_dv = linspace(0.0, 0.03, 10)
# Number of trials
iter_mc = 1
# Do static case?
do_static = true
# Threshold to accept/reject reconstruction
threshold = 1e-4
threshold_weight = 1e-3 # threshold to discard points
# Relative noise level in measure generation
noise_level_x, noise_level_v, noise_level_weights = 0.1, 0.1, 0.1
