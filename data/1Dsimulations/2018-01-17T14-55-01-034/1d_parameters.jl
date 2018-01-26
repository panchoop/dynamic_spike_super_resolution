# Static Parameters
f_c = 20
x_max = 1.0
num_x = 10*f_c
filter_x = ones(2*f_c+1)

# Dynamic Parameters
K  = 2
tau = 1.0/(K*1.0)
v_max = 0.05
num_v = 10

# Variability of weights
minWeights = 1
maxWeights = 10

# Variability of number of particles
min_number_part = 4
max_number_part = 10

# Number of generated examples
num_trials = 5

# test case
test_case = () -> TestCases.cloud_1d_full(x_max, v_max, minWeights, maxWeights, K, tau, rand(min_number_part:max_number_part))
# WARNING: changing anything in the test_case will generate troubles if the rejection_sampling algorithm
# in Utils.js is not either updated for it, or disabled.

# Noise parameters
noises_data = linspace(0,0.1,5)
noises_data = noises_data[2:end]
noises_position = linspace(0,0.01,5)
noises_position = noises_position[2:end]

srand(100000000)
