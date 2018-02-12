# Static Parameters
f_c = 20
x_max = 1.0
num_x = 10*f_c
filter_x = ones(2*f_c+1)

# Dynamic Parameters
K  = 2
tau = 1.0/(K*1.0)
v_max = 0.5
num_v = 10

# Variability of weights
minWeights = 0.9
maxWeights = 1.1

# Variability of number of particles
min_number_part = 4
max_number_part = 10

# Number of generated examples
num_trials = 1000

# test case
test_case = () -> TestCases.cloud_1d_full(x_max, v_max, minWeights, maxWeights, K, tau, rand(min_number_part:max_number_part))
# WARNING: changing anything in the test_case will generate troubles if the rejection_sampling algorithm
# in Utils.js is not either updated for it, or disabled.

# Noise parameters
noises_data = linspace(0,0.1,5)
noises_data = noises_data[2:end]
noises_position = linspace(0,0.06,5)
noises_position = noises_position[2:end]

### Location of data folder
dataFolder = "data/1Dsimulations"
# loaded data for rejection sampling of simulations
bins = np.load(dataFolder*"/separationDistribBins.npy")
density = np.load(dataFolder*"/separationDistribVal.npy")

println(" WARNING: The rejection sampling algorithm is being used, with interval of interest: [",bins[1]," ",bins[end],"]. To change it check the separationDistribution.jl file.")

### Cases to be tested

noiseless_dynamic = true
noiseless_static = false
noise_dynamic = false
noise_static = false
curvature_static = true

cases = (noiseless_dynamic, noiseless_static, noise_dynamic, noise_static, curvature_static)
