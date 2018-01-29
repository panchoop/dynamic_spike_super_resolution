# The infite loop was patched with a killswitch after certain number of iterations.
# Now the obtained reconstructions are too good to be real and I need to check what
push!(LOAD_PATH, "../models")
push!(LOAD_PATH, "../")
push!(LOAD_PATH, "../SparseInverseProblems/src")
using SparseInverseProblemsMod
using SuperResModels
using TestCases
using Utils
using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

## Parameters

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
minWeights = 1
maxWeights = 10

# Variability of number of particles
min_number_part = 15
max_number_part = 15

# Number of generated examples
num_trials = 10

# test case
test_case = () -> TestCases.cloud_1d_full(x_max, v_max, minWeights, maxWeights, K, tau, rand(min_number_part:max_number_part))
# WARNING: changing anything in the test_case will generate troubles if the rejection_sampling algorithm
# in Utils.js is not either updated for it, or disabled.

# Noise parameters
noises_data = linspace(0,0.1,5)
noises_data = noises_data[2:end]
noises_position = linspace(0,0.01,5)
noises_position = noises_position[2:end]

# To discard faulty reconstructions
threshold_weight = 0.1

# Randomness set for replication
srand(1)

model_static = SuperResModels.Fourier1d(f_c, x_max, filter_x, num_x)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, v_max, tau, K, num_v*K)

### Location of data folder
dataFolder = "data/1Dsimulations"

bins = np.load("../"*dataFolder*"/separationDistribBins.npy")
density = np.load("../"*dataFolder*"/separationDistribVal.npy")

### Number of points
@everywhere begin
    model_static = $model_static
    model_dynamic = $model_dynamic
    x_max = $x_max
    v_max = $v_max
    minWeights = $minWeights
    maxWeights = $maxWeights
    K = $K
    tau = $tau
    dataFolder = $dataFolder
    test_case = $test_case
    noises_data = $noises_data
    noises_position = $noises_position
    bins = $bins
    density = $density
end

println("### Beggining round ###")
    K = div(length(model_dynamic.times)-1,2)
    println("K = ", K)
    tau = maximum(model_dynamic.times)/K
    println("tau = ", tau)
    println(" Visualization of generated example" )
    (thetas, weights) =  Utils.Rejection_sampling(test_case, bins, density, K, tau, model_static.x_max)
    # Obtain the separation of the configuration of particles
    separation = Utils.separation_val(thetas,K,tau,model_static.x_max)
    # And the separationg given by the dynamic norm.
    separation_dyn = Utils.separation_norm(thetas,K,tau,model_static.x_max)
    sizeAmp = 50
#    plt.scatter(thetas[1,:], thetas[2,:], s = weights*sizeAmp, alpha = 0.8)
#    plt.xlim((0, x_max))
#    plt.ylim((-v_max, v_max))
#    plt.title(" Sep: "*string(separation)*", SepDyn: "*string(separation_dyn))
#    plt.show()
    d = dim(model_dynamic)
    println("dimension : ",d)
    results = zeros(1, 7)
    ## 	Noiseless case
    # Dynamic case
    # We generate the target measure
    target = Utils.generate_target(model_dynamic, thetas, weights)
    (results[1, 1], results[1, 2], results[1,3]) = Utils.generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
    println(" Results: dx = "*string(results[1,1])*", dv = "*string(results[1,2])*", dw = "*string(results[1,3]))
    # Inspecting the explicit reconstruction
    results2 = zeros(1,7)
    (thetas_est, weights_est) =  Utils.run_simulation_target(model_dynamic, thetas, weights, target)
    # Discard obviously defective reconstructions
    outputSize = length(thetas_est)
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    println(" Discarded reconstructions = ", outputSize- length(thetas_est))
    if (length(thetas) == length(thetas_est))
        # The output of the minimization algorithm returns an unsorted vector with the
        # estimated positions ans weights of particles. match_points is a function that
        # returns a vector that tries to match the best possible the corresponding particles
        # to the estimated particles.
        corres = Utils.match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
        dist_w = norm(weights[:] - weights_est[corres], Inf)
        # returns the infinite distance between the matched particles, in space and velocity.
        (results2[1,1], results2[1,2], results2[1,3]) = (dist_x, dist_v, dist_w)
    else
         (results2[1,1], results2[1,2], results2[1,3]) = model_dynamic.static.x_max, model_dynamic.v_max, maximum(weights)
    end
    println(" Results2: dx = "*string(results[1,1])*", dv = "*string(results[1,2])*", dw = "*string(results[1,3]))
        # Plotting reconstructions constrasted to the real solutions.
    plt.scatter(thetas[1,:], thetas[2,:], s = weights*sizeAmp, alpha = 0.8)
    plt.scatter(thetas_est[1,:], thetas_est[2,:], s=weights_est*sizeAmp/2, alpha = 0.8)
    plt.xlim((0, x_max))
    plt.ylim((-v_max, v_max))
    plt.title(" Sep: "*string(separation)*", SepDyn: "*string(separation_dyn))
    plt.show()
	# Trying the static reconstructions
    (results[1,4], results[1,5], results[1,6], results[1,7]) = Utils.generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target)
	# Checking deeply the static reconstructions.
    static_target = Utils.target_to_static(target_dynamic, model_dynamic.times)
        # Projecting the measurements to static
    k = 3
    thetas_t = Utils.to_static(thetas,model_dynamic.times[k],model_static.xmax)
        # Show the static measurements
    plt.scatter(thetas_t, zeros(length(thetas_t)), s = weights*sizeAmp, alpha = 0.8)
    plt.xlim((0,x_max))
    plt.ylim((-v_max,v_max))
    (thetas_t_est, weights_t_est) = Utils.run_simulation_target(model_static, thetas_t, weights, static_target[k])
    plt.scatter(thetas_t_est, zeros(length(thetas_t)), s = weights_t*sizeAmp/2, alpha=0.8)
    plt.show()



