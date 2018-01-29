# The infite loop was patched with a killswitch after certain number of iterations.
# Now the obtained reconstructions are too good to be real and I need to check what
push!(LOAD_PATH, "../models")
push!(LOAD_PATH, "../")
using SparseInverseProblems
using SuperResModels
using TestCases
using Utils

using PyCall
@pyimport numpy as np

type StaticParameters
    # allowed frequencies
    f_c::Int64
    # spatial max val.
    x_max::Float64
    # ???
    num_x::Int64
    # Weights on the allowed frequencies.
    filter_x::Array{Float64}
end
type DynamicParameters
    # 2K+1 is the number of time samples.
    K::Int64
    # sampling rate
    tau::Float64
    # maxium allowed speed
    v_max::Float64
    # ????
    num_v::Int64
end

### Static parameters and container for use.
f_c = 20
x_max = 1.0
num_x = 10*f_c
filter_x =  ones(2*f_c+1);
static_parameters = StaticParameters(f_c, x_max, num_x, filter_x)

model_static = SuperResModels.Fourier1d(static_parameters.f_c, static_parameters.x_max, static_parameters.filter_x, static_parameters.num_x)
### Dynamic parameters and container for use.
K = 2
tau = 1.0/(K*1.0)
v_max = 0.05
num_v = 10
dynamic_parameters = DynamicParameters(K, tau, v_max, num_v)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, dynamic_parameters.v_max, dynamic_parameters.tau, dynamic_parameters.K, dynamic_parameters.num_v*dynamic_parameters.K)
### Variabiliy of weights
minWeights = 1
maxWeights = 10
### Location of data folder
dataFolder = "data/1Dsimulations"

dynamic_parameters = DynamicParameters(K, tau, v_max, num_v)
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
    test_case = () -> TestCases.cloud_1d_full(x_max, v_max, minWeights, maxWeights, K, tau, rand(4:10))
    noises_data = linspace(0, 0.1, 5)
    noises_data = noises_data[2:end]
    noises_position = linspace(0, 0.01, 5)
    noises_position = noises_position[2:end]
    bins = $bins
    density = $density
end

num_trials = 1
srand(100000000)
println("### Beggining round ###")
    K = div(length(model_dynamic.times)-1,2)
    tau = maximum(model_dynamic.times)/K
    (thetas, weights) =  Utils.Rejection_sampling(test_case, bins, density, K, tau, model_static.x_max)
    d = dim(model_dynamic)
    results = zeros(1, 7)
    # Obtain the separation of the configuration of particles
    separation = Utils.separation_val(thetas,K,tau,model_static.x_max)
    # And the separationg given by the dynamic norm.
    separation_dyn = Utils.separation_norm(thetas,K,tau,model_static.x_max)
    ## 	Noiseless case
    # Dynamic case
    # We generate the target measure
    target = Utils.generate_target(model_dynamic, thetas, weights)
    # recomputes, adding differents levels of noise, for both dynamic and static.
    for noise_data in noises_data[2]
    	target = Utils.generate_target(model_dynamic,thetas,weights, noise_data, 0.0)
    end
    target = Utils.generate_target(model_dynamic,thetas,weights, noises_data[3], 0.0)
    println("---- Printing setup: ")
    println("thetas:",thetas)
    println("weights:",weights)
    println("target:",target)
    # Dynamic case
    println("#### going for noise_data dynamic ",noises_data[3]," ###")
    (thetas_est, weights_est) =  Utils.run_simulation_target(model_dynamic, thetas, weights, target)

println(" Simulation finished with results: ", thetas_est)
