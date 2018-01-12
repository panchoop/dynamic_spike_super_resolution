push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
using SparseInverseProblems
using SuperResModels
using TestCases
using Utils
# Containers to handle parameters
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
using PyCall
@pyimport numpy as np
bins = np.load(dataFolder*"/separationDistribBins.npy")
density = np.load(dataFolder*"/separationDistribVal.npy")

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

num_trials = 10
results = pmap(x -> Utils.generate_and_reconstruct_all(model_static, model_dynamic, bins, density, test_case, noises_data, noises_position), 1:num_trials)
results_array = vcat([result[3] for result in results]...)
separations_array = vcat([result[1] for result in results]...)
separation_dyn_array = vcat([result[2] for result in results]...)

#
now_str = string(now())
now_str = replace(now_str, ":", "-")
now_str = replace(now_str, ".", "-")

# Save results
saveFolder = dataFolder*"/"*now_str
mkdir(saveFolder)
println(" The results are : ", results_array)
println(" Simulations finished, we save the files into folder :/" *saveFolder )

np.save(saveFolder*"/datanoise.npy", noises_data)
np.save(saveFolder*"/positionnoise.npy", noises_position)
np.save(saveFolder*"/separationDynamic.npy", separation_dyn_array)
np.save(saveFolder*"/separations.npy", separations_array)
np.save(saveFolder*"/results.npy", results_array)
