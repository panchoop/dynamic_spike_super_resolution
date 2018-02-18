push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
push!(LOAD_PATH, "./SparseInverseProblems/src")
using SparseInverseProblemsMod
using SuperResModels
using TestCases
using Utils
using PyCall
@pyimport numpy as np
include("1d_parameters.jl")

# To see the progress in pmap
# To use PmapProgressMeter you need to clone it manually
# do in Julia: Pkg.clone("https://github.com/slundberg/PmapProgressMeter.jl")
# if this is no longer existent, uncoment the alternative pmap in this code.
@everywhere using ProgressMeter
@everywhere using PmapProgressMeter

model_static = SuperResModels.Fourier1d(f_c, x_max, filter_x, num_x)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, v_max, tau, K, num_v*K)
if K*tau != 1.0
    warning(" The function match_points < Utils considers K*tau = 1 to set distance in Omega, change this if it's not the case")
end


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
    cases = $cases
end

results = pmap(x -> begin sleep(1) ; Utils.generate_and_reconstruct_all(model_static, model_dynamic, bins, density, algIter, test_case, noises_data, noises_position, cases) end, Progress(num_trials), 1:num_trials)
# results = pmap(x -> Utils.generate_and_reconstruct_all(model_static, model_dynamic, bins, density, test_case, noises_data, noises_position, cases), 1:num_trials)
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
cp("1d_parameters.jl", string(saveFolder, "/1d_parameters.jl"))
cd(saveFolder)

println(" Simulations finished, we save the files into folder :/" *saveFolder )

np.save("datanoise", noises_data)
np.save("positionnoise.npy", noises_position)
np.save("separationDynamic.npy", separation_dyn_array)
np.save("separations.npy", separations_array)
np.save("results.npy", results_array)
