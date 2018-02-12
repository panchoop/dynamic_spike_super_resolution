# I wish yo find an explanation for the infinite loop thing problem. Also motivated because this issue arrise when adding noise and in Tim's code this wasn't an Issue.
# I modified to noise model so it is normalized and we can actually speak of noise strength. I believe this to be the cause, as in Tim's code there was no normalization issue. I want to quantify this missmatch and maybe conclude that the infinite loop thing arise just because there is too much noise involved.
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
maxWeights = 1

# Variability of number of particles
min_number_part = 4
max_number_part = 10

# Number of generated examples
num_trials = 1000

# test case
test_case = () -> TestCases.cloud_1d_full(x_max, v_max, minWeights, maxWeights, K, tau, rand(min_number_part:max_number_part))
# WARNING: changing anything in the test_case will generate troubles if the rejection_sampling algorithm
# in Utils.js is not either updated for it, or disabled.

model_static = SuperResModels.Fourier1d(f_c, x_max, filter_x, num_x)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, v_max, tau, K, num_v*K)

### Location of data folder
dataFolder = "data/1Dsimulations"

bins = np.load("../"*dataFolder*"/separationDistribBins.npy")
density = np.load("../"*dataFolder*"/separationDistribVal.npy")

    K = div(length(model_dynamic.times)-1,2)
    tau = maximum(model_dynamic.times)/K
    # Dynamic case
    # We generate the target measure
    results = zeros(num_trials,2)
    for i in 1:num_trials
	(thetas, weights) =  Utils.Rejection_sampling(test_case, bins, density, K, tau, model_static.x_max)
        target = Utils.generate_target(model_dynamic, thetas, weights)
        # We generate our model noise
        noise = randn(size(target))
        Normalizer = norm(target[:],2)/norm(noise[:],2)
        results[i,1] = Normalizer
	results[i,2] = length(weights)
    end

indexMax = indmax(results[:,1])
println(results)
println(" Maximum is : ", results[indexMax,1], ", with ", results[indexMax,2], " particles. ")


