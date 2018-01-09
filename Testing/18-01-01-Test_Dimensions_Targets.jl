# Test to rewrite the reconstructions functions with inputs the data instead of some noise level.

push!(LOAD_PATH, "../models")
push!(LOAD_PATH, "../")
using SparseInverseProblems
using SuperResModels
using TestCases
using Utils

using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

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
# the maximum velocity on the domain is given by x_max/2 + K tau v_max = 1
v_max = 0.5
num_v = 10
dynamic_parameters = DynamicParameters(K, tau, v_max, num_v)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, dynamic_parameters.v_max, dynamic_parameters.tau, dynamic_parameters.K, dynamic_parameters.num_v*dynamic_parameters.K)

# We check the dimensions of each case.
(thetas, weights) = TestCases.cloud_1d_full(x_max, v_max, 0.1,10,K, tau, 1)
println(" thetas = ", thetas)
println(" weights = ", weights)

# Size of the dynamic reconstruction
target_dynamic = Utils.generate_target(model_dynamic, thetas, weights)
println(" Size of the dynamic reconstruction is: ", size(target_dynamic) ) 

# Size of a single static reconstruction
target_static = Utils.target_to_static(target_dynamic, model_dynamic.times)
#for k = 1: length(model_dynamic.times)
#thetas_t = Utils.to_static(thetas, model_dynamic.times[k], model_static.x_max)
#push!(target_static, Utils.generate_target(model_static,thetas_t, weights))
#println(" Size of a single static reconstruction is: ", size(target_static[k]) )
#end


res = zeros(size(target_dynamic))
for k = 1:length(model_dynamic.times)
    for j = 1:length(target_static[1])
	index = (k-1)*length(target_static[1]) + j;
        res[index] = target_dynamic[index] - target_static[k][j]
	println("res = ", res[(k-1)*length(target_static[1]) + j], "target static = ", target_static[k][j], "target dynamic = ", target_dynamic[(k-1)*length(target_static[1]) + j] )
    end
end
println(" The difference is: ", norm(res)/norm(target_dynamic))
#println(" Target dynamic = ", target_dynamic)
#println(" Target static = ", target_static)


