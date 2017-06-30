push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
using SparseInverseProblems
using SuperResModels
using TestCases
using Utils
type StaticParameters
    f_c::Int64
    x_max::Float64
    num_x::Int64
    filter_x::Array{Float64}
end
type DynamicParameters
    K::Int64
    tau::Float64
    v_max::Float64
    num_v::Int64
end
# Static parameters
f_c = 20
x_max = 0.01
num_x = 10*f_c
filter_x =  ones(2*f_c+1);
static_parameters = StaticParameters(f_c, x_max, num_x, filter_x)
model_static = SuperResModels.Fourier1d(static_parameters.f_c, static_parameters.x_max, static_parameters.filter_x, static_parameters.num_x)
# Dynamic parameters
K = 2
tau = 1.0/(K*30)
v_max = 0.05
num_v = 10
dynamic_parameters = DynamicParameters(K, tau, v_max, num_v)
model_dynamic = SuperResModels.DynamicFourier1d(model_static, dynamic_parameters.v_max, dynamic_parameters.tau, dynamic_parameters.K, dynamic_parameters.num_v*dynamic_parameters.K)
# Number of points
@everywhere begin
    model_static = $model_static
    model_dynamic = $model_dynamic
    x_max = $x_max
    v_max = $v_max
    test_case = () -> TestCases.cloud_1d(x_max, v_max, rand(3:10))
    noises_data = linspace(0, 0.1, 5)
    noises_data = noises_data[2:end]
    noises_position = linspace(0, 3e-3, 5)
    noises_position = noises_position[2:end]
end

num_trials = 1000
results = pmap(x -> Utils.generate_and_reconstruct_all(model_static, model_dynamic, test_case, noises_data, noises_position), 1:num_trials)
results_array = vcat([result[4] for result in results]...)
norm_array = vcat([result[3] for result in results])
dx_array = vcat([result[1] for result in results]...)
dv_array = vcat([result[2] for result in results])

# Save results
using PyCall
@pyimport numpy as np
np.save("datanoise.npy", noises_data)
np.save("positionnoise.npy", noises_position)
np.save("norm.npy", norm_array)
np.save("dx.npy", dx_array)
np.save("dv.npy", dv_array)
np.save("results.npy", results_array)
