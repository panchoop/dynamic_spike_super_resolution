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
### Number of points
@everywhere begin
    model_static = $model_static
    model_dynamic = $model_dynamic
    x_max = $x_max
    v_max = $v_max
    test_case = () -> TestCases.cloud_1d_full(x_max, v_max, 0.1,10,K, tau, 4)
    noises_data = [0.01, 0.05, 0.1]
    #noises_data = linspace(0, 0.1, 5)
    #noises_data = noises_data[2:end]
    noises_position = [0.0001, 0.001]
    #noises_position = linspace(0, 2.5e-3, 5)
    #noises_position = noises_position[2:end]
end


# Replicating the process of Utils.generate_and_reconsturct_all step by step.
#results = [Utils.generate_and_reconstruct_all(model_static, model_dynamic, test_case, noises_data, noises_position)]
#println(" the results array is:", results )
#println(" with size : ", size(results))

(thetas,weights) = test_case()
# We plot the generated particles and their respective weights.
# This is plotted in space x velocity plane, with the size of the particles
# indicating the weights of the respective particles.
sizeAmp = 50
plt.scatter(thetas[1,:], thetas[2,:], s = weights*sizeAmp, alpha = 0.8)
plt.xlim((0, x_max))
plt.ylim((-v_max, v_max))
#plt.show()

results = zeros(1 + length(noises_data) + length(noises_position), 7)
# Obtain the dynamical norm of the configuration of particles
norm = minimum([abs(thetas[1, i] - thetas[1, j]) + model_dynamic.times[end]* abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
norm2 = Utils.separation_val(thetas,K,tau,x_max)
# Obtain the minimum separation in space
dx = minimum([abs(thetas[1, i] - thetas[1, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
# Obtain the minimum separation in velocity
dv = minimum([abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])

println(" Results: ",results)
println(" Norm: ", norm)
println(" Norm2: ", norm2)
println(" dx infinite norm: ", dx)
println(" dv infinite norm: ", dv)

# Dismembering the function Utils.generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, 0.0,0.0)
(thetas_est, weights_est) = Utils.run_simulation(model_dynamic, thetas, weights)
# See the estimated particles and weights, compared to the real ones
plt.scatter(thetas[1,:], thetas[2,:], s = weights*sizeAmp, alpha = 0.8)
plt.scatter(thetas_est[1,:], thetas_est[2,:], s=weights_est*sizeAmp/3, alpha = 0.8)
plt.xlim((0, x_max))
plt.ylim((-v_max, v_max))
plt.show()

# Now we obtain the difference with the actual solution

# We set a threshold weight in which we discard the generated particles.
(dist_x, dist_v, dist_w) = Utils.generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, 0.0, 0.0)
println("Recons Dx = ", dist_x)
println("Recons Dv = ", dist_v)
println("Recons Dw = ", dist_w)

(dx, dv, norm, results) = Utils.generate_and_reconstruct_all(model_static, model_dynamic, test_case, noises_data, noises_position)
println("The noiseless results are: ", results[1,:])
println("Measurement noise results :", results[2,:])
println("position noise results :", results[3:end,:])
# We test the whole results function
