push!(LOAD_PATH, "../models")
using SuperResModels
using SparseInverseProblems

@everywhere begin
    include("../ufus_parameters.jl")
    parameters = SuperResModels.Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
    model_static = SuperResModels.Conv2d(parameters)
    dynamic_parameters = SuperResModels.DynamicConv2dParameters(K, tau, v_max, 20)
    model_dynamic = SuperResModels.DynamicConv2d(model_static, dynamic_parameters)
    n_x = model_static.n_x
    n_y = model_static.n_y
end

# Get time of script start
now_str = string(now())
now_str = replace(now_str, ":", "-")
now_str = replace(now_str, ".", "-")

# L2 norm single particle
thetas = reshape([x_max/2; x_max/2], 2, 1)
weights = [1.0]
single_particle_norm = norm(phi(model_static, thetas, weights), lp_norm)
@everywhere single_particle_norm = $single_particle_norm
println("norm = ", single_particle_norm)

# Test dynamic
println("Testing dynamic...")
particles_m = [x_max/4]
#particles_p = [3*x_max/4]
thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'; 0.0; -v_max/2])
              #[(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'; 0.0; v_max/2])
weights = ones(1)
target = phi(model_dynamic, thetas, weights)
function callback(old_thetas, thetas, weights, output, old_obj_val)
    #evalute current OV
    new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
    #println("gap = $(old_obj_val - new_obj_val)")
    if old_obj_val - new_obj_val < 1E-4
        return true
    end
    return false
end
(thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, 2.0, callback=callback, max_iters=200)
println("theta = ", thetas, ", weights = ", weights)
println("theta_est = ", thetas_est, ", weights_est = ", weights_est)
