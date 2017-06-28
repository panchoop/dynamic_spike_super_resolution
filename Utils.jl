push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
module Utils
using SparseInverseProblems
using SuperResModels
threshold_weight = 1e-2
export run_simulation, generate_and_reconstruct_all
function phi_noise(model, thetas, weights, sigma)
    return sum([weights[i] * psi_noise(model, vec(thetas[:, i]), sigma) for i in 1:size(thetas, 2)])
end
function psi_noise(model :: DynamicSuperRes, theta :: Vector{Float64}, dsec :: Float64)
    # This function computes the direct problem for a single point theta
    d = dim(model)
    return vcat([psi(model.static, theta[1:d] + t * theta[d+1:end] + t^2*dsec) for t in model.times]...)
end
function run_simulation(model, thetas, weights, noise_level=0.0, noise_position=0.0)
    if is_in_bounds(model, thetas) 
        if noise_position > 0
            target =  phi_noise(model, thetas, weights, noise_position)
        else
            target = phi(model, thetas, weights)
        end
        noise = randn(size(target))
        noise = noise_level * noise
        target = target + noise
        function callback(old_thetas, thetas, weights, output, old_obj_val)
            #evalute current OV
            new_obj_val,t = loss(LSLoss(), output - target)
            #println("gap = $(old_obj_val - new_obj_val)")
            if old_obj_val - new_obj_val < 1E-4
                return true
            end
            return false
        end
        # Inverse problem
        (thetas_est,weights_est) = ADCG(model, LSLoss(), target,sum(weights), callback=callback, max_iters=200)
        return (thetas_est, weights_est)
    else
        warn("Out of bounds !")
        return ([], [])
    end
end

function measure_noise(thetas, weights, noise_level_x, noise_level_v, noise_level_weights)
    d = div(size(thetas, 1), 2)
    num_points = size(thetas, 2)
    new_thetas = zeros(2*d, num_points)
    new_thetas[1:d, :] = thetas[1:d, :] + (1-2*rand(d, num_points)) * noise_level_x
    new_thetas[d+1:end, :] = thetas[d+1:end, :] + (1-2*rand(d, num_points)) * noise_level_v
    new_weights = weights + noise_level_weights * rand(num_points)
    return new_thetas, new_weights
end

function match_points(theta_1, theta_2)
    n_points = size(theta_1, 2)
    corres = zeros(Int, n_points)
    for i = 1:n_points
        dist = Inf
        for j = 1:n_points
            if norm(theta_1[:,i] - theta_2[:,j]) < dist
                dist = norm(theta_1[:,i] - theta_2[:,j])
                corres[i] = j
            end
        end
    end
    return corres
end

function to_static(thetas, t, x_max)
    d = div(size(thetas, 1),2)
    pts = thetas[1:d,:]
    velocities = thetas[d+1:2*d,:]
    pts_t = pts + velocities * t
    pts_t = mod(pts_t, x_max)
    return pts_t
end
function generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, noise_data, noise_position)
    d = dim(model_dynamic)
    (thetas_est, weights_est) = try run_simulation(model_dynamic, thetas, weights, noise_data, noise_position) catch ([0], [0]) end
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    if (length(thetas) == length(thetas_est))
        corres = match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
        return (dist_x, dist_v)
    end
    return model_dynamic.static.x_max, model_dynamic.static.x_max
end
function generate_and_reconstruct_static(model_static, thetas, weights, noise_data, noise_position)
    d = dim(model_static)
    (thetas_est, weights_est) = try 
        run_simulation(model_static, thetas, weights, noise_data, noise_position)
    catch
        warn("fail")
        ([0], [0])
    end
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    if (length(thetas) == length(thetas_est))
        corres = match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        return dist_x
    end
    return model_static.x_max
end
function generate_and_reconstruct_all(model_static, model_dynamic, test_case, noises_data, noises_position)
    ### This function generates a test case and verifies that the reconstruction works
    ### Both in the static and dynamic cases
    ok_dynamic = false
    ok_static = false
    (thetas, weights) = test_case()
    println(thetas)
    d = dim(model_dynamic)
    results = zeros(1 + length(noises_data) + length(noises_position), 3)
    norm = minimum([abs(thetas[1, i] - thetas[1, j]) + model_dynamic.times[end]* abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
    dx = minimum([abs(thetas[1, i] - thetas[1, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
    dv = minimum([abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
    println("dx = ", dx, ", dv = ", dv, ", norm = ", norm)
    # Dynamic case, no noise
    (results[1, 1], results[1, 2]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, 0.0, 0.0)
    # Static case, no noise
    temp = 0
    for k = 1:length(model_dynamic.times)
        thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
        temp = max(temp, generate_and_reconstruct_static(model_static, thetas_t, weights, 0.0, 0.0))
    end
    results[1, 3] = temp
    i = 2
    for noise_data in noises_data
        # Dynamic case, noisy
        (results[i, 1], results[i, 2]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, noise_data, 0.0)
        # Static case, noisy
        temp = 0.0
        for k = 1:length(model_dynamic.times)
            thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
            temp = max(temp, generate_and_reconstruct_static(model_static, thetas_t, weights, noise_data, 0.0))
        end
        results[i, 3] = temp
        i = i + 1
    end
    for noise_position in noises_position
        (results[i, 1], results[i, 2]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, 0.0, noise_position)
        results[i, 3] = 0.0
        i = i + 1
    end
    return dx, dv, norm, results
end
end
