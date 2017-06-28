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
