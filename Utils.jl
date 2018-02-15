push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
module Utils
using SparseInverseProblemsMod
using SuperResModels
threshold_weight = 1e-1
export run_simulation, generate_and_reconstruct_all

using PyCall
@pyimport numpy as np

function phi_noise(model, thetas, weights, sigma)
    return sum([weights[i] * psi_noise(model, vec(thetas[:, i]), sigma) for i in 1:size(thetas, 2)])
end
function psi_noise(model :: DynamicSuperRes, theta :: Vector{Float64}, dsec :: Float64)
    # This function computes the direct problem for a single point theta, with
    # and added acceleration that is proportional to the speed by dsec value.
    # aceleration * tau K / speed = dsec.
    d = dim(model)
    tauK = maximum(model.times)
    return vcat([psi(model.static, theta[1:d] + t * theta[d+1:end] + t^2*dsec*theta[d+1:end]/tauK/2) for t in model.times]...)
end

function run_simulation_target(model, thetas, weights, target)
# We simulate the reconstruction algorithm on the data given by target, and we return
# the error values from the background truth.
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
end

function generate_target(model, thetas, weights, noise_level = 0.0, noise_position = 0.0)
    # Noise is added to the simulations. Noise_level is a percentage of the total
    # signal in L2 norm, in other words, it is the 1/signal-to-noise.
    if noise_position > 0
        target =  phi_noise(model, thetas, weights, noise_position)
    else
        target = phi(model, thetas, weights)
    end
    # Noise is bounded by 1 in modulus
    noise = randn(size(target))*noise_level
    # Each particle's signal is bounded by it's weights, we set them to 1 ideally
    if (maximum(weights)>2 || minimum(weights)<0.1) && noise_level > 0 
	warn(" The proportionality of noise is not accurate as the weights are too high or low ")
    end
    # So a porcentage is obtained by just ponderating the noise.
    return target + noise
end

function match_points(theta_1, theta_2)
    # Method to match the reconstruction's particles with the given ones.
    n_points = size(theta_1, 2)
    corres = zeros(Int, n_points)
    distances = zeros(n_points,n_points)
    # to get from one dimensional array to the indexes in two dimensions.
    function iindex(x,N)
	return mod(x-1,N)+1
    end
    function jindex(x,N)
	return div(x-iind(x),N)+1
    end
    # distances matrix.
    for i = 1:n_points
        for j = 1:n_points
            distances[i,j] = norm(theta_1[:,i] - theta_2[:,j])
        end
    end
    # We take the closest ones and match them
    for k = 1:n_points
	x = indmin(distances)
	(i,j) = (iindex(x,n_points), jindex(x,n_points))
	corres[j]=i
	distances[i,:] = Inf
	distances[:,j] = Inf
    end
    return corres
end

function to_static(thetas, t, x_max)
    # For a moving particle and a time, gives its static position.
    d = div(size(thetas, 1),2)
    pts = thetas[1:d,:]
    velocities = thetas[d+1:2*d,:]
    pts_t = pts + velocities * t
    pts_t = mod.(pts_t, x_max)
    return pts_t
end

function target_to_static(target_dynamic, times)
    # For time dependent measurements, return a container with each static measurement.
    static_size = div(length(target_dynamic), length(times))
    static_target = []
    for k = 1:length(times)
	push!(static_target, target_dynamic[static_size*(k-1) + 1: static_size*k])
    end
    return static_target
end

function separation_val(thetas,K,tau,x_max)
# function that evaluates the separation between the particles
# We project into the time measurements, and we take the third best
# separation.
separations = zeros(2*K+1)
# We also consider toroidal modulus distance on the interval
toroidalSeparation = zeros(2*K+1)
for k in 1:2*K+1
   separations[k] = minimum([abs(thetas[1,i]-thetas[1,j]+tau*(k-K-1)*(thetas[2,i]-thetas[2,j])) + (i==j)*x_max for i in 1:size(thetas,2), j in 1:size(thetas,2)])
   toroidalSeparation[k] = minimum([abs(thetas[1,i]+tau*(k-K-1)*(thetas[2,i]) - (thetas[1,j]+tau*(k-K-1)*(thetas[2,j])-x_max)) + (i==j)*x_max for i in 1:size(thetas,2), j in 1:size(thetas,2)])
end
separations = [min(separations[k],toroidalSeparation[k]) for k in 1:length(separations)]
sort!(separations)
return separations[end-2]
end

function separation_norm(thetas,K,tau,x_max)
# function that evaluates the separation between the particles, but in terms
# of the dynamic norm.
delta = K*tau
norm = (x,v) -> abs(x) + delta*abs(v)
return minimum([norm(thetas[1,i] - thetas[1,j], thetas[2,i] - thetas[2,j]) + (i==j)*Inf for i in size(thetas,2), j in 1:size(thetas,2) ])
end

function Rejection_sampling(test_case, bins, density, K, tau, x_max)
    # We use a rejection sampling algorithm to generate test_cases whose
    # separation is distributed uniformly. To save time in Montecarlo simulations.
    # Since we have no formula for the distribution of separations for random particles,
    # the distribution is estimated with separationDistribution.jl script
    # WARNING: This works only for cloud_1d_full in TestCases.jl. 
    dx = bins[2]-bins[1]
    minDensity = minimum(density)
    densityFunc = y -> density[floor(Int,y/dx)+1]
    densityLength = length(density)
    safetyInd = 1
    (thetas, weights) = test_case()
    y = separation_val(thetas,K,tau,x_max)
    # Reject all the particules outside the desired interval.
    while floor(Int,y/dx)+1 >= densityLength
        (thetas, weights) = test_case()
        y = separation_val(thetas,K,tau,x_max)
    end
    # rejection-aceptation method
    reject_var = rand()
    while safetyInd<1e6 && reject_var >= minDensity/densityFunc(y)
        (thetas, weights) = test_case()
        y = separation_val(thetas,K,tau,x_max)
        while floor(Int,y/dx)+1 >= densityLength
            (thetas, weights) = test_case()
            y = separation_val(thetas,K,tau,x_max)
        end
        reject_var = rand()
	safetyInd = safetyInd + 1
    end
    if safetyInd == 1e6
        error(" Something is going wrong with the rejection algorithm ! We are
        not finding any appropiate sample ! ")
    end
    return (thetas, weights)
end

function generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
    d = dim(model_dynamic)
    # run the algorithm on the points thetas with the respective noises.
    (thetas_est, weights_est) = try run_simulation_target(model_dynamic, thetas, weights, target) catch  
	warn("fail") 
	([0], [0]) 
	end
    # Discard any reconstruction with smaller weight than some threshold.
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    if (length(thetas) == length(thetas_est))
        # The output of the minimization algorithm returns an unsorted vector with the
        # estimated positions ans weights of particles. match_points is a function that
        # returns a vector that tries to match the best possible the corresponding particles
        # to the estimated particles.
        corres = match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
        dist_w = norm(weights[:] - weights_est[corres], Inf)
        # returns the infinite distance between the matched particles, in space and velocity.
        return (dist_x, dist_v, dist_w)
    end
    # if the reconstructed particles don't match in size, we call this a failure and the
    # distance is the maximal given one.
    return model_dynamic.static.x_max, model_dynamic.v_max, maximum(weights)
end

function generate_and_reconstruct_static(model_static, thetas, weights, target_static)
    # Description is similar to the one in generate_and_reconstruct_dynamic.
    d = dim(model_static)
    (thetas_est, weights_est) = try
        run_simulation_target(model_static, thetas, weights, target_static)
    catch
        warn("fail")
        ([0], [0])
    end
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    if (length(thetas) == length(thetas_est))
        corres = match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        dist_w = norm(weights[:] - weights_est[corres], Inf)
        return dist_x, dist_w
    end
    return model_static.x_max, maximum(weights)
end

function generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target_dynamic)
    # Given a dynamic set of particles, we obtain the respective static reconstructions
    # at each time sample, and afterwards we return the error in space and weight
    # for the case of the best one, and the third best one.
    dist_x = zeros(length(model_dynamic.times))
    dist_w = zeros(length(model_dynamic.times))
    static_target = target_to_static(target_dynamic, model_dynamic.times)
    for k = 1:length(model_dynamic.times)
        thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
        (dist_x[k], dist_w[k]) = generate_and_reconstruct_static(model_static, thetas_t, weights, static_target[k])
    end
    sortOrder = sortperm(dist_x)
    dist_x = dist_x[sortOrder]
    dist_w = dist_w[sortOrder]
    # The first two outputs are the best case and its weight
    # the second two outpust are the third best case with the worse weight.
    return dist_x[1], dist_w[1], dist_x[3], maximum(dist_w[1:3])
end

function generate_and_reconstruct_all(model_static, model_dynamic, bins, density, test_case, noises_data, noises_position, cases)
    ### This function generates a test case and returns the distances between the generated particles
    ### given by dx, dv and norm. Furthermore it returns a result matrix that contains
    ### the first 3 colums correspond to the dynamic results, (space x velocity x weights)
    ### whereas the last 4 columns correspond to the static results-: best measurement location
    ### and weight, third best location, worse weight difference among the best 3.
    ### The rows are respectively no noise, noises in data, noises in position.
    (noiseless_dynamic, noiseless_static, noise_dynamic, noise_static, curvature_dynamic) = cases
    K = div(length(model_dynamic.times)-1,2)
    tau = maximum(model_dynamic.times)/K
    (thetas, weights) =  Rejection_sampling(test_case, bins, density, K, tau, model_static.x_max)
    d = dim(model_dynamic)
    results = zeros(1 + length(noises_data) + length(noises_position), 7)
    # Obtain the separation of the configuration of particles
    separation = separation_val(thetas,K,tau,model_static.x_max)
    # And the separationg given by the dynamic norm.
    separation_dyn = separation_norm(thetas,K,tau,model_static.x_max)
    # Dynamic case, no noise
    target = generate_target(model_dynamic, thetas, weights)
    println("######### STARTING ########")
    # We reconsctruct the location of the particles for the target data.
    if noiseless_dynamic == true
	println(" trying noiseless dynamical ###### ")
       (results[1, 1], results[1, 2], results[1,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
    else
        (results[1,1]. results[1,2], results[1,3]) = (0,0,0)
    end
    # Static case, no noise
    if noiseless_static == true
	println(" trying noiseless static  ###### ")
        (results[1,4], results[1,5], results[1,6], results[1,7]) = generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target)
    else
        (results[1,4], results[1,5], results[1,6], results[1,7]) = (0,0,0,0)
    end
    i = 2
    # recomputes, adding differents levels of noise, for both dynamic and static.
    for noise_data in noises_data 
        target = generate_target(model_dynamic,thetas,weights, noise_data, 0.0)
        # Dynamic case
        if noise_dynamic == true
	   println(" trying dynamical noise data ", noise_data, "####" )
           (results[i, 1], results[i, 2], results[i,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
	else
	   (results[i, 1], results[i, 2], results[i,3]) = (0,0,0)
	end 
        # Static case
	if noise_static == true
	   println(" trying static noise data ", noise_data, "####" )
           (results[i,4], results[i,5], results[i,6], results[i,7]) = generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target)
	else
	   (results[i,4], results[i,5], results[i,6], results[i,7]) = (0,0,0,0)
	end
        i = i + 1
    end
    for noise_position in noises_position
        target = generate_target(model_dynamic,thetas, weights, 0.0, noise_position)
        # This result is only analyzed in the Dynamic case for the moment
	if curvature_dynamic == true
	    println(" trying noise position", noise_position, "#####" )
            (results[i, 1], results[i, 2], results[i,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
	else 
	    (results[i, 1], results[i, 2], results[i,3]) = (0,0,0)
	end
        results[i, 4] = 0.0
        results[i, 5] = 0.0
        results[i, 6] = 0.0
        results[i, 7] = 0.0
        i = i + 1
    end
    println(" ################# FINISHED #######################") 
    return separation, separation_dyn, results
end
end
