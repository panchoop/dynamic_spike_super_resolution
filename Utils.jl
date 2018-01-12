push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
module Utils
using SparseInverseProblems
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

################## Function soon to be obsolete ##################
function run_simulation(model, thetas, weights, noise_level=0.0, noise_position=0.0)
    # Noise is added to the simulations. Noise_level is a percentage of the total
    # signal in L2 norm, in other words, it is the 1/signa-to-noise.
    if noise_position > 0
        target =  phi_noise(model, thetas, weights, noise_position)
    else
        target = phi(model, thetas, weights)
    end
    noise = randn(size(target))
    noise = (noise/norm(noise[:],2))*norm(target[:],2)*noise_level
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
end
###################################################################
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
    # signal in L2 norm, in other words, it is the 1/signa-to-noise.
    if noise_position > 0
        target =  phi_noise(model, thetas, weights, noise_position)
    else
        target = phi(model, thetas, weights)
    end
    noise = randn(size(target))
    noise = (noise/norm(noise[:],2))*norm(target[:],2)*noise_level
    target = target + noise
end
####################################### Seems to be a useless piece of code !
function measure_noise(thetas, weights, noise_level_x, noise_level_v, noise_level_weights)
    d = div(size(thetas, 1), 2)
    num_points = size(thetas, 2)
    new_thetas = zeros(2*d, num_points)
    new_thetas[1:d, :] = thetas[1:d, :] + (1-2*rand(d, num_points)) * noise_level_x
    new_thetas[d+1:end, :] = thetas[d+1:end, :] + (1-2*rand(d, num_points)) * noise_level_v
    new_weights = weights + noise_level_weights * rand(num_points)
    return new_thetas, new_weights
end
######################################################

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
function target_to_static(target_dynamic, times)
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
# of the dynamic norm. We ommited the toroidal geometry for this norm.
delta = K*tau
norm = (x,v) -> abs(x) + delta*abs(v)
return minimum([norm(thetas[1,i] - thetas[1,j], thetas[2,i] - thetas[2,j]) + (i==j)*Inf for i in size(thetas,2), j in 1:size(thetas,2) ])
end

function Rejection_sampling(test_case, bins, density, K, tau, x_max)
    # We use a rejection sampling algorithm to generate test_cases that whose
    # separation is distributed uniformly. This wil save time in Montecarlo simulations
    # to have adequate resolution in the final plots.
    # The distribution is unknown but we have simulated it enough times to approximate
    # it (see separationDistribution.jl)
    # On average, this method takes 1/bins[end]/minimum(density)
    dx = bins[2]-bins[1]
    minDensity = minimum(density)
    densityFunc = y -> density[floor(Int,y/dx)+1]
    safetyInd = 1
    (thetas, weights) = test_case()
    y = separation_val(thetas,K,tau,x_max)
    # in case we escape of our selected interval. -we discard the simulation,
    # we select the desired separation interval when computing the density function
    #(see separationDistribution.jl).
    while y > bins[end-1]
        (thetas, weights) = test_case()
        y = separation_val(thetas,K,tau,x_max)
    end
    # rejection variable
    u = rand()
    while safetyInd<1e6 && u >= minDensity/densityFunc(y)
        (thetas, weights) = test_case()
        y = separation_val(thetas,K,tau,x_max)
        while y > bins[end-1]
            (thetas, weights) = test_case()
            y = separation_val(thetas,K,tau,x_max)
        end
        u = rand()
    end
    if safetyInd == 1e6
        error(" Something is going wrong with the rejection algorithm ! We are
        not finding any appropiate sample ! ")
    end
    return (thetas, weights)
end
#######################################################
function generate_and_reconstruct_dynamic_old(model_dynamic, thetas, weights, noise_data, noise_position)
    d = dim(model_dynamic)
    # run the algorithm on the points thetas with the respective noises.
    (thetas_est, weights_est) = try run_simulation(model_dynamic, thetas, weights, noise_data, noise_position) catch ([0], [0]) end
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
#######################################################
function generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
    d = dim(model_dynamic)
    # run the algorithm on the points thetas with the respective noises.
    (thetas_est, weights_est) = try run_simulation_target(model_dynamic, thetas, weights, target) catch  ([0], [0]) end
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
function generate_and_reconstruct_all(model_static, model_dynamic, bins, density, test_case, noises_data, noises_position)
    ### This function generates a test case and returns the distances between the generated particles
    ### given by dx, dv and norm. Furthermore it returns a result matrix that contains
    ### the first 3 colums correspond to the dynamic results, (space x velocity x weights)
    ### whereas the last 4 columns correspond to the static results-: best measurement location
    ### and weight, third best location, worse weight difference among the best 3.
    ### The rows are respectively no noise, noises in data, noises in position.
    srand(100000000)
    println("### Beggining round ###")
    K = div(length(model_dynamic.times)-1,2)
    tau = maximum(model_dynamic.times)/K
    (thetas, weights) =  Rejection_sampling(test_case, bins, density, K, tau, model_static.x_max)
    d = dim(model_dynamic)
    results = zeros(1 + length(noises_data) + length(noises_position), 7)
    # Obtain the separation of the configuration of particles
    separation = separation_val(thetas,K,tau,model_static.x_max)
    # And the separationg given by the dynamic norm.
    separation_dyn = separation_norm(thetas,K,tau,model_static.x_max)
    ## 	Noiseless case
    # Dynamic case
    # We generate the target measure
    target = generate_target(model_dynamic, thetas, weights)
    # We reconsctruct the location of the particles for the target data.
    (results[1, 1], results[1, 2], results[1,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
    # Static case, no noise
    (results[1,4], results[1,5], results[1,6], results[1,7]) = generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target)
    i = 2
    # recomputes, adding differents levels of noise, for both dynamic and static.
    for noise_data in noises_data
        # We generate target measure with the included noise
        target = generate_target(model_dynamic,thetas,weights, noise_data, 0.0)
	println("---- Printing setup: ")
	println("thetas:",thetas)
	println("weights:",weights)
	println("target:",target)
        # Dynamic case
        println("#### going for noise_data dynamic ",noise_data," ###")
        (results[i, 1], results[i, 2], results[i,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
        # Static case
        println("#### going for noise_data static ",noise_data ,"###")
        (results[i,4], results[i,5], results[i,6], results[i,7]) = generate_and_reconstuct_static_best(model_dynamic, model_static, thetas, weights, target)
        i = i + 1
    end
    for noise_position in noises_position
        # We generate target measure with the included noise
        target = generate_target(model_dynamic,thetas, weights, 0.0, noise_position)
        # This result is only analyzed in the Dynamic case for the moment
        println("#### going for noise_position dynamic ",noise_position, " ###")
        (results[i, 1], results[i, 2], results[i,3]) = generate_and_reconstruct_dynamic(model_dynamic, thetas, weights, target)
        results[i, 4] = 0.0
        results[i, 5] = 0.0
        results[i, 6] = 0.0
        results[i, 7] = 0.0
        i = i + 1
    end
    println("#########################################################")
    println("#########################################################")
    println("#########################################################")
    println("#########################################################")
    println("             FINISHED ONE ROUND                          ")
    println("#########################################################")
    println("#########################################################")
    println("#########################################################")
    println("#########################################################")
    return separation, separation_dyn, results
end
end
