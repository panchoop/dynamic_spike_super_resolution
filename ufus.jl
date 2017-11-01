push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

@everywhere begin
    include("ufus_parameters.jl")
    parameters = SuperResModels.Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
    model_static = SuperResModels.Conv2d(parameters)
    dynamic_parameters = SuperResModels.DynamicConv2dParameters(K, tau, v_max, 20)
    model_dynamic = SuperResModels.DynamicConv2d(model_static, dynamic_parameters)
    n_x = model_static.n_x
    n_y = model_static.n_y
end

# L2 norm single particle
thetas = reshape([x_max/2; x_max/2], 2, 1)
weights = [1.0]
single_particle_norm = norm(phi(model_static, thetas, weights), lp_norm)
println("norm = ", single_particle_norm)


# Generate sequence
println("Generating sequence...")
video = SharedArray{Float64}(n_x * n_y, n_im)
particles_m = [initial_position_generator()]
particles_p = [initial_position_generator()]
for i in 1:n_im
    println(i, "/", n_im)
    thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'],
                  [(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'])
    weights = ones(size(thetas, 2))
    video[:,i] = phi(model_static, thetas, weights)
    particles_m -= v_max/2 * tau
    particles_p += v_max/2 * tau
    remove = []
    for i in 1:length(particles_m)
        if rand() < p
            push!(remove, i)
        end
    end
    deleteat!(particles_m, remove)
    remove = []
    for i in 1:length(particles_p)
        if rand() < p
            push!(remove, i)
        end
    end
    deleteat!(particles_p, remove)
    if length(weights) == 0
        push!(particles_m, initial_position_generator())
        push!(particles_p, initial_position_generator())
    end
    if rand() < p
        push!(particles_m, initial_position_generator())
    end
    if rand() < p
        push!(particles_p, initial_position_generator())
    end
end

# Getting frame sequences
println("Getting short sequences without jump...")
@everywhere frame_norms = [norm(video[:, i], lp_norm) for i in 1:n_im]
println("norms: ", frame_norms)
jumps = find(abs.(frame_norms[2:end] - frame_norms[1:end-1]) .> jump_threshold)
jumps = [0; jumps; n_im]
short_seqs = []
for i in 1:length(jumps)-1
    append!(short_seqs, [(jumps[i] + 5*j + 1):(jumps[i] + 5*j + 5) for j in 0:(div(jumps[i+1] - jumps[i], 5) -1)])
end
println("jumps: ", jumps)
println("short seqs: ", short_seqs)

@everywhere function posvel_from_seq(video, seq)
    assert(length(seq) == 5)
    target = video[:,seq][:]
    est_num_particles = div(frame_norms[seq[1]], single_particle_norm*0.95)
    if (est_num_particles == 0)
        return Matrix{Float64}(5,0)
    end
    function callback(old_thetas, thetas, weights, output, old_obj_val)
        #evalute current OV
        new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
        #println("gap = $(old_obj_val - new_obj_val)")
        if old_obj_val - new_obj_val < 1E-4
            return true
        end
        return false
    end
    (thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, frame_norms[seq[1]], callback=callback, max_iters=200)
    if length(thetas_est) > 0
        println("est_num = ", est_num_particles)
        println("thetas = ", thetas_est) 
        println("weights = ", weights_est)
        return [thetas_est; weights_est']
    else
        return Matrix{Float64}(5,0)
    end
end

#Inverse problem
println("Inverting...")
all_thetas = pmap(seq -> posvel_from_seq(video, seq), short_seqs)

println("Reprojecting...")
# Reprojection error
for bundle in bundles
    target = video[:, bundle][:]
    reprojection = phi(model_dynamic, all_thetas[1:4,:], all_thetas[5,:])
    println("error = ", norm(target-reprojection))
end
