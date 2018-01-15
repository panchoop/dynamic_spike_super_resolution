push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

@everywhere begin
    include("2d_parameters.jl")
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


# A single static particle.
thetas = reshape([x_max/2; x_max/2], 2, 1)
weights = [1.0]

# Phi represents the forward operator.
single_particle_norm = norm(phi(model_static, thetas, weights), lp_norm)
@everywhere single_particle_norm = $single_particle_norm

# The examples we seek to emulate correspond to two sets of particles moving along a two stripes on each side.
# So particles_m stands for those located on the west side, moving towards the south. 
# The vector particles_m basically are the locations at time 0 of the particles, it also defines quantity.
# Particles_p are the symmetric equivalent, on the east and moving towards the north. 


### Generate sequence ###
println("Generating sequence...")
video = SharedArray{Float64}(n_x * n_y, n_im)
particles_m = [x_max/4]
particles_p = [3*x_max/4]


activation_probability=0.02
for i in 1:n_im
    println(i, "/", n_im) 
    # Obtain the current static particles, just spatial, there is no speed.
    thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'],
                  [(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'])
    weights = ones(size(thetas, 2))
    # Image it and add it to our data vector: video
    if (length(weights) > 0)
        video[:,i] = phi(model_static, thetas, weights)
        video[:,i] = video[:,i] + sigma_noise * randn(size(video[:,i]))
    end
    # Displace the particles by their speeds
    particles_m -= v_max/2 * tau
    particles_p += v_max/2 * tau
    # Randomly deactivate particles
        remove = []
        for i in 1:length(particles_m)
            if rand() < activation_probability
                 push!(remove, i)
            end
        end
        deleteat!(particles_m, remove)
        remove = []
        for i in 1:length(particles_p)
        if rand() < activation_probability
            push!(remove, i)
        end
	end
	deleteat!(particles_p, remove)
    # Randomly activate particles
	if length(weights) == 0
	    push!(particles_m, initial_position_generator())
	    push!(particles_p, initial_position_generator())
        end
        if rand() < activation_probability
            push!(particles_m, initial_position_generator())
        end
        if rand() < activation_probability
            push!(particles_p, initial_position_generator())
        end
end


### Obtaining time frames in which the quantity of particles remained constant ###
println("Getting short sequences without jump...")
# get the total mass at each time step
frame_norms = [norm(video[:, i], lp_norm) for i in 1:n_im]
@everywhere frame_norms = $frame_norms
println("norms: ", frame_norms)
# Find locations in which there was a significative mass difference between two time steps.
jumps = find(abs.(frame_norms[2:end] - frame_norms[1:end-1]) .> jump_threshold*single_particle_norm)
jumps = [0; jumps; n_im]
# Obtain non-overlapping intervals of at least 2K+1 consecutive time samples in which the mass didn't changed.
short_seqs = []
for i in 1:length(jumps)-1
    append!(short_seqs, [(jumps[i] + 5*j + 1):(jumps[i] + 5*j + 5) for j in 0:(div(jumps[i+1] - jumps[i], 5) -1)])
end
println("jumps: ", jumps)
println("short seqs: ", short_seqs)

### Function that given the a subquence of the total video, will estimate the locations and weights of the
### involved particles. 
@everywhere function posvel_from_seq(video, seq)
    assert(length(seq) == 5)
    target = video[:,seq][:]
    # estimated number of particles in the sequence.
    est_num_particles = div(frame_norms[seq[1]], single_particle_norm*0.95)
    if (est_num_particles == 0)
        return Matrix{Float64}(5,0)
    end
    # Function required to use the SparseInverseProblems.ADCG method.
    function callback(old_thetas, thetas, weights, output, old_obj_val)
        #evalute current OV
        new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
        #println("gap = $(old_obj_val - new_obj_val)")
        if old_obj_val - new_obj_val < 1E-4
            return true
        end
        return false
    end
    # It uses the ACDG algorithm to estimate the location and weights of the particles. Using the estimated number of particles we can bound the total variation on the solutions.
    (thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, frame_norms[seq[1]], callback=callback, max_iters=2000)
    if length(thetas_est) > 0
        println("est_num = ", est_num_particles)
        println("thetas = ", thetas_est) 
        println("weights = ", weights_est)
        return [thetas_est; weights_est']
    else
        return Matrix{Float64}(5,0)
    end
end

println("Inverting...")
all_thetas = pmap(seq -> posvel_from_seq(video, seq), short_seqs)

println("Reprojecting...")
errors = zeros(length(short_seqs))
### Measurements error, we simulate the measurements that would be obtained with our reconstructed values ###
for i in 1:length(short_seqs)
    seq=  short_seqs[i]
    target = video[:, seq][:]
    if length(all_thetas[i]) > 0
        reprojection = phi(model_dynamic, all_thetas[i][1:4,:], all_thetas[i][5,:])
        println("error = ", norm(target-reprojection))
        errors[i] = norm(target-reprojection)
    else
        errors[i] = norm(target)
    end
end

### Save the simulated data ###
data_folder = "data/2Dsimulations/"*now_str

using PyCall
@pyimport numpy as np
mkdir(data_folder)
cp("2d_parameters.jl", string(data_folder, "/2d_parameters.jl"))
cd(data_folder)
short_seq_array = hcat(short_seqs...)
np.save("video", video)
np.save("frame_norms", frame_norms)
np.save("jumps", jumps)
np.save("short_seq_array", short_seq_array)
for i in 1:length(short_seqs)
    np.save(string("thetas-", i), all_thetas[i])
end
np.save("errors", errors)
cd("..")
