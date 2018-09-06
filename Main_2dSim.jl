# Sequencing process particle centered.

push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
push!(LOAD_PATH, "./SparseInverseProblems/src")

using SuperResModels
using SparseInverseProblemsMod
using Distributions
using QuadGK
using Interpolations
using Roots

# To see the progress in pmap
# To use PmapProgressMeter you need to clone it manually
# do in Julia: Pkg.clone("https://github.com/slundberg/PmapProgressMeter.jl")
# if this is no longer existent, uncoment the alternative pmap in this code.
@everywhere using ProgressMeter
@everywhere using PmapProgressMeter

using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

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


Nparticles = 500
activation_prob = 0.0002
expected_time = 34

### Generate sequence ###
println("Generating sequence... ")
video = SharedArray{Float64}(n_x * n_y, n_im)

function activate_particles(Nparticles,activation_prob, expected_time)
    # particles are tuples of position,velocity, weight, survival time.
    # The position is a one dimensional variable, that once passed to the vessel function
    # and displaced accordingly, would return the respective position.
    B = Binomial(Nparticles,activation_prob)
    P = Binomial(1,0.5)
    U = Uniform(x_max*bndry_sep,t_max)
    N = Poisson(expected_time)

    N_new_particles = rand(B)
    new_particles = []
    for i in 1:N_new_particles
        # decide if it is flowing downwards (=0) or upwards (=1)
        direction = rand(P)
        position = rand(U)
        weight = 1
	    velocity = (direction-0.5)*2*particle_velocity
        time = rand(N)
        push!(new_particles,(position, velocity, weight, time))
    end
    return new_particles
end

function time_step(particles)
    for j in length(particles):-1:1
	particle = particles[j]
	# update position
        new_position = particle[1] + tau*particle[2]
	# update time
	new_time = particle[4]-1
	# discard if time is over
	if (new_time <= 0) | (new_position>t_max) | (new_position<bndry_sep*x_max)
	    deleteat!(particles,j)
	else
	    particles[j]=(new_position,particle[2], particle[3], new_time)
	end
    end
end
particles = []

for i in 1:n_im
    # Move particles forward in time
    time_step(particles)
    # Activate new particles and push them inside the list of particles
    new_particles = activate_particles(Nparticles,activation_prob, expected_time)
    for new_particle in new_particles
	push!(particles, new_particle)
    end
    # Represent the static particles to input into the phi function (forward operator)
    # thetas are 2d vectors representing position
    thetas = zeros(2,length(particles))
    weights = zeros(length(particles))
    for j in 1:length(particles)
    # The case in the upward or downward vessel
    if particles[j][2]>0
	    thetas[1,j] = vessel(particles[j][1])[1]
	    thetas[2,j] = vessel(particles[j][1])[2]
    else
        thetas[1,j] = vessel(particles[j][1])[1]+dx
        thetas[2,j] = vessel(particles[j][1])[2]-dx
    end
	weights[j] = particles[j][3]
    end
    ## Image the particles
    # include noise
    video[:,i] = sigma_noise*randn(size(video[:,i]))
    if (length(weights) > 0)
        video[:,i] += phi(model_static, thetas, weights)
    end
end

### Obtaining time frames in which the quantity of particles remained constant ###
println("Getting short sequences without jump...")
# get the total mass at each time step
frame_norms = [norm(video[:, i], lp_norm) for i in 1:n_im]
@everywhere frame_norms = $frame_norms
# Find locations in which there was a significative mass difference between two time steps.
jumps = find(abs.(frame_norms[2:end] - frame_norms[1:end-1]) .> jump_threshold*single_particle_norm)
jumps = [0; jumps; n_im]
# Obtain non-overlapping intervals of at least 2K+1 consecutive time samples in which the mass didn't changed.
short_seqs = []
for i in 1:length(jumps)-1
    append!(short_seqs, [(jumps[i] + 5*j + 1):(jumps[i] + 5*j + 5) for j in 0:(div(jumps[i+1] - jumps[i], 5) -1)])
end

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
        new_obj_val,t = SparseInverseProblemsMod.loss(SparseInverseProblemsMod.LSLoss(), output - target)
        #println("gap = $(old_obj_val - new_obj_val)")
        if old_obj_val - new_obj_val < 1E-4
            return true
        end
        return false
    end
    # It uses the ACDG algorithm to estimate the location and weights of the particles. Using the estimated number of particles we can bound the total variation on the solutions.
    (thetas_est,weights_est) = SparseInverseProblemsMod.ADCG(model_dynamic, SparseInverseProblemsMod.LSLoss(), target, frame_norms[seq[1]], callback=callback, max_iters=100)
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

all_thetas = pmap(seq -> begin sleep(1); posvel_from_seq(video, seq) end, Progress(length(short_seqs)), short_seqs)

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


#
# if false
# for i = 1:500
# plt.figure()
# plt.pcolormesh(np.linspace(0, 0.01, n_x), np.linspace(0, 0.01, n_x), reshape(video[:,i],n_x,n_y))
# plt.colorbar()
# plt.show()
# end
# end

# showFrames = 1:500
# plt.figure()
# # Compute the L2 norm at each time sample
# L2norms = sqrt.(sum(video[:,showFrames].^2,1))[:]
# plt.plot(1:length(L2norms), L2norms)
# # Paint the close to constant cases
# minSnapshot = 5
# tolerance = 0.05
# j = 1
# i = 1
# while i <= length(L2norms)-minSnapshot
#     while (j+i <= length(L2norms)-minSnapshot) & (abs.(L2norms[i]-L2norms[i+j])<=tolerance)
# 	j = j +1
#     end
#     if j-1 >= minSnapshot
# 	plt.plot([i, i+j-1], [mean(L2norms[i:i+j-1]), mean(L2norms[i:i+j-1])],color="r")
#     end
#     i = i+j
#     j = 1
# end
# plt.show()
