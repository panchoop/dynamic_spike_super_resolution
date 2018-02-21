# Sequencing process particle centered.

push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
push!(LOAD_PATH, "./SparseInverseProblems/src")
using SuperResModels
using SparseInverseProblemsMod
using Distributions

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

x_max = 1/100 # [mm]
v_max = 15/100 # [mm/s]
tau = 1/500 # sampling rate.
sigma_noise = 0.00 # noise amplitude.
Nparticles = 500
activation_prob = 0.0002
expected_time = 35

### Generate sequence ###
println("Generating sequence... ")
video = SharedArray{Float64}(n_x * n_y, n_im)

function activate_particles(Nparticles,activation_prob, expected_time)
    # particles are tuples of position,velocity, weight, survival time.
    B = Binomial(Nparticles,activation_prob)
    P = Binomial(1,0.5)
    U = Uniform(0,x_max)
    N = Poisson(expected_time)

    function flowSpeeds(yposition)
         return v_max/3 + yposition/x_max*v_max/3*2
    end
    N_new_particles = rand(B)
    new_particles = []
    for i in 1:N_new_particles
        # decide if it is flowing downwards (=0) or upwards (=1)
        direction = rand(P)
        yposition = rand(U)
        weight = 1
        if direction == 0
            position = (x_max/4, yposition)
            velocity = flowSpeeds(yposition)
        else
            position = (x_max*3/4, yposition)
            velocity = -flowSpeeds(yposition)
        end
        time = rand(N)
        push!(new_particles,(position, velocity, weight, time))
    end
    return new_particles
end

function time_step(particles)
    for j in length(particles):-1:1
	particle = particles[j]
	# update position
        new_position = particle[1][2] + tau*particle[2]
	# update time
	new_time = particle[4]-1
	# discard if time is over
	if new_time <= 0
	    deleteat!(particles,j)
	else
	    particles[j]=((particle[1][1], new_position),particle[2], particle[3], new_time)
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
    thetas = zeros(2,length(particles))
    weights = zeros(length(particles))
    for j in 1:length(particles)
	thetas[1,j] = particles[j][1][1]
	thetas[2,j] = particles[j][1][2]
	weights[j] = particles[j][3]
    end
    ## Image the particles
    # include noise
    video[:,i] = sigma_noise*randn(size(video[:,i]))
    if (length(weights) > 0)
        video[:,i] += phi(model_static, thetas, weights)
    end
end

for i = 1:500
plt.figure()
plt.pcolormesh(np.linspace(0, 0.01, 13), np.linspace(0, 0.01, 13), reshape(video[:,i],13,13))
plt.colorbar()
plt.show()
end

showFrames = 1:500
plt.figure()
# Compute the L2 norm at each time sample
L2norms = sqrt.(sum(video[:,showFrames].^2,1))[:]
plt.plot(1:length(L2norms), L2norms)
# Paint the close to constant cases
minSnapshot = 5
tolerance = 0.05
j = 1
i = 1
while i <= length(L2norms)-minSnapshot
    while (j+i <= length(L2norms)-minSnapshot) & (abs.(L2norms[i]-L2norms[i+j])<=tolerance)
	j = j +1
    end
    if j-1 >= minSnapshot
	plt.plot([i, i+j-1], [mean(L2norms[i:i+j-1]), mean(L2norms[i:i+j-1])],color="r")
    end
    i = i+j
    j = 1
end
plt.show()

