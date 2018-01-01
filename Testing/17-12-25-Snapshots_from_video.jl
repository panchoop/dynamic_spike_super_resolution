# 25/12/17
# Francisco Romero H.

# Making a video to see how the measurements evolve in time.
# First we plot the error of the method, to see how many successful reconstructions there were.
# Secondly we plot the L1 norm of the video at each time sample, to 
# see when there is a constant number of particles, and have an idea of when the particles are reconstructed.
# Thirdly we plot the sum of all the particles in time, so we can see the domain in which they move
# The matrix is flipped so the particles seem to go horizontally instead of vertically.
# Finally we present a screenshot of every time step, where the moving particles can be observed.


# File to include and test
foldername = "../data/2017-11-03T15-27-19-546"

using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
include(foldername * "/ufus_parameters.jl")
errors = np.load(foldername * "/errors.npy")
video = np.load(foldername * "/video.npy")
n_x = 13
n_y = 13
n_im = size(video)[2]

# Plotting the errors.
println(size(errors))
plt.figure()
plt.plot(1:length(errors),errors)
plt.show()

# Plot the L1 norm of the video at each frame, to see how many particles are present at every timestep
norms = zeros(n_im,1);
for i = 1:n_im
    norms[i] = norm(video[:,i],1)
end
plt.figure()
plt.plot(1:n_im, norms)
plt.show()


# Check all the places where the particles have been
plt.figure()
plt.pcolormesh(np.linspace(0, 0.01, n_x), np.linspace(0, 0.01, n_y), reshape(sum(video,2),n_x,n_y))
plt.colorbar()
#plt.savefig("singleframe.pdf")
plt.show()


for i = 1:n_im
plt.figure()
plt.pcolormesh(np.linspace(0, 0.01, n_x), np.linspace(0, 0.01, n_y), reshape(video[:,i],n_x,n_y))
plt.colorbar()
#plt.savefig("singleframe.pdf")
plt.show()
end
