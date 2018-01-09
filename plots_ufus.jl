using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
folder = ARGS[1]
include(folder * "/ufus_parameters.jl")
errors = np.load(folder * "/errors.npy")
video = np.load(folder * "/video.npy")
# We define threshold to accept or reject recostructions:
# the error threshold is how much measurement missmatch are we going to tolerate.
threshold_error = 0.1
# The weight threshold is at which mass we are going to consider the reconstruction a valid particle.
threshold_weight = 0.1
all_thetas = np.zeros((5, 0))
for i in 1:length(errors)
    if errors[i] < threshold_error
        theta = np.load(folder * "/thetas-" * string(i) * ".npy")
        all_thetas = hcat(all_thetas, theta[:, theta[5,:].>threshold_weight])
    end
end
# We plot the values, the speeds are described by the colors, as they have two possible speeds only.
sizeAmp = 50;
plt.scatter(all_thetas[1,:], all_thetas[2,:], c=all_thetas[4,:], s=all_thetas[5,:]*sizeAmp, alpha = 0.5)
plt.colorbar()
plt.xlim((0, x_max))
plt.ylim((0, x_max))
plt.savefig("superres.pdf")
plt.show()

println(size(video))
println(length(np.sum(video,1)))
plt.pcolormesh(np.linspace(0, 0.01, 13), np.linspace(0, 0.01, 13),reshape(np.sum(video, 1),13,13))
plt.colorbar()
plt.savefig("bmode.pdf")
plt.show()


println(size(video[:,1]))
plt.figure()
plt.pcolormesh(np.linspace(0, 0.01, 13), np.linspace(0, 0.01, 13), reshape(video[:,1],13,13))
plt.colorbar()
plt.savefig("singleframe.pdf")
plt.show()
