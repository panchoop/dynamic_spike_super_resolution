using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
# Change for the respective example to load
 example = "/2017-11-03T09-40-41-038"
# example = "/2017-11-03T10-05-27-667"
# example = "/2017-11-04T08-04-38-411"
# example = "/2017-11-04T11-57-14-36"
folder = "data/2Dsimulations"*example


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
plt.savefig(folder*"/superres.pdf")
plt.show()

plt.pcolormesh(np.linspace(0, 0.01, 13), np.linspace(0, 0.01, 13),reshape(np.sum(video, 1),13,13))
plt.colorbar()
plt.savefig(folder*"/bmode.pdf")
plt.show()


plt.figure()
plt.pcolormesh(np.linspace(0, 0.01, 13), np.linspace(0, 0.01, 13), reshape(video[:,1],13,13))
plt.colorbar()
plt.savefig(folder*"/singleframe.pdf")
plt.show()
