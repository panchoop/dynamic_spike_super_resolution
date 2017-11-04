using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
folder = ARGS[1]
include(folder * "/ufus_parameters.jl")
errors = np.load(folder * "/errors.npy")
video = np.load(folder * "/video.npy")
threshold = 0.1
threshold_weight = 0.1
all_thetas = np.zeros((5, 0))
for i in 1:length(errors)
    if errors[i] < threshold
        theta = np.load(folder * "/thetas-" * string(i) * ".npy")
        all_thetas = hcat(all_thetas, theta[:, theta[5,:].>threshold_weight])
    end
end
plt.scatter(all_thetas[1,:], all_thetas[2,:], c=all_thetas[4,:])
plt.colorbar()
plt.xlim((0, x_max))
plt.ylim((0, x_max))
plt.savefig("superres.pdf")
plt.show()
# plt.pcolormesh(np.linspace(0, 0.01, 26), np.linspace(0, 0.01, 26),np.sum(video, 1).reshape((26,26)))
# plt.colorbar()
# plt.savefig("bmode.pdf")
# plt.figure()
# plt.pcolormesh(np.linspace(0, 0.01, 26), np.linspace(0, 0.01, 26), video[:,0].reshape(26,26))
# plt.colorbar()
# plt.savefig("singleframe.pdf")
# plt.show()
