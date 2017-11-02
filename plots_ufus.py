import numpy as np
from matplotlib import pyplot as plt
import sys
folder = sys.argv[1]
errors = np.load(folder + "/errors.npy")
video = np.load(folder + "/video.npy")
threshold = 0.02
threshold_weight = 0.1
all_thetas = np.zeros((5, 0))
for i in range(len(errors)):
    if errors[i] < threshold:
        theta = np.load(folder + "/thetas-" + str(i+1) + ".npy")
        print(theta.shape)
        all_thetas = np.hstack((all_thetas, theta[:, theta[4,:]>threshold_weight]))
video = np.load("video.npy")
plt.scatter(all_thetas[0,:], all_thetas[1,:], c=all_thetas[3,:])
plt.colorbar()
plt.xlim((0, 0.01))
plt.ylim((0, 0.01))
plt.savefig("superres.pdf")
plt.figure()
# plt.pcolormesh(np.linspace(0, 0.01, 26), np.linspace(0, 0.01, 26),np.sum(video, 1).reshape((26,26)))
# plt.colorbar()
# plt.savefig("bmode.pdf")
# plt.figure()
# plt.pcolormesh(np.linspace(0, 0.01, 26), np.linspace(0, 0.01, 26), video[:,0].reshape(26,26))
# plt.colorbar()
# plt.savefig("singleframe.pdf")
# plt.show()
