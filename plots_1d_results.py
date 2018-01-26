import numpy as np
import os
from matplotlib import pyplot as plt

# Folder with data files
#example = "2018-01-09T08-12-44-596"
example = "2018-01-25T16-56-54-564"
folder = "data/1Dsimulations/"+example
os.chdir(folder)

x_max = 1.00
delta = 1.0
v_max = 0.05
f_c = 20

dx = np.load("dx.npy")
dv = np.load("dv.npy")
norm = np.load("norm.npy")
datanoise = np.load("datanoise.npy")
positionnoise = np.load("positionnoise.npy")
results = np.load("results.npy")
results[results==0] = x_max

norm = norm
N = len(norm)

# Plot position noise
times = [k*delta for k in range(-2,3)]
plt.plot(times, [v_max*t for t in times])
plt.plot(times, [v_max*t + positionnoise[-1] * t**2 for t in times])

plt.legend(["second derivative = 0", "second derivative = " + str(positionnoise[-1])])
plt.savefig("curvature.pdf")
plt.figure()


# Plot bins
def plot_success(norm, success, n_bins = 20, **kwargs):
    bins = np.linspace(np.percentile(norm, 2), np.percentile(norm, 85), n_bins)
    vals = np.zeros(len(bins) - 1)
    norm_success = norm[success]
    for i in range(len(bins) - 1):
        n_success = ((norm_success >= bins[i]) * (norm_success < bins[i+1])).sum()
        n_total = ((norm >= bins[i]) * (norm < bins[i+1])).sum()
        vals[i] = n_success / float(n_total)
    centers = 0.5 * (bins[0:len(bins) - 1] + bins[1:len(bins)])
    plt.plot(centers, vals, **kwargs)
    plt.ylim((0, 1))
    plt.xlabel("$\Delta$", usetex=True)
    plt.ylabel("Correct recontruction rate")


# Noiseless case
dx_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 0]
dv_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 1]
dx_static = results[0::(1+len(datanoise)+len(positionnoise)), 2]
srf_static = x_max/dx_static/f_c
srf_dyn_x = x_max/dx_dyn/f_c
srf_dyn_v = x_max/dv_dyn/f_c/delta
srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
srf_threshold = 40
success = np.nonzero(srf_static > srf_threshold)
plot_success(norm, success, linestyle="dotted")
success = np.nonzero(srf_dyn > srf_threshold)
plot_success(norm, success, linestyle="solid")
plt.legend(["static", "dynamic"])

styles = ["solid", "dashed", "dotted", "dashdot", "solid", "solid"]
plt.savefig("noiseless.pdf")

# Noise comparison
plt.figure()
for i in range(len(datanoise)):
    dx_dyn = results[i::(1+len(datanoise)+len(positionnoise)), 0]
    dv_dyn = results[i::(1+len(datanoise)+len(positionnoise)), 1]
    srf_dyn_x = x_max/dx_dyn/f_c
    srf_dyn_v = x_max/dv_dyn/f_c/delta
    srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
    success = np.nonzero(srf_dyn > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend([str(int(100*datanoise[i])) + "% noise" for i in range(len(datanoise))])
plt.savefig("noisecomp-dyn.pdf")
plt.figure()
for i in range(len(datanoise)):
    dx_static = results[i::(1+len(datanoise)+len(positionnoise)), 2]
    srf_static = x_max/dx_static/f_c
    success = np.nonzero(srf_static > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend([str(int(100*datanoise[i])) + "% noise" for i in range(len(datanoise))])
plt.savefig("noisecomp-static.pdf")

# Nonlinearity comparison
plt.figure()
srf_threshold = 5
srf_thresholds = [1, 5, 10, 15, 20]
i = len(positionnoise) - 1
for j in range(len(srf_thresholds)):
    srf_threshold= srf_thresholds[j]
    dx_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 0]
    dv_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 1]
    srf_dyn_x = x_max/dx_dyn/f_c
    srf_dyn_v = x_max/dv_dyn/f_c/delta
    srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
    success = np.nonzero(srf_dyn > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend(["SRF = " + str(srf_thresholds[j]) for j in range(len(srf_thresholds))])
plt.savefig("curvcomp_srf.pdf")

plt.figure()
srf_threshold = 5
for i in range(len(positionnoise)):
    dx_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 0]
    dv_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 1]
    srf_dyn_x = x_max/dx_dyn/f_c
    srf_dyn_v = x_max/dv_dyn/f_c/delta
    srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
    success = np.nonzero(srf_dyn > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend(["second derivative = " + str(positionnoise[i]) for i in range(len(positionnoise))])
plt.savefig("curvcomp.pdf")
# SRF comparison
plt.show()
