import numpy as np
from matplotlib import pyplot as plt

x_max = 0.01
delta = 1.0/30
f_c = 20

dx = np.load("dx.npy")
dv = np.load("dv.npy")
norm = np.load("norm.npy")
datanoise = np.load("datanoise.npy")
positionnoise = np.load("positionnoise.npy")
results = np.load("results.npy")
results[results==0] = x_max

N = len(norm)


# Plot bins
def plot_success(norm, success, n_bins = 10, **kwargs):
    bins = np.linspace(np.percentile(norm, 10), np.percentile(norm, 90), n_bins)
    vals = np.zeros(len(bins) - 1)
    norm_success = norm[success]
    for i in range(len(bins) - 1):
        n_success = ((norm_success >= bins[i]) * (norm_success < bins[i+1])).sum()
        n_total = ((norm >= bins[i]) * (norm < bins[i+1])).sum()
        vals[i] = n_success / float(n_total)
    centers = 0.5 * (bins[0:len(bins) - 1] + bins[1:len(bins)])
    plt.plot(centers, vals, **kwargs)
    plt.ylim((0, 1))


# Noiseless case
dx_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 0]
dv_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 1]
dx_static = results[0::(1+len(datanoise)+len(positionnoise)), 2]
srf_static = x_max/dx_static/f_c
srf_dyn_x = x_max/dx_dyn/f_c
srf_dyn_v = x_max/dv_dyn/f_c/delta
srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
srf_threshold = 50
success = np.nonzero(srf_static > srf_threshold)
plot_success(norm, success, linestyle="dotted")
success = np.nonzero(srf_dyn > srf_threshold)
plot_success(norm, success, linestyle="solid")
plt.legend(["static", "dynamic"])

styles = ["solid", "dashed", "dotted", "dashdot", "solid", "solid"]

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
plt.figure()
for i in range(len(datanoise)):
    dx_static = results[i::(1+len(datanoise)+len(positionnoise)), 2]
    srf_static = x_max/dx_static/f_c
    success = np.nonzero(srf_static > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend([str(int(100*datanoise[i])) + "% noise" for i in range(len(datanoise))])

# Nonlinearity comparison
plt.figure()
for i in range(3):
    dx_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 0]
    dv_dyn = results[1+len(datanoise)+i::(1+len(datanoise)+len(positionnoise)), 1]
    srf_dyn_x = x_max/dx_dyn/f_c
    srf_dyn_v = x_max/dv_dyn/f_c/delta
    srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
    success = np.nonzero(srf_dyn > srf_threshold)
    plot_success(norm, success, linestyle=styles[i])

plt.legend([str(positionnoise[i]) + " curvature" for i in range(3)])
# SRF comparison
plt.show()
