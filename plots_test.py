import numpy as np
import os
from matplotlib import pyplot as plt

# Folder with data files
example = "2018-01-09T19-13-58-745"
folder = "data/1Dsimulations/"+example
os.chdir(folder)

x_max = 1.00
delta = 1.0
v_max = 0.05
f_c = 20

separations = np.load("separations.npy")
separationDyn = np.load("separationDynamic.npy")
datanoise = np.load("datanoise.npy")
positionnoise = np.load("positionnoise.npy")
results = np.load("results.npy")
results[results==0] = x_max

N = len(separations)

# Plot position noise example
times = [k*delta for k in range(-2,3)]
plt.plot(times, [v_max*t for t in times])
plt.plot(times, [v_max*t + positionnoise[-1] * t**2 for t in times])

plt.legend(["second derivative = 0", "second derivative = " + str(positionnoise[-1])])
plt.savefig("curvature.pdf")
plt.figure()


# Plot bins to see success ratio
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
    plt.ylim((0, 1.2))
    plt.xlabel("$\Delta$", usetex=True)
    plt.ylabel("Correct recontruction rate")


# Noiseless case
# dynamic results
dx_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 0]
dv_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 1]
dw_dyn = results[0::(1+len(datanoise)+len(positionnoise)), 2]
# best static result
dx_stat_best = results[0::(1+len(datanoise)+len(positionnoise)),3]
dw_stat_best = results[0::(1+len(datanoise)+len(positionnoise)),4]
# third best static result
dx_stat_third = results[0::(1+len(datanoise)+len(positionnoise)),5]
dw_stat_third = results[0::(1+len(datanoise)+len(positionnoise)),6]

# Set the superresolution factor to decide if the reconstruction was a success.
srf_static = x_max/dx_stat_best/f_c
srf_dyn_x = x_max/dx_dyn/f_c
srf_dyn_v = x_max/dv_dyn/f_c/delta
srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
srf_threshold = 40
success = np.nonzero(srf_static > srf_threshold)
plot_success(separations, success, linestyle="dotted")
success = np.nonzero(srf_dyn > srf_threshold)
plot_success(separations, success, linestyle="solid")
plt.legend(["static", "dynamic"])

styles = ["solid", "dashed", "dotted", "dashdot", "solid", "solid"]
plt.savefig("noiseless.pdf")

# Noise comparison
plt.figure()
for i in range(len(datanoise)):
    dx_dyn = results[i::(1+len(datanoise)+len(positionnoise)), 0]
    dv_dyn = results[i::(1+len(datanoise)+len(positionnoise)), 1]
    dw_dyn = results[i::(1+len(datanoise)+len(positionnoise)), 2]
    srf_dyn_x = x_max/dx_dyn/f_c
    srf_dyn_v = x_max/dv_dyn/f_c/delta
    srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
    success = np.nonzero(srf_dyn > srf_threshold)
    plot_success(separations, success, linestyle=styles[i])

plt.legend([str(int(100*datanoise[i])) + "% noise" for i in range(len(datanoise))])
plt.savefig("noisecomp-dyn.pdf")
plt.figure()
for i in range(len(datanoise)):
    # best static result
    dx_stat_best = results[i::(1+len(datanoise)+len(positionnoise)),3]
    dw_stat_best = results[i::(1+len(datanoise)+len(positionnoise)),4]
    # third best static result
    dx_stat_third = results[i::(1+len(datanoise)+len(positionnoise)),5]
    dw_stat_third = results[i::(1+len(datanoise)+len(positionnoise)),6]
    srf_static = x_max/dx_stat_best/f_c
    success = np.nonzero(srf_static > srf_threshold)
    plot_success(separations, success, linestyle=styles[i])

plt.legend([str(int(100*datanoise[i])) + "% noise" for i in range(len(datanoise))])
plt.savefig("noisecomp-static.pdf")
