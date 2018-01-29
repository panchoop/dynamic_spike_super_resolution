import numpy as np
import os
from matplotlib import pyplot as plt

# Folder with data files
example = "2018-01-29T15-41-08-74"
folder = "data/1Dsimulations/"+example
os.chdir(folder)

x_max = 1.00
delta = 1.0
v_max = 0.05
f_c = 20

separations = np.load("separations.npy")
separationsDyn = np.load("separationDynamic.npy")
datanoise = np.load("datanoise.npy")
positionnoise = np.load("positionnoise.npy")
results = np.load("results.npy")
results[results==0] = x_max

aux = np.sort(separations)
print(aux[0])
print(aux[len(aux)-1])

N = len(separations)

# Plot position noise
times = [k*delta for k in range(-2,3)]
plt.plot(times, [v_max*t for t in times])
plt.plot(times, [v_max*t + positionnoise[-1] * t**2 for t in times])

plt.legend(["second derivative = 0", "second derivative = " + str(positionnoise[-1])])
plt.savefig("curvature.pdf")
plt.figure()


# Function to plot the success rate as bins
def plot_success(norm, success, n_bins = 30, **kwargs):
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

# Function to plot each specific cases
def plot_case(separations, case, noiseType, srf_threshold, weights_threshold,**kwargs):
	# case = 'static', 'dynamic' or 'static3'
	# noiseType depends on how the codes where generated, typically
	# 0 = no noise. 1 .. .N = measurement noise, N+1... = position noise.
	if case == "dynamic":
		# Obtain from the results the reconstruction missmatch: space, velocity, weight.
		dx_dyn = results[noiseType::(1+len(datanoise)+len(positionnoise)),0]
		dv_dyn = results[noiseType::(1+len(datanoise)+len(positionnoise)),1]
		dw_dyn = results[noiseType::(1+len(datanoise)+len(positionnoise)),2]
		# Compute the obtained super resolution factors
		srf_dyn_x = x_max/dx_dyn/f_c
		srf_dyn_v = x_max/dv_dyn/f_c/delta
		srf_dyn = np.minimum(srf_dyn_x, srf_dyn_v)
		success = np.nonzero(np.logical_and(srf_dyn > srf_threshold, dw_dyn < weights_threshold))
		plot_success(separations, success, **kwargs)
	elif case == "static":
		# Obtain from the results the reconstruction missmatch: space, weight.
		dx_static = results[noiseType::(1+len(datanoise)+len(positionnoise)), 3]
		dw_static = results[noiseType::(1+len(datanoise)+len(positionnoise)), 4]
		# Compute super resolution factor
		srf_static = x_max/dx_static/f_c
		success = np.nonzero(np.logical_and(srf_static > srf_threshold, dw_static < weights_threshold))
		plot_success(separations, success, **kwargs)
	elif case == "static3":
		# Obtain from the results the reconstruction missmatch: space, weight.
		dx_static3 = results[noiseType::(1+len(datanoise)+len(positionnoise)), 5]
		dw_static3 = results[noiseType::(1+len(datanoise)+len(positionnoise)), 6]
		# Compute super resolution factor
		srf_static3 = x_max/dx_static3/f_c
		success = np.nonzero(np.logical_and(srf_static3 > srf_threshold, dw_static3 < weights_threshold))
		plot_success(separations, success, **kwargs)
	else:
		error(" No adequate case assigned ")

		
### Noiseless case
# super resolution factor threshold for declaring accurate reconstruction.
srf_th = 40
# weight threshold to declare accurate reconstruction.
w_th = 0.1

plot_case(separations, "dynamic", 0, srf_th, w_th, linestyle = "-")
plot_case(separations, "static", 0, srf_th, w_th, linestyle = "-.")
plot_case(separations, "static3", 0, srf_th, w_th, linestyle = ":")
plt.legend(["dynamic", "static", "static3"])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.savefig("noiseless.pdf")
plt.figure()

plot_case(separationsDyn, "dynamic", 0, srf_th, w_th, linestyle = "-")
plot_case(separationsDyn, "static", 0, srf_th, w_th, linestyle = "-.")
plot_case(separationsDyn, "static3", 0, srf_th, w_th, linestyle = ":")
plt.legend(["dynamic", "static", "static3"])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.savefig("noiseless_DynNorm.pdf")
plt.figure()


### Noise in measurements case
## Dynamic reconstructions

# super resolution factor threshold for declaring accurate reconstruction.
srf_th = 40
# weight threshold to declare accurate reconstruction.
w_th = 0.1

styles = ["-", "--", "-.", ":", "-"]

for i in range(len(datanoise)+1):
	plot_case(separations, "dynamic", i, srf_th, w_th, linestyle = styles[i])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.legend(np.append(["0% L2 noise"],[str(int(100*datanoise[i])) + "% L2 noise" for i in range(len(datanoise))]))
plt.savefig("noisecomp-dyn.pdf")
plt.figure()

## Static reconstructions

# super resolution factor threshold for declaring accurate reconstruction.
srf_th = 40
# weight threshold to declare accurate reconstruction.
w_th = 0.1

styles = ["-", "--", "-.", ":", "-"]
for i in range(len(datanoise)+1):
	plot_case(separations, "static", i, srf_th, w_th, linestyle = styles[i])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.legend(np.append(["0% L2 noise"],[str(int(100*datanoise[i])) + "% L2 noise" for i in range(len(datanoise))]))
plt.savefig("noisecomp-static.pdf")
plt.figure()

## Static3 reconstructions

# super resolution factor threshold for declaring accurate reconstruction.
srf_th = 40
# weight threshold to declare accurate reconstruction.
w_th = 0.1

styles = ["-", "--", "-.", ":", "-"]
for i in range(len(datanoise)+1):
	plot_case(separations, "static3", i, srf_th, w_th, linestyle = styles[i])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.legend(np.append(["0% L2 noise"],[str(int(100*datanoise[i])) + "% L2 noise" for i in range(len(datanoise))]))
plt.savefig("noisecomp-static3.pdf")
plt.figure()

### Super resolution factor comparison
styles = ["-", "--", "-.", ":", "-"]
# weight threshold to declare accurate reconstruction.
w_th = 0.1
# Considered super resolution factors for comparison.
srf_thresholds = [1, 10, 100, 1000, 10000]
for i in range(len(srf_thresholds)):
	plot_case(separations, "dynamic", 0, srf_thresholds[i], w_th, linestyle = styles[i])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.legend(["SRF = "+str(int(srf_thresholds[i])) for i in range(len(srf_thresholds))])
plt.savefig("noiseless_SRF.pdf")
plt.figure()

###  Nonlinearity comparison
styles = ["-", "--", "-.", ":", "-"]
# weight threshold to declare accurate reconstruction.
w_th = 0.1
# super resolution factor threshold for declaring accurate reconstruction.
srf_th = 40

plot_case(separations,"dynamic", 0, srf_th, w_th, linestyle = styles[0])
for i in range(len(positionnoise)):
	plot_case(separations, "dynamic", 1+len(datanoise)+i, srf_th, w_th, linestyle = styles[i+1])
axes = plt.gca()
axes.set_ylim([0,1.05])
plt.legend(np.append(["second derivative = 0"],["second derivative = " + str(positionnoise[i]) for i in range(len(positionnoise))]))
plt.savefig("curvcomp.pdf")
plt.show()
