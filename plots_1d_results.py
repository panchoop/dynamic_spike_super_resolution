import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

### Plotting options:

# If you want to get the plots on a specific experiment-folder or in all of them
PlotSpecificFolder = True
PlotAllFolders = not(PlotSpecificFolder)
specificFolder = ["data/1Dsimulations/"+"2018-02-19T04-35-58-77"]

### Success criteria parameters:
## Noiseless case:
srf_th_noiseless = 10000
w_th_noiseless = 0.01
## Measurement noise
srf_th_noise_dynamic = 40
w_th_noise_dynamic = 0.05
srf_th_noise_static = 40
w_th_noise_static = 0.05
## Curvature noise
srf_th_curvature = 1
w_th_curvature = 0.1
## SRF comparison
w_th_comparison = 0.01
srf_comparison = [1, 10, 100, 1000, 10000]

### Plotting parameters
# number of considered bins for plots
num_bins = 30

### Wanna see all the generated plots ?
visualize_plots = False

### Scripts to modify tikz files

def readEliminate(filename, text, nextLines):
	# Function to open a text file, and eliminate all lines that
	# star with the text line. It also eliminates the following 
	# predefined next lines.
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	counterNextLine = 0
	for i in d:
		if (i[1:len(text)+1] != text)and(counterNextLine==0):
			f.write(i)
		else:
			counterNextLine +=1
			if counterNextLine > nextLines:
				counterNextLine = 0
	f.truncate()
	f.close()

def readInsert(filename, text_start, text_insert):
	# Function to read lines and insert in the middle of a text 
	# an additional text.
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	for i in d:
		if i[1:len(text_start)+1] == text_start:
			f.write(i[0:len(text_start)+1]+text_insert+i[len(text_start)+1:len(i)])
		else:
			f.write(i)
	f.truncate()
	f.close()
	
def readNewline(filename, text, newline):
	# Function to insert a complete new line
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	for i in d:
		f.write(i)
		if i[1:len(text)+1] == text:
			f.write(newline)
	f.truncate()
	f.close()

def readRetrieve(filename, text):
	# Function that for a specific line of text, will get the  array that describes.
	# it assumes that the last element in text is '{'
	if text[len(text)-1] != '{':
		print('This is not working')
	f = open(filename,"r+")
	d = f.readlines()
	f.seek(0)
	values=''
	for i in d:
		f.write(i)
		if i[0:len(text)] == text:
			values= i[len(text):len(i)]
			print(values)
			break
	while values != '':
		print(values)
		if values[-1]!='}':
			values = values[1:-1]
		else:
			values=values[1:-1]
			break
	print(values)
	return map(float, values.split(","))


def fixTikz(filename, linewidth):
	## Function to fix the mistakes by tikztolatex
	# eliminate the grid legend errors
	# Include the appropiate line width
	# Eliminate some strange added lines
	readEliminate(filename, 'addlegendimage{no marker',0)
	readEliminate(filename, 'path [draw=black', 1)
	readInsert(filename, 'addplot [', 'line width = '+linewidth +' ,')
	readNewline(filename, 'begin{axis}[', 'name=ax,\n')
	readNewline(filename, 'end{axis}', '\\node at ($(ax.outer south east)+(3pt,10pt)$) {$\cdot \\nicefrac{1}{f_c}$};')

### Selecting the folders to generate plots from

x_max = 1.00
tau = 0.5
v_max = 0.5
f_c = 20
K = 2

if PlotAllFolders:
	subfolders = [x[0] for x in os.walk("data/1Dsimulations/")]
	subfolders = subfolders[1:len(subfolders)]

else:
	subfolders = specificFolder

# Function to plot the success rate as bins
def plot_success(norm, success, n_bins = num_bins, **kwargs):
		bins = np.linspace(np.percentile(norm, 2), np.percentile(norm, 85), n_bins)
		bins = np.linspace(np.min(norm), np.max(norm), n_bins)
		vals = np.zeros(len(bins) - 1)
		norm_success = norm[success]
		for i in range(len(bins) - 1):
			n_success = ((norm_success >= bins[i]) * (norm_success < bins[i+1])).sum()
			n_total = ((norm >= bins[i]) * (norm < bins[i+1])).sum()
			vals[i] = n_success / float(n_total)
		centers = 0.5 * (bins[0:len(bins) - 1] + bins[1:len(bins)])
		plt.plot(centers, vals, **kwargs)
		plt.ylim((-0.05, 1.05))
		plt.grid()
		plt.xlabel("$\Delta_{dyn}$", usetex=True)
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
		srf_dyn_v = x_max/dv_dyn/f_c/tau/K
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

# Actualy plotting process

for i in range(len(subfolders)):
	plt.close("all")
	print("Plotting in folder: ")
	print(subfolders[i])
	os.chdir(subfolders[i])
	## Un comment if you wanna just check one particular folder.
	#example = "2018-02-10T10-06-29-426"
	#folder = "data/1Dsimulations/"+example
	#os.chdir(folder)

	separations = np.load("separations.npy")
	separationsDyn = np.load("separationDynamic.npy")
	datanoise = np.load("datanoise.npy")
	positionnoise = np.load("positionnoise.npy")
	results = np.load("results.npy")
	results[results==0] = x_max
	N = len(separations)

	### Noiseless case
	# super resolution factor threshold for declaring accurate reconstruction.
	srf_th = srf_th_noiseless
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_noiseless

	plt.figure()
	plot_case(separations, "dynamic", 0, srf_th, w_th, linestyle = "-")
	plot_case(separations, "static", 0, srf_th, w_th, linestyle = "-.")
	plot_case(separations, "static3", 0, srf_th, w_th, linestyle = ":")
	plt.legend(["dynamic", "static", "static3"])
	axes = plt.gca()
	plt.savefig("noiseless.pdf")
	tikz_save("noiseless.tikz", figureheight="\\figureheight", figurewidth="\\figurewidth")
	print(readRetrieve("noiseless.tikz","xtick={"))
	fixTikz("noiseless.tikz",'1.5pt')
	if visualize_plots==True:
		plt.show()

	plt.figure()
	plot_case(separationsDyn, "dynamic", 0, srf_th, w_th, linestyle = "-")
	plot_case(separationsDyn, "static", 0, srf_th, w_th, linestyle = "-.")
	plot_case(separationsDyn, "static3", 0, srf_th, w_th, linestyle = ":")
	plt.legend(["dynamic", "static", "static3"])
	axes = plt.gca()
	plt.savefig("noiseless_DynNorm.pdf")
	tikz_save("noiseless_DynNorm.tikz", figureheight="\\figureheight", figurewidth="\\figurewidth")
	fixTikz("noiseless_DynNorm.tikz",'1.5pt')
	if visualize_plots==True:
		plt.show()


	### Noise in measurements case
	## Dynamic reconstructions

	# super resolution factor threshold for declaring accurate reconstruction.
	srf_th = srf_th_noise_dynamic
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_noise_dynamic

	styles = ["-", "--", "-.", ":", "-"]
	plt.figure()
	for i in range(len(datanoise)+1):
		plot_case(separations, "dynamic", i, srf_th, w_th, linestyle = styles[i])
	axes = plt.gca()
	plt.legend(np.append([r"$\alpha = 0$"],[ r"$\alpha = {0}$".format(str(datanoise[i]))  for i in range(len(datanoise))]))
	plt.savefig("noisecomp-dyn.pdf")
	tikz_save("noisecomp-dyn.tikz", figureheight="\\figureheight", figurewidth="\\figurewidth")
	fixTikz("noisecomp-dyn.tikz",'1.5pt')
	if visualize_plots==True:
		plt.show()

	## Static reconstructions

	# super resolution factor threshold for declaring accurate reconstruction.
	srf_th = srf_th_noise_static
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_noise_static

	styles = ["-", "--", "-.", ":", "-"]
	plt.figure()
	for i in range(len(datanoise)+1):
		plot_case(separations, "static", i, srf_th, w_th, linestyle = styles[i] )
	axes = plt.gca()
	plt.legend(np.append([r"$\alpha = 0$"],[ r"$\alpha = {0}$".format(str(datanoise[i])) for i in range(len(datanoise))]))
	plt.savefig("noisecomp-static.pdf")
	if visualize_plots == True:
		plt.show()

	## Static3 reconstructions

	# super resolution factor threshold for declaring accurate reconstruction.
	srf_th = srf_th_noise_static
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_noise_static

	styles = ["-", "--", "-.", ":", "-"]
	plt.figure()
	for i in range(len(datanoise)+1):
		plot_case(separations, "static3", i, srf_th, w_th, linestyle = styles[i])
	axes = plt.gca()
	plt.legend(np.append([r"$\alpha = 0$"],[ r"$\alpha = {0}$".format(str(datanoise[i]))  for i in range(len(datanoise))]))
	plt.savefig("noisecomp-static3.pdf")
	if visualize_plots == True:
		plt.show()

	### Super resolution factor comparison
	styles = ["-", "--", "-.", ":", "-"]
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_comparison
	# Considered super resolution factors for comparison.
	srf_thresholds = srf_comparison
	plt.figure()
	for i in range(len(srf_thresholds)):
		plot_case(separations, "dynamic", 0, srf_thresholds[i], w_th, linestyle = styles[i])
	axes = plt.gca()
	plt.legend(["SRF = "+str(int(srf_thresholds[i])) for i in range(len(srf_thresholds))])
	plt.savefig("noiseless_SRF.pdf")
	if visualize_plots == True:
		plt.show()

	###  Nonlinearity comparison
	# Plot visual curvature
	plt.figure()
	times = [k*tau for k in range(-2,3)]
	plt.plot(times, [v_max/2*t for t in times])
	plt.plot(times, [v_max/2*t + positionnoise[-1]/2 * t**2*v_max/2 for t in times])

	plt.legend([r"$\beta = 0$", r"$\beta = {0}$".format(str(positionnoise[-1]/2))])
	plt.savefig("curvature.pdf")
	if visualize_plots == True:
		plt.show()


	styles = ["-", "--", "-.", ":", "-"]
	plt.figure()
	# weight threshold to declare accurate reconstruction.
	w_th = w_th_curvature
	# super resolution factor threshold for declaring accurate reconstruction.
	srf_th = srf_th_curvature

	plot_case(separations,"dynamic", 0, srf_th, w_th, linestyle = styles[0])
	for i in range(len(positionnoise)):
		plot_case(separations, "dynamic", 1+len(datanoise)+i, srf_th, w_th, linestyle = styles[i+1])
	axes = plt.gca()
	plt.legend(np.append([r"$\beta = 0$"],[r"$\beta = {0}$".format(str(positionnoise[i]/2)) for i in range(len(positionnoise))]))
	plt.savefig("curvcomp.pdf")
	if visualize_plots == True:
		plt.show()
	os.chdir("../../..")
