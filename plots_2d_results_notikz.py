# -*- coding: utf-8 -*-

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#~ from matplotlib2tikz import save as tikz_save
#~ import helpToTikz as fix

## Folder/Example to load
#example = "/2017-11-03T09-40-41-038"
# example = "/2017-11-03T10-05-27-667"
# example = "/2017-11-04T08-04-38-411"
# example = "/2017-11-04T11-57-14-36"
# example = "/2018-02-26T11-02-23-312"
example = "/2018-09-05T15-15-37-072"
folder = "data/2Dsimulations"+example

### Parameters for reconstruction figure
# Threshold to accept or reject recostructions:
# the error threshold is how much measurement missmatch we tolerate.
threshold_error = 0.65
# The weight threshold is at which mass we are going to consider the 
# reconstruction a valid particle.
threshold_weight = 0.1
# Size of the plotter figure
sizeAmp = 50;

### Parameters for single frame
frameNum1 = 20
frameNum2 = frameNum1+4
frameNum3 = frameNum1+8

### If you wanna see each generated figure
seeFigs = True

os.chdir(folder)
# loading Simulated values
errors = np.load("errors.npy")
video = np.load("video.npy")

# Getting the x_max value, dimensions of the simulation
f = open("2d_parameters.jl","r")
d = f.readlines()
f.close()
for i in d:
    if i[0:len("x_max = ")] == "x_max = ":
        x_max = float(i[len("x_max = "):len(i)])

#~ def fixTikz(filename, scatter=False, picture=False):
    #~ ## Function to fix the mistakes by tikztolatex
    #~ # Weird automatically included lines
    #~ fix.readEliminate(filename, '\\path [draw=black', 1)
    #~ # Somehow it includes in the tikz file a weird "minus sign"
    #~ # that Latex doesn't likes
    #~ fix.readReplaceAll(filename,"−", "-")
    #~ if scatter==True:
        #~ fix.readInsert(filename, '\\addplot [', 'fill opacity = 0.4, mark size=0.5pt,')
        #~ fix.readReplaceAll(filename, "size=\\perpointmarksize",
                           #~ "")

#~ # Generating reconstruction figure
#~ plt.figure()
#~ all_thetas = np.zeros((5, 0))
#~ for i in range(len(errors)):
    #~ if errors[i] < threshold_error:
        #~ theta = np.load("thetas-" + str(i+1) + ".npy")
        #~ print("**************************")
        #~ print(theta)
        #~ print("**************************")
        #~ all_thetas = np.concatenate((all_thetas,
						  #~ theta[:,theta[4,:]>threshold_weight]), axis=1)

#~ plt.scatter(all_thetas[0,:], all_thetas[1,:], c=all_thetas[3,:], 
			#~ s=1*sizeAmp, alpha = 0.5)
#~ plt.colorbar()
#~ plt.xlim((0, x_max))
#~ plt.ylim((0, x_max))
#~ plt.savefig("superres.pdf")
#~ tikz_save("superres.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("superres.tikz",scatter=True)
#~ fix.readEliminate("superres.tikz", '\\path [draw=black', 1)
#~ if seeFigs:
    #~ plt.show()
#~ fix.readNewline("superres.tikz", "\\begin{axis}[", 
		     	#~ "colorbar style = { width = 0.2cm, at = {(1.15,0.5)},"
			    #~ +" anchor = east,},")
#~ fix.readReplaceAll("superres.tikz","superres", 
				   #~ "figures/superres")				    
			    
			    
# Generating B mode figure

# We crop the used cmap for the snapshots to create a new one
seismic_cmap = plt.cm.get_cmap('seismic')
colors = []
colors.append(seismic_cmap(0.0))
colors.append(seismic_cmap(0.5))
my_cmap = LinearSegmentedColormap.from_list('my_cmap',colors, 100)


plt.figure()
n_x = int(np.round(np.sqrt(video.shape[0])))
plt.pcolormesh(np.linspace(0, x_max, n_x), np.linspace(0, x_max, n_x),
			   np.reshape(np.sum(video, 1)/float(video.shape[1]),(n_x,n_x)),cmap = my_cmap)
plt.colorbar()
plt.savefig("bmode.pdf")
#~ tikz_save("bmode.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("bmode.tikz")
if seeFigs:
    plt.show()
#~ fix.readNewline("bmode.tikz", "\\begin{axis}[", 
				#~ "colorbar style = { width = 0.2cm, at = {(1.15,0.5)},"
			    #~ +" anchor = east,},")
#~ fix.readReplaceAll("bmode.tikz","bmode", 
				   #~ "figures/bmode")	

# To see a video of all the snapshots, set to True. It also helps to 
# decide which frame to choose for images.

if True:
    plt.figure()
    for i in range(video.shape[1]):
        plt.pcolormesh(np.linspace(0, x_max, n_x), 
    				   np.linspace(0, x_max, n_x), 
		     		   np.reshape(video[:,i],(n_x,n_x)), cmap = 'seismic')
        plt.colorbar()
        plt.clim((0,1.8))
        plt.title("frame: "+str(i))
        #plt.show()
        plt.pause(0.3)
        plt.clf()

# Generating different snapshots figure

plt.figure()
plt.pcolormesh(np.linspace(0, x_max, n_x), np.linspace(0, x_max, n_x),
			   np.reshape(video[:,frameNum1],(n_x,n_x)), cmap = 'seismic')
plt.clim((0,1.8))
plt.savefig("singleframe1.pdf")
if seeFigs:
    plt.show()
#~ tikz_save("singleframe1.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("singleframe1.tikz")
#~ fix.readReplace("singleframe1.tikz", "xticklabels", 
			    #~ "0.0,0.2,0.4,0.6,0.8,1.0", "0,0.2,0.4,0.6,0.8,1")
#~ fix.readReplaceAll("singleframe1.tikz","singleframe1", 
				   #~ "figures/singleframe1")			    

plt.figure()
plt.pcolormesh(np.linspace(0, x_max, n_x), np.linspace(0, x_max, n_x), 
			   np.reshape(video[:,frameNum2],(n_x,n_x)), cmap = 'seismic')
plt.clim((0,1.8))
plt.savefig("singleframe2.pdf")
if seeFigs:
    plt.show()
#~ tikz_save("singleframe2.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("singleframe2.tikz")
#~ fix.readReplace("singleframe2.tikz", "yticklabels", 
			    #~ "0.0,0.2,0.4,0.6,0.8,1.0", "")
#~ fix.readReplace("singleframe2.tikz", "xticklabels", 
			    #~ "0.0,0.2,0.4,0.6,0.8,1.0", "0,0.2,0.4,0.6,0.8,1")
#~ fix.readReplaceAll("singleframe2.tikz","singleframe2", 
				   #~ "figures/singleframe2")
				   
plt.figure()
plt.pcolormesh(np.linspace(0, x_max, n_x), np.linspace(0, x_max, n_x), 
			   np.reshape(video[:,frameNum3],(n_x,n_x)), cmap = 'seismic')
plt.colorbar()
plt.clim((0,1.8))
plt.savefig("singleframe3.pdf")
if seeFigs:
    plt.show()
#~ tikz_save("singleframe3.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("singleframe3.tikz")
#~ fix.readReplace("singleframe3.tikz", "yticklabels", 
			    #~ "0.0,0.2,0.4,0.6,0.8,1.0", "")
#~ fix.readReplace("singleframe3.tikz", "xticklabels", 
			    #~ "0.0,0.2,0.4,0.6,0.8,1.0", "0,0.2,0.4,0.6,0.8,1")
#~ fix.readNewline("singleframe3.tikz", "\\begin{axis}[", 
		     	#~ "colorbar style = { width = 0.2cm, at = {(1.15,0.5)},"
			    #~ +" anchor = east,},")
#~ fix.readReplaceAll("singleframe3.tikz","singleframe3", 
				   #~ "figures/singleframe3")

# Generating norm profile
plt.figure()
L2norms = np.sqrt(np.sum(np.power(video,2), 0))
plt.plot(np.arange(1,len(L2norms)+1), L2norms, linestyle = '--')
# Paint the close to constant cases
minSnapshot = 5
tolerance = 0.1
j = 1
i = 1

while i <= len(L2norms)-minSnapshot:
    while (j+i <= len(L2norms)-minSnapshot) & (np.abs(L2norms[i-1]-L2norms[i+j-1])<=tolerance):
        j = j +1
    if j-1 >= minSnapshot:
        plt.plot([i, i+j-1], [np.mean(L2norms[i-1:i+j-2]), 
		                           np.mean(L2norms[i-1:i+j-2])],color="r")
    i = i+j
    j = 1
plt.savefig("L2profile.pdf")
if seeFigs:
    plt.show()
#~ tikz_save("L2profile.tikz", figureheight="\\figureheight",
		  #~ figurewidth="\\figurewidth")
#~ fixTikz("L2profile.tikz")
