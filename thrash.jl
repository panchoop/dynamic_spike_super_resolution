push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

# Get time of script start
now_str = string(now())
now_str = replace(now_str, ":", "-")
now_str = replace(now_str, ".", "-")

# Data folder
data_folder = "data/2Dsimulations/"*now_str

video = rand()
frame_norms = rand()
jumps = rand()

### Save the simulated data ###
using PyCall
@pyimport numpy as np
mkdir(data_folder)
cp("ufus_parameters.jl", string(data_folder, "/ufus_parameters.jl"))
cd(data_folder)
np.save("video", video)
np.save("frame_norms", frame_norms)
np.save("jumps", jumps)
cd("..")
