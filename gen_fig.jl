push!(LOAD_PATH, "./models")
using PyCall
@everywhere using SparseInverseProblems
if length(ARGS) > 0
    parameters_file = ARGS[1]
else
    parameters_file = "parameters.jl"
end
@everywhere using SuperResModels
@eval @everywhere parameters_file=$parameters_file
println("Using parameter file: ", parameters_file)
@everywhere begin
    include("TestCases.jl")
    include("utils.jl")
    include(parameters_file)
end
iostream = open(parameters_file, "r")
parameters_str = read(iostream)
close(iostream)

id = string(now())
id = split(id, '.')[1]
id = replace(id, ':', '-')
mkdir(string("./figures/", id))
cd(string("./figures/", id))
iostream = open("parameters.jl", "w")
write(iostream, parameters_str)
close(iostream)
dx = vec_dx[end]
dv = vec_dv[end]
(thetas, weights) = test_case(dx, dv)
(thetas, weights) = measure_noise(thetas, weights, dx*noise_level_x, dv*noise_level_v, noise_level_weights)
@pyimport numpy
@pyimport matplotlib as mpl
mpl.use("Agg")
@pyimport matplotlib.pyplot as plt
plt.figure()
d = div(size(thetas, 1), 2);
if d == 1
    plt.scatter(thetas[1, :], zeros(size(thetas, 2)))
    plt.quiver(thetas[1, :], zeros(size(thetas, 2)), thetas[2, :], zeros(size(thetas, 2)))
else
    plt.scatter(thetas[1,:], thetas[2, :])
    plt.quiver(thetas[1, :], thetas[2, :], thetas[3, :], thetas[4, :])
end
plt.savefig("figure.png")
#Return to origin
cd("../..")
