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
@everywhere function generate_and_reconstruct(dx, dv, iter)
    ### This function generates a test case and verifies that the reconstruction works
    ### Both in the static and dynamic caseska
    ok_dynamic = 0.0
    ok_static = 0.0
    for i in 1:iter
        (thetas, weights) = test_case(dx, dv)
        (thetas_est, weights_est) = run_simulation(model_static, thetas, weights)
        thetas_est = thetas_est[:, weights_est .> threshold_weight]
        weights_est = weights_est[weights_est .> threshold_weight]
        if (length(thetas_t) == length(thetas_est))
            corres = match_points(thetas, thetas_est)
            dist_x = norm(thetas_t[1:d, :] - thetas_est[1:d, corres], Inf)
            if (dist_x < threshold)
                temp_static = true
                break
            end
        end
        ok_static += temp_static
    end
    println("dx = ", dx, ", dv = ", dv, ", static: ", ok_static/iter, ", dynamic: ", ok_dynamic/iter)
    return ok_static/iter
end
DX = [dx for dx = vec_dx, dv = vec_dv]
DV = [dv for dx = vec_dx, dv = vec_dv]
res_static = reshape(pmap((dx, dv) -> generate_and_reconstruct(dx, dv, iter_mc), DX, DV), size(DX))

# Save results
id = string(now())
id = split(id, '.')[1]
id = replace(id, ':', '-')
mkdir(string("./results/", id))
cd(string("./results/", id))
@pyimport numpy
x = convert(Array{Float64}, vec_dv * tau * 2*K)
y = convert(Array{Float64}, vec_dx)
numpy.save("x.npy", x)
numpy.save("y.npy", y)
numpy.save("res_static.npy", res_static)
iostream = open("parameters.jl", "w")
write(iostream, parameters_str)
close(iostream)

#Generate a figure
@pyimport matplotlib as mpl
mpl.use("Agg")
@pyimport matplotlib.pyplot as plt
plt.figure()
plt.imshow(res_static, interpolation="none",extent=[minimum(x); maximum(x); maximum(y); minimum(y)])
plt.xlim((minimum(x), maximum(x)))
plt.ylim((minimum(y), maximum(y)))
plt.xlabel("velocity * T")
plt.ylabel("distance")
plt.savefig("static.png")

#Return to origin
cd("../..")
