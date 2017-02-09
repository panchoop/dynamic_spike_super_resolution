push!(LOAD_PATH, "./models")
using PyCall
@everywhere using SparseInverseProblems
if length(ARGS) > 0
    parameters_file = ARGS[1]
else
    parameters_file = "parameters.jl"
end
@everywhere using SuperResModels
@everywhere do_allfreqs = false
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
@everywhere function generate_and_reconstruct()
    ### This function generates a test case and verifies that the reconstruction works
    ### Both in the static and dynamic caseska
    ok_dynamic = false
    ok_static = false
    ok_allfreqs = false
    (thetas, weights) = test_case()
    d = dim(model_dynamic)
    (thetas_est, weights_est) = run_simulation(model_dynamic, thetas, weights, noise_level_data, noise_level_position)
    thetas_est = thetas_est[:, weights_est .> threshold_weight]
    weights_est = weights_est[weights_est .> threshold_weight]
    if (length(thetas) == length(thetas_est))
        corres = match_points(thetas, thetas_est)
        dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
        dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
        if (dist_x < threshold && model_dynamic.times[end]*dist_v < threshold)
            ok_dynamic = true
        end
    end
    temp_static = false
    if do_static
        for k = 1:length(model_dynamic.times)
            thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
            (thetas_est, weights_est) = run_simulation(model_static, thetas_t, weights, noise_level_data, noise_level_position)
            thetas_est = thetas_est[:, weights_est .> threshold_weight]
            weights_est = weights_est[weights_est .> threshold_weight]
            if (length(thetas_t) == length(thetas_est))
                corres = match_points(thetas_t, thetas_est)
                dist_x = norm(thetas_t - thetas_est[:, corres], Inf)
                if (dist_x < threshold)
                    temp_static = true
                    break
                end
            end
        end
        ok_static = temp_static
    end
    if do_allfreqs
        (thetas_est, weights_est) = run_simulation(model_allfreqs, thetas, weights, noise_level_data, noise_level_position)
        if (length(thetas_est) > 0)
            thetas_est = thetas_est[:, weights_est .> threshold_weight]
            weights_est = weights_est[weights_est .> threshold_weight]
        end
        if (length(thetas) == length(thetas_est))
            corres = match_points(thetas, thetas_est)
            dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
            dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
            if (dist_x < threshold && model_dynamic.times[end]*dist_v < threshold)
                ok_allfreqs = true
            end
        end
    end
    dx = minimum([abs(thetas[1, i] - thetas[1, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
    dv = minimum([abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
    println("dx = ", dx, ", dv = ", dv, ", static: ", ok_static, ", dynamic: ", ok_dynamic, ", afreq: ", ok_allfreqs)
    return (dx, dv, ok_static, ok_dynamic, ok_allfreqs)
end
res = pmap(x -> generate_and_reconstruct(), 1:num_trials)
dx = [x[1] for x = res]
dv = [x[2] for x = res]
res_static = [x[3] for x = res]
res_dynamic = [x[4] for x = res]
res_allfreqs = [x[5] for x = res]

# Save results
id = string(now())
id = split(id, '.')[1]
id = replace(id, ':', '-')
mkdir(string("./results/", id))
cd(string("./results/", id))
@pyimport numpy
x = convert(Array{Float64}, dv * tau * 2*K)
y = convert(Array{Float64}, dx)
numpy.save("x.npy", x)
numpy.save("y.npy", y)
numpy.save("res_static.npy", res_static)
numpy.save("res_dynamic.npy", res_dynamic)
numpy.save("res_allfreqs.npy", res_allfreqs)
iostream = open("parameters.jl", "w")
write(iostream, parameters_str)
close(iostream)

#Generate a figure
@pyimport matplotlib as mpl
mpl.use("Agg")
@pyimport matplotlib.pyplot as plt
function binary_scatter(x, y, bin)
    x_pos = x[bin]
    y_pos = y[bin]
    nbin = [!x for x in bin]
    x_neg = x[nbin]
    y_neg = y[nbin]
    plt.scatter(x_pos, y_pos, c="red",  label="ok",
                alpha=0.3, edgecolors="none")
    plt.scatter(x_neg, y_neg, c="blue", label="not ok",
                alpha=0.3, edgecolors="none")
    plt.xlim((minimum(x), maximum(x)))
    plt.ylim((minimum(y), maximum(y)))
end
if (do_static)
    plt.figure()
    binary_scatter(x, y, res_static)
    plt.savefig("static.png")
end
plt.figure()
binary_scatter(x, y, res_dynamic)
plt.savefig("dynamic.png")
if (do_allfreqs)
    plt.figure()
    binary_scatter(x, y, res_allfreqs)
    plt.savefig("allfreqs.png")
end

#Return to origin
cd("../..")
