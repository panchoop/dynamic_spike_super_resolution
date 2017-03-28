push!(LOAD_PATH, "./models")
using PyCall
@everywhere using SparseInverseProblems
@everywhere using SuperResModels
@everywhere begin
    include("TestCases.jl")
    include("utils.jl")
    # Static parameters
    f_c = 10
    x_max = 0.02
    pix_size = 0.5 * x_max/f_c
    num_x = 10*f_c
    filter_x =  ones(2*f_c+1);
    model_static = Fourier1d(f_c, x_max, filter_x, num_x)
    # Dynamic parameters
    K = 2
    tau = 1.0/(K*30)
    v_max = 0.15
    num_v = 10
    model_dynamic = DynamicFourier1d(model_static, v_max, tau, K, num_v*K)
    # Particles
    test_case = () -> cloud_1d(x_max, v_max, 5)
    # Number of points
    num_trials = 3000
    # Threshold to accept/reject reconstruction
    threshold = 1e-4
    threshold_weight = 1e-1
    # Test parameters
    noises_measure = linspace(0, 0.2, 5)
    noises_position = linspace(0, 3e-5, 5)
    function generate_and_reconstruct(noise_level_data, noise_level_position)
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
        norm_1 = minimum([abs(thetas[1, i] - thetas[1, j]) + model_dynamic.times[end]* abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
        return (norm_1, ok_static, ok_dynamic)
    end
end
@pyimport matplotlib as mpl
mpl.use("Agg")
@pyimport matplotlib.pyplot as plt
num_bins = 10
bins = linspace(0, 0.004, num_bins + 1)
res_static_all = zeros(length(noises_measure), num_bins)
res_dynamic_all = zeros(length(noises_measure), num_bins)
i = 1
for noise_measure in noises_measure
    res = pmap(x -> generate_and_reconstruct(noise_measure, 0.0), 1:num_trials)
    res_dynamic = [x[3] for x = res]
    res_static = [x[2] for x = res]
    norm_1  = [x[1] for x = res]
    for j in 1:num_bins
        index = find([norm >= bins[j] && norm < bins[j+1] for norm in norm_1])
        res_dynamic_bin = convert(Array{Float64}, res_dynamic[index])
        res_static_bin = convert(Array{Float64}, res_static[index])
        norm_bin = norm_1[index]
        res_static_all[i, j] = sum(res_static_bin) / length(norm_bin)
        res_dynamic_all[i, j] = sum(res_dynamic_bin) / length(norm_bin)
    end
    i = i+1
end
id = string(now())
id = split(id, '.')[1]
id = replace(id, ':', '-')
mkdir(string("./results/", id))
cd(string("./results/", id))
plt.figure()
plt.plot(res_static_all')
plt.savefig("static_measure.png")
plt.figure()
plt.plot(res_dynamic_all')
plt.savefig("dynamic_measure.png")
i = 1
res_static_all = zeros(length(noises_position), num_bins)
res_dynamic_all = zeros(length(noises_position), num_bins)
for noise_position in noises_position
    res = pmap(x -> generate_and_reconstruct(0.0, noise_position), 1:num_trials)
    res_dynamic = [x[3] for x = res]
    res_static = [x[2] for x = res]
    norm_1  = [x[1] for x = res]
    for j in 1:num_bins
        index = find([norm >= bins[j] && norm < bins[j+1] for norm in norm_1])
        res_dynamic_bin = convert(Array{Float64}, res_dynamic[index])
        res_static_bin = convert(Array{Float64}, res_static[index])
        norm_bin = norm_1[index]
        res_static_all[i, j] = sum(res_static_bin) / length(norm_bin)
        res_dynamic_all[i, j] = sum(res_dynamic_bin) / length(norm_bin)
    end
    i = i+1
end
plt.figure()
plt.plot(res_static_all')
plt.savefig("static_position.png")
plt.figure()
plt.plot(res_dynamic_all')
plt.savefig("dynamic_position.png")
cd("../..")
