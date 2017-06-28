push!(LOAD_PATH, "./models")
using PyCall
@everywhere using SparseInverseProblems
@everywhere using SuperResModels
@everywhere begin

    include("TestCases.jl")
    include("utils.jl")
    # Static parameters
    f_c = 20
    x_max = 0.01
    pix_size = 0.5 * x_max/f_c
    num_x = 10*f_c
    filter_x =  ones(2*f_c+1);
    model_static = Fourier1d(f_c, x_max, filter_x, num_x)
    # Dynamic parameters
    K = 2
    tau = 1.0/(K*30)
    v_max = 0.05
    num_v = 10
    model_dynamic = DynamicFourier1d(model_static, v_max, tau, K, num_v*K)
    # Particles
    test_case = () -> begin
        n_points = rand(4:10)
        (thetas, weights) = cloud_1d(x_max, v_max, n_points)
        (thetas, weights)
    end 
    # Number of points
    num_trials = 1000
    # Threshold to accept/reject reconstruction
    threshold = 1e-4
    threshold_weight = 1e-1
    # Test parameters
    noises_data = linspace(0, 0.1, 4)
    noises_data = noises_data[2:end]
    noises_position = linspace(0, 1e-2, 6)
    noises_position = noises_position[2:end]
    function generate_and_reconstruct_dynamic(thetas, weights, noise_data, noise_position)
        d = dim(model_dynamic)
        (thetas_est, weights_est) = try run_simulation(model_dynamic, thetas, weights, noise_data, noise_position) catch ([0], [0]) end
        thetas_est = thetas_est[:, weights_est .> threshold_weight]
        weights_est = weights_est[weights_est .> threshold_weight]
        if (length(thetas) == length(thetas_est))
            corres = match_points(thetas, thetas_est)
            dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
            dist_v = norm(thetas[d+1:end, :] - thetas_est[d+1:end, corres], Inf)
            return (x_max / f_c / dist_x, x_max / f_c / dist_v / model_dynamic.times[end])
            end
            return 0.0, 0.0
        end
        function generate_and_reconstruct_static(thetas, weights, noise_data, noise_position)
            d = dim(model_dynamic)
            (thetas_est, weights_est) = try 
                run_simulation(model_static, thetas, weights, noise_data, noise_position)
            catch
                warn("fail")
                ([0], [0])
            end
            thetas_est = thetas_est[:, weights_est .> threshold_weight]
            weights_est = weights_est[weights_est .> threshold_weight]
            if (length(thetas) == length(thetas_est))
                corres = match_points(thetas, thetas_est)
                dist_x = norm(thetas[1:d, :] - thetas_est[1:d, corres], Inf)
                return x_max / f_c / dist_x
            end
            return 0.0
        end
        function generate_and_reconstruct_all(noises_data, noises_position)
            ### This function generates a test case and verifies that the reconstruction works
            ### Both in the static and dynamic caseska
            ok_dynamic = false
            ok_static = false
            (thetas, weights) = test_case()
            d = dim(model_dynamic)
            results = zeros(1 + length(noises_data) + length(noises_position), 3)
            norm = minimum([abs(thetas[1, i] - thetas[1, j]) + model_dynamic.times[end]* abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
            dx = minimum([abs(thetas[1, i] - thetas[1, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
            dv = minimum([abs(thetas[2, i] - thetas[2, j]) + (i==j) for i in 1:size(thetas, 2), j in 1:size(thetas, 2)])
            # Dynamic case, no noise
            (results[1, 1], results[1, 2]) = generate_and_reconstruct_dynamic(thetas, weights, 0.0, 0.0)
            # Static case, no noise
            temp = 0
            for k = 1:length(model_dynamic.times)
                thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
                temp = max(temp, generate_and_reconstruct_static( thetas_t, weights, 0.0, 0.0))
            end
            results[1, 3] = temp
            i = 2
            for noise_data in noises_data
                # Dynamic case, noisy
                (results[i, 1], results[i, 2]) = generate_and_reconstruct_dynamic(thetas, weights, noise_data, 0.0)
                # Static case, noisy
                temp = 0.0
                for k = 1:length(model_dynamic.times)
                    thetas_t = to_static(thetas, model_dynamic.times[k], model_static.x_max)
                    temp = max(temp, generate_and_reconstruct_static(thetas_t, weights, noise_data, 0.0))
                end
                results[i, 3] = temp
                i = i + 1
            end
            for noise_position in noises_position
                (results[i, 1], results[i, 2]) = generate_and_reconstruct_dynamic(thetas, weights, 0.0, noise_position)
                results[i, 3] = 0.0
                i = i + 1
            end
            return dx, dv, norm, results
        end
    end
    results = pmap(x -> generate_and_reconstruct_all(noises_data, noises_position), 1:num_trials)
    results_array = vcat([result[4] for result in results]...)
    norm_array = vcat([result[3] for result in results])
    dx_array = vcat([result[1] for result in results]...)
    dv_array = vcat([result[2] for result in results])

    using PyCall
    @pyimport numpy as np
    np.save("datanoise.npy", noises_data)
    np.save("positionnoise.npy", noises_position)
    np.save("norm.npy", norm_array)
    np.save("dx.npy", dx_array)
    np.save("dv.npy", dv_array)
    np.save("results.npy", results_array)

    """
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
    """
