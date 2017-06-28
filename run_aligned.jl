push!(LOAD_PATH, "./models")
using PyCall
@everywhere using SparseInverseProblems
@everywhere using SuperResModels
@everywhere begin

    include("TestCases.jl")
    include("utils.jl")
    # Static parameters
    f_c = 40
    x_max = 0.01
    pix_size = 0.5 * x_max/f_c
    num_x = 10*f_c
    filter_x =  ones(2*f_c+1);
    model_static = Fourier1d(f_c, x_max, filter_x, num_x)
    # Dynamic parameters
    K = 1
    tau = 1.0/(K*30)
    v_max = 0.07
    num_v = 40
    model_dynamic = DynamicFourier1d(model_static, v_max, tau, K, num_v*K)
    # Particles
    test_case = () -> begin
        n_points = 3
        dx = 0.0007
        (thetas, weights) = aligned_1d(x_max, dx, n_points)
        (thetas, weights)
    end 
    function generate_and_reconstruct_dynamic(thetas, weights, noise_data, noise_position)
            (thetas_est, weights_est) = try run_simulation(model_dynamic, thetas, weights, noise_data, noise_position) catch ([0], [0]) end
            return thetas_est, weights_est
    end
end
thetas, weights = test_case()
thetas[2,1] = 0.0011
thetas[2,3] = 0.0011
t, w = generate_and_reconstruct_dynamic(thetas, weights, 0.012, 0.0)
using PyCall
@pyimport matplotlib.pyplot as plt
err = norm(phi(model_dynamic,t, w) - phi(model_dynamic, thetas, weights))/norm(phi(model_dynamic, thetas, weights))

plt.scatter(t[1,:], t[2,:], 10*w, alpha=0.5)
plt.scatter(thetas[1,:], thetas[2,:], 10*weights, alpha=0.5)
plt.show()
