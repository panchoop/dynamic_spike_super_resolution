push!(LOAD_PATH, "./models")
using PyCall
using SparseInverseProblems
using SuperResModels
@pyimport matplotlib.pyplot as plt
x_max = 0.01
thetas = [0.0035 0.0048]
weights = [0.5; 0.5]
f_c = 10
#filter = x -> 1 + 2*sum([cos(k*2*pi*x/x_max) for k=1:f_c])
#filter_grad = x -> -2*sum([2*k*pi/x_max * sin(k*2*pi*x/x_max) for k=1:f_c])
#filter = x -> exp(-(x/x_max)^2/2*f_c^2)
#filter_grad = x -> -x/x_max*f_c^2 * filter(x)
pix_size = x_max / f_c / 2
dirichlet = Fourier1d(f_c, x_max)
v_max = 0.1
τ = 1/30
K = 1
dirichlet_dynamic = DynamicFourier1d(dirichlet, v_max, τ, K)
u = phi(dirichlet, thetas, weights)


function run_simulation(model, thetas, weights)
    if is_in_bounds(model, thetas)
        target = phi(model, thetas, weights)
        function callback(old_thetas, thetas, weights, output, old_obj_val)
            #evalute current OV
            new_obj_val,t = loss(LSLoss(), output - target)
            ##println("gap = $(old_obj_val - new_obj_val)")
            if old_obj_val - new_obj_val < 1E-4
                return true
            end
            return false
        end
        ## Inverse problem
        (thetas_est,weights_est) = ADCG(model, LSLoss(), target,sum(weights), callback=callback)
        return (thetas_est, weights_est)
    else
        warn("Out of bounds !")
        return ([], [])
    end
end

(thetas_est, weights_est) = run_simulation(dirichlet, thetas, weights)
println(thetas_est)
println(weights_est)
