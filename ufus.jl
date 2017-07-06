push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

# Domain
x_max = 0.01

# Models
sigma = 0.001
filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
parameters = Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
model_static = Conv2d(parameters)
K = 2
v_max = 0.02
tau = 1/(K*30)
dynamic_parameters = DynamicConv2dParameters(K, tau, v_max, 20)
model_dynamic = DynamicConv2d(model_static, dynamic_parameters)

# Medium
n_im = 5
dx = 0.001
particles_m = [x_max * rand()]
particles_p = [x_max * rand()]
n_x = model_static.n_x
n_y = model_static.n_y
video = Matrix{Float64}(n_x * n_y, 0)
# Test static
#thetas = [0.0025 0.0075; 0.00183382 0.00408266]
#target = copy(phi(model_static, thetas, weights))
#function callback(old_thetas, thetas, weights, output, old_obj_val)
    ##evalute current OV
    #new_obj_val,t = loss(LSLoss(), output - target)
    ##println("gap = $(old_obj_val - new_obj_val)")
    #if old_obj_val - new_obj_val < 1E-4
        #return true
    #end
    #return false
#end
#(thetas_est,weights_est) = ADCG(model_static, LSLoss(), target, sum(weights), callback=callback, max_iters=200)

# Test dynamic
#thetas = [0.0025 0.0075; 0.0025 0.0075; 0.01 -0.01; 0.0 0.0]
#target = copy(phi(model_dynamic, thetas, weights))
#function callback(old_thetas, thetas, weights, output, old_obj_val)
    ##evalute current OV
    #new_obj_val,t = loss(LSLoss(), output - target)
    ##println("gap = $(old_obj_val - new_obj_val)")
    #if old_obj_val - new_obj_val < 1E-4
        #return true
    #end
    #return false
#end
#(thetas_est,weights_est) = ADCG(model_dynamic, LSLoss(), target, sum(weights), callback=callback, max_iters=200)

# Sequence
for i in 1:n_im
    thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'],
                  [(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'])
    weights = ones(size(thetas, 2))
    video = hcat(video, phi(model_static, thetas, weights))
    particles_m -= v_max/2 * tau
    particles_p += v_max/2 * tau
end


target = video[:]
function callback(old_thetas, thetas, weights, output, old_obj_val)
    #evalute current OV
    new_obj_val,t = loss(LSLoss(), output - target)
    #println("gap = $(old_obj_val - new_obj_val)")
    if old_obj_val - new_obj_val < 1E-4
        return true
    end
    return false
end
(thetas_est,weights_est) = ADCG(model_dynamic, LSLoss(), target, 2.0, callback=callback, max_iters=200)


#Plots
using PyCall
@pyimport matplotlib.pyplot as plt
frame = 3
plt.imshow(reshape(target[1:n_x*n_y], n_x, n_y))
plt.show()
