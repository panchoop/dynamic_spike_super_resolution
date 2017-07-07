push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

@everywhere begin
# Domain
x_max = 0.02

# Models
sigma = 0.001
filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
parameters = SuperResModels.Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
model_static = SuperResModels.Conv2d(parameters)
K = 2
v_max = 0.02
tau = 1/(K*30)
dynamic_parameters = SuperResModels.DynamicConv2dParameters(K, tau, v_max, 20)
model_dynamic = SuperResModels.DynamicConv2d(model_static, dynamic_parameters)

# Medium
n_im = 1000
dx = 0.001
particles_m = [x_max * rand()]
particles_p = [x_max * rand()]
n_x = model_static.n_x
n_y = model_static.n_y
end
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
p = 0.1
for i in 1:n_im
    thetas = hcat([(x_max/2 - dx) * ones(1, length(particles_m)); particles_m'],
                  [(x_max/2 + dx) * ones(1, length(particles_p)); particles_p'])
    weights = ones(size(thetas, 2))
    video = hcat(video, phi(model_static, thetas, weights))
    particles_m -= v_max/2 * tau
    particles_p += v_max/2 * tau
    remove = []
    for i in 1:length(particles_m)
        if rand() < p
            push!(remove, i)
        end
    end
    deleteat!(particles_m, remove)
    remove = []
    for i in 1:length(particles_p)
        if rand() < p
            push!(remove, i)
        end
    end
    deleteat!(particles_p, remove)
    if rand() < p
        push!(particles_m, rand() * x_max)
    end
    if rand() < p
        push!(particles_p, rand() * x_max)
    end
end
@everywhere video = $video

frame_norms = [norm(video[:,i]) for i in 1:n_im]
cur_frames = [1]
frame_packs = []
threshold = 0.1
for i in 2:n_im
    if maximum(abs.([frame_norms[i] - frame_norms[j] for j in cur_frames])) > threshold
        push!(frame_packs, cur_frames)
        cur_frames = [i]
    else
        push!(cur_frames, i)
    end
end
@everywhere frame_norms = $frame_norms
@everywhere frame_packs = $frame_packs
#target = video[:]
#function callback(old_thetas, thetas, weights, output, old_obj_val)
    ##evalute current OV
    #new_obj_val,t = loss(LSLoss(), output - target)
    ##println("gap = $(old_obj_val - new_obj_val)")
    #if old_obj_val - new_obj_val < 1E-4
        #return true
    #end
    #return false
#end
#(thetas_est,weights_est) = ADCG(model_dynamic, LSLoss(), target, 2.0, callback=callback, max_iters=200)


#Plots
using PyCall
#@pyimport matplotlib
#matplotlib.use("Agg")
#@pyimport matplotlib.pyplot as plt
#plt.plot(1:n_im, frame_norms, linestyle="dashed", color="black")
#for i in 1:length(frame_packs)
    #if length(frame_packs[i]) >= 5
        #plt.plot(frame_packs[i], frame_norms[frame_packs[i]], linewidth=2, color="red")
    #end
#end
#plt.xlabel("frame number", usetex=true)
#plt.ylabel("\$l^2\$ norm", usetex=true)
#plt.show()
#for i in 1:n_im
    #plt.imshow(reshape(video[:,i], n_x, n_y))
    #plt.savefig(string("movie/", i, ".jpg"))
#end
@everywhere function from_pack(pack)
    if length(pack) >= 5 && frame_norms[pack[1]] > 0
        frames = pack[1:5]
        target = video[:,frames][:]
        function callback(old_thetas, thetas, weights, output, old_obj_val)
            #evalute current OV
            new_obj_val,t = SparseInverseProblems.loss(SparseInverseProblems.LSLoss(), output - target)
            #println("gap = $(old_obj_val - new_obj_val)")
            if old_obj_val - new_obj_val < 1E-4
                return true
            end
            return false
        end
        (thetas_est,weights_est) = SparseInverseProblems.ADCG(model_dynamic, SparseInverseProblems.LSLoss(), target, 1.1*frame_norms[frames[1]], callback=callback, max_iters=200)
        println(thetas_est)
        if length(thetas_est) > 0
            return [thetas_est; weights_est']
        else
            return Matrix{Float64}(5,0)
        end
    else
        return Matrix{Float64}(5,0)
    end
end

all_thetas = pmap(i -> from_pack(frame_packs[i]), 1:length(frame_packs))
all_thetas = hcat(all_thetas...)
