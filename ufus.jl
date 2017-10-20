push!(LOAD_PATH, "./models")
using SuperResModels
using SparseInverseProblems

@everywhere begin
# Domain
x_max = 0.01

# Models
sigma = 0.0004
filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
parameters = SuperResModels.Conv2dParameters(x_max, x_max, filter, filter_dx, filter_dy, sigma, sigma, sigma, sigma)
model_static = SuperResModels.Conv2d(parameters)
K = 2
v_max = 0.05
tau = 1/200
dynamic_parameters = SuperResModels.DynamicConv2dParameters(K, tau, v_max, 20)
model_dynamic = SuperResModels.DynamicConv2d(model_static, dynamic_parameters)

# Medium
n_im = 400
dx = 0.0002
particles_m = [x_max * rand()]
particles_p = [x_max * rand()]
n_x = model_static.n_x
n_y = model_static.n_y
end
video = Matrix{Float64}(n_x * n_y, 0)

# Sequence
p = 0.05
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

cur_frames = []
frame_packs = []
for i in 1:n_im
end
@everywhere frame_norms = $frame_norms
@everywhere frame_packs = $frame_packs
@everywhere function from_pack(frames)
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
end

bundles = [(5*i+1):(5*i+5) for i in 0:(n_im/5 - 1)]
all_thetas = pmap(bundle -> from_pack(bundle), bundles)

# Reprojection error
for bundle in bundles
    target = video[:, bundle][:]
    reprojection = phi(model_dynamic, all_thetas[1:4,:], all_thetas[5,:])
    println(norm(target-reprojection))
end
