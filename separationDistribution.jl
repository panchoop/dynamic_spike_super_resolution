# Script to obtain the distribution of separations for random particles

push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
using SparseInverseProblems
using SuperResModels
using TestCases
using Utils

using PyCall
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt


### Static parameters and container for use.
f_c = 20
x_max = 1.0
num_x = 10*f_c
filter_x =  ones(2*f_c+1);
model_static = SuperResModels.Fourier1d(f_c, x_max, filter_x, num_x)
### Dynamic parameters and container for use.
K = 2
tau = 1.0/(K*1.0)
# the maximum velocity on the domain is given by x_max/2 + K tau v_max = 1
v_max = 0.5
num_v = 10
model_dynamic = SuperResModels.DynamicFourier1d(model_static, v_max, tau, K, num_v*K)

# How particles are generated
test_case = () -> TestCases.cloud_1d_full(x_max, v_max, 1, 1, K, tau, rand(4:10))

### We generate particles and we evaluate their separation
N = 10000000
separations = zeros(N)
for i = 1:N
    (thetas, weights) = test_case()
    separations[i] = Utils.separation_val(thetas,K,tau,x_max)
end

#plt.figure()
#plt.hist(separations,1000)
#plt.show()

# Spatial resolution of this approximation
M = 1000
if x_max != 1.0
    warn(" Beware, the interval where we decided to consider the separations was tuned
    for the specific case of x_max equal to 1, since you modified this value, check if
        x_max/10 is adequate for your use, just plot an histogram of the separations.")
end
binss = linspace(0,x_max/10,M)
vals = np.histogram(separations, bins = binss )
vals = vals[1]/(binss[2]-binss[1])/N

println(vals[1])
println(vals[end])
plt.figure()
plt.plot(binss[1:end-1], vals)
plt.show()

np.save("data/separationDistribVal.npy", vals)
np.save("data/separationDistribBins.npy", binss)
