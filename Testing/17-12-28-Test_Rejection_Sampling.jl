# Test to see if the rejection algorithm is working !

push!(LOAD_PATH, "../models")
push!(LOAD_PATH, "../")
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
N = 10000
separations = zeros(N)
dataFolder = "../data"
for i = 1:N
    (thetas, weights) = Utils.Rejection_sampling(test_case,dataFolder, K,tau,x_max)
    separations[i] = Utils.separation_val(thetas,K,tau,x_max)
end

plt.figure()
plt.hist(separations,100)
plt.show()
