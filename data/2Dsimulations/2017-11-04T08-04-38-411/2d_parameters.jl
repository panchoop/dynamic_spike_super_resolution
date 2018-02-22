# Domain
x_max = 0.005

# Models
sigma = 0.0004
filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
K = 2
v_max = 0.05
tau = 1/200

# Noise
sigma_noise = 0.005

# Medium
n_im = 500
dx = 0.0002

# Inverse problem
lp_norm = 1
jump_threshold = 0.2

# Sequence
p = 0.05
initial_position_generator = 
() -> x_max*(1.2 * rand() - 0.1)
