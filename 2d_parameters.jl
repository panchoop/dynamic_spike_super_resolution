### Domain ###
x_max = 0.01

### Models ###
# The considered PSF for the considered imaging tecnique
	sigma = 0.0004
	filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
	filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
	filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
# The number of time samples for reconstruction: 2K+1 
	K = 2
# the maximum allowed speed for any considered particle
	v_max = 0.05
# The sampling rate
	tau = 1/200

### Noise ###
sigma_noise = 0.01

### Medium ###
# total number of the experimets measurements (gives the length of the experience)
	n_im = 2000
# Spatial resolution
	dx = 0.0002

### Inverse problem ###
# Considered norm to minimize
	lp_norm = 1
# Threshold on when to consider that there was a difference on the number of particles
# at two consecutive time steps.
	jump_threshold = 0.2

### Sequence ###
# Probability of a particle to activate or deactivate
	activation_probability=0.02
# Function that gives the position of the new activated particle.
initial_position_generator = 
    () -> x_max/2 + (1-2*rand()) * x_max/4 
	# obs: new particles are not generated all over [0,x_max], but rather
	# in a subinterval, to ensure that all generated particles
	# will be contain in our domain Omega. To ensure this, it is 
	# required that x + \tau K v_max belongs to [0, x_max].
	# for both positive and negative K's. 
