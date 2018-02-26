### Domain [mm]
x_max = 1

### Models ###
# The considered PSF for the considered imaging tecnique
	# sigma also defines the resolution of the imaging system.
	sigma = 0.04
	filter = (x, y) -> exp(-(x^2 + y^2)/2/sigma^2)
	filter_dx = (x, y) -> -x/sigma^2*filter(x, y)
	filter_dy = (x, y) -> -y/sigma^2*filter(x, y)
# The number of time samples for reconstruction: 2K+1 
	K = 2
# the maximum allowed speed for any considered particle
	v_max = 15 #[mm/s]
# The sampling rate
	tau = 1/500 

### Noise ###
sigma_noise = 0.01

### Medium ###
# total number of the experimets measurements (gives the length of the experience)
	n_im = 1000
# Vessels separation
	dx = 0.02

### Inverse problem ###
# Considered norm to minimize
	lp_norm = 1
# Threshold on when to consider that there was a difference on the number of particles
# at two consecutive time steps.
	jump_threshold = 0.1

### Sequence ###
# Probability of a particle to activate or deactivate
	activation_probability=0.02

