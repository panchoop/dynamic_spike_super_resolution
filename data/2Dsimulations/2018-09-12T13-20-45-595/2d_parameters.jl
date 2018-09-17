#
# 12-09-2018, full experiment with complex vessel network and 2 seconds exposure.
#

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
        particle_velocity = 1 #[mm/s]
# The sampling rate
	tau = 1/500

### Noise ###
sigma_noise = 0.01

### Medium ###
# total number of the experimets measurements (gives the length of the experience)
	n_im = 1000
# Vessels separation
	dx = 0.02
# microbubbles distance to booundary
	bndry_sep = 0.05

### Inverse problem ###
# Considered norm to minimize
	lp_norm = 1
# Threshold on when to consider that there was a difference on the number of particles
# at two consecutive time steps.
	jump_threshold = 0.1

### Sequence ###
# Probability of a particle to activate or deactivate
	activation_probability=0.02

# Building the vessel network
x0 = [0.1,0.1]
# arrays of t_max, relative direction, curve_type, parameters
curve1 = [0.3, -pi/6,1, "straight", [] ]
curve2 = [0.4, 0,1,"ellipse", [1,2,0.5]]
curve3 = [0.55, 0,1, "ellipse", [-1,2,1]]
curve4 = [0.45, pi/2,1, "ellipse", [-1,2,1]]
curve5 = [0.35, 0,1, "ellipse", [1,2,1]]

Vessel = Tree(curve1)      #curve 1
addchild(Vessel,1, curve2) #curve 2
addchild(Vessel,1, curve3) #curve 3
addchild(Vessel,2, curve4) #curve 4
addchild(Vessel,2, curve5) #curve 5

# The tree is transform to obtain the curves precisely,
# i.e. not defined relative to the neighbouring ones.
Absolute_Vessel(Vessel, 1, x0, 0)
