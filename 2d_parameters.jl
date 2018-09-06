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

### Function describing a curved domain describing a vessel. At the beggining we impose
### a string line, that afterwards curves

# Elliptic extension of a vessel
using Interpolations
using Roots
using QuadGK

function vessel(t)
    # Straight line parameters and starting point
    line_len = 1/4
    start_pos = 1/4
    # Ellipse curve parameters
    x_semiax = x_max - start_pos*x_max
    y_semiax = x_max - line_len*x_max
    max_t = 2
    # when straight line
    if t < line_len*x_max
        return (start_pos*x_max, t)
    end
    # else, the ellipse
    tt = t - line_len*x_max
    ellipse_curve(tt) = (-x_semiax*cos(tt) + x_semiax, y_semiax*sin(tt))
    # We seek to obtain the parametrization with speed one.
    curve_deriv(tt) = (x_semiax*sin(tt), y_semiax*cos(tt))
    deriv_norm(tt) = sqrt(curve_deriv(tt)[1]^2 + curve_deriv(tt)[2]^2)
    arclength(tt) = quadgk(deriv_norm, 0, tt)
    # we invert the arclength
    t_sample = [t for t = linspace(0,max_t,1000)]
    arclength_sample = [arclength(t)[1] for t in t_sample]
    inv_interp = LinearInterpolation(arclength_sample, t_sample)
    natural_curve(tt) = ellipse_curve(inv_interp(tt))
    x0 = start_pos*x_max
    y0 = line_len*x_max
    return (natural_curve(tt)[1] + x0, natural_curve(tt)[2] + y0)
end

target_funct(t) = max(vessel(t)[1] - x_max*(1-bndry_sep), vessel(t)[2] - x_max*(1-bndry_sep))
impossible_time = 1.5
lower_time = 0
if target_funct(lower_time)*target_funct(impossible_time) > 0
    print("This is not going to work")
end
t_max = find_zero(target_funct, (lower_time, impossible_time), Bisection())
