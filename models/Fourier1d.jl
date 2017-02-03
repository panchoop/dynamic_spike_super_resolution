type Fourier1d <: SuperRes #1D Static model
    freqs :: Vector{Int}
    filter :: Vector{Float64}
    x_max :: Float64
    approx_grid :: Vector{Float64}
    bounds :: Tuple{Vector{Float64},Vector{Float64}}
    function Fourier1d(f_c, x_max, filter = ones(2*f_c+1), n_approx = 10*f_c+1)
        ind = find(filter)
        freqs = -f_c:f_c
        filter = filter[ind]
        freqs = freqs[ind]
        new(freqs, filter, x_max,
            linspace(0, x_max),
            ([0.0], [x_max]))
    end
end

function setBounds(model :: Fourier1d, bounds)
    model.bounds = bounds
    model.approx_grid = linspace(bounds[1][1], bounds[2][1], length(model.approx_grid))
end

function psi(model :: Fourier1d, theta :: Vector{Float64})
    # This function computes the direct problem for a single point theta
    psi_cpx =vec(Complex128[model.filter[k] * exp(-2im * pi * model.freqs[k] / model.x_max* theta[1])                                   for k in eachindex(model.freqs)])
    return [real(psi_cpx); imag(psi_cpx)]
end

function dpsi(model :: Fourier1d, theta :: Vector{Float64})
    # This function computes the gradient of psi
    dx = vec(Complex128[-model.filter[k] * model.freqs[k]/model.x_max* 2im * pi *
                        exp(-2im * pi * model.freqs[k] / model.x_max * theta[1]) for k in eachindex(model.freqs)])
    return reshape([real(dx); imag(dx)], 2*length(model.freqs), 1)
end

function getStartingPoint(model :: Fourier1d, v :: Vector{Float64})
    # This function gets an initial value for the location
    vals = Float64[dot(v,psi(model, [x])) for x = model.approx_grid]
    ind = indmin(vals);
    return [model.approx_grid[ind]]
end

parameterBounds(model :: Fourier1d) = model.bounds # Bounds for the parameters
dim(model :: Fourier1d) = 1

type DynamicFourier1d <: DynamicSuperRes #1D Space-time model
    static :: Fourier1d
    times :: Vector{Float64}
    v_max :: Float64
    grid_v :: Vector{Float64}
    bounds :: Tuple{Vector{Float64},Vector{Float64}}
    DynamicFourier1d(static, v_max, tau, K, num_v = 20) = new(static,
                                                              linspace(-tau*K, tau*K, 2*K+1),
                                                              v_max,
                                                              linspace(-v_max, v_max, num_v),
                                                              ([0; -v_max], [static.x_max; v_max]))
end
function setBounds(model :: DynamicFourier1d, bounds)
    setBounds(model.static, (bounds[1][1], bounds[2][1]))
    model.bounds = bounds
    model.grid_v = linspace(bounds[1][2], bounds[2][2], length(grid_v))
end

function getStartingPoint(model :: DynamicFourier1d, v :: Vector{Float64})
    # This function gets an initial value for the location
    vals = Float64[dot(v,psi(model, [x; vel])) for x = model.static.approx_grid, vel = model.grid_v]
    ind = indmin(vals);
    i,j = ind2sub(vals, ind)
    return [model.static.approx_grid[i]; model.grid_v[j]]
end

parameterBounds(model :: DynamicFourier1d) = model.bounds
