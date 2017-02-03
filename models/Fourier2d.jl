type Fourier2d <: SuperRes #1D Static model
    freqs :: Vector{Tuple{Int, Int}}
    filter :: Vector{Float64}
    x_max :: Float64
    z_max :: Float64
    approx_grid_x :: Vector{Float64}
    approx_grid_z :: Vector{Float64}
    bounds :: Tuple{Array{Float64}, Array{Float64}}
    function Fourier2d(x_max, z_max, filter, n_approx_x, n_approx_z)
        ind = find(filter)
        f_c_x = div(size(filter, 1) - 1, 2)
        f_c_z = div(size(filter, 2) - 1, 2)
        freqs_x = -f_c_x:f_c_x
        freqs_z = -f_c_z:f_c_z
        freqs2d = [(freq1, freq2) for freq1 in freqs_x, freq2 in freqs_z]
        freqs2d = freqs2d[ind]
        filter2d = filter[ind]
        grid_x = linspace(0, x_max, n_approx_x)
        grid_z = linspace(0, z_max, n_approx_z)
        return new(freqs2d,
                   filter2d,
                   x_max,
                   z_max,
                   grid_x,
                   grid_z,
                   ([0.0; 0.0], [x_max; z_max]))
    end
    function Fourier2d(x_max, filter, n_approx_x, n_approx_z)
        return Fourier2d(x_max, x_max, filter, n_approx_x, n_approx_z) 
    end
end
function setBounds(model :: Fourier2d, bounds)
    model.bounds = bounds
    model.approx_grid_x = linspace(bounds[1][1], bounds[2][1], length(model.approx_grid_x))
    model.approx_grid_z = linspace(bounds[1][2], bounds[2][2], length(model.approx_grid_z))
end

function psi(model :: Fourier2d, theta :: Vector{Float64})
  # This function computes the direct problem for a single point theta
  psi_cpx = vec(Complex128[model.filter[k] *
    exp(-2im * pi * (model.freqs[k][1] / model.x_max * theta[1] + model.freqs[k][2] / model.z_max * theta[2] )) for k in eachindex(model.freqs)])
  return [real(psi_cpx); imag(psi_cpx)]
end

function dpsi(model :: Fourier2d, theta :: Vector{Float64})
  # This function computes the gradient of psi
  dx1 = vec(Complex128[-model.filter[k] *
    model.freqs[k][1] / model.x_max * 2im * pi *
    exp(-2im * pi * (model.freqs[k][1]  * theta[1]/model.x_max +  model.freqs[k][2] * theta[2]/model.z_max )) for k in eachindex(model.freqs)])
  dx2 = vec(Complex128[-model.filter[k] *
    model.freqs[k][2] / model.z_max * 2im * pi *
    exp(-2im * pi * (model.freqs[k][1] * theta[1]/model.x_max + model.freqs[k][2] * theta[2]/model.z_max )) for k in eachindex(model.freqs)])
  return [[real(dx1); imag(dx1)] [real(dx2); imag(dx2)]]
end

function getStartingPoint(model :: Fourier2d, v :: Vector{Float64})
  # This function gets an initial value for the location
  values = Float64[dot(v,psi(model, [x; y])) for x = model.approx_grid_x, y = model.approx_grid_z]
  ind = indmin(values);
  i,j = ind2sub(values, ind)
  return vec([model.approx_grid_x[i]; model.approx_grid_z[j]])
end

parameterBounds(model :: Fourier2d) =
    model.bounds

dim(model :: Fourier2d) = 2

type DynamicFourier2d <: DynamicSuperRes #1D Space-time model
    static :: Fourier2d
    times :: Vector{Float64}
    v_max :: Float64
    grid_v :: Vector{Float64}
    bounds :: Tuple{Array{Float64}, Array{Float64}}
    DynamicFourier2d(static, v_max, tau, K, num_v = 20) = new(static,
                                                              linspace(-tau*K, tau*K, 2*K+1),
                                                              v_max,
                                                              linspace(-v_max, v_max, num_v),
                                                              ([0; 0; -v_max;v_max], [static.x_max; static.z_max; v_max; v_max]))

end

function getStartingPoint(model :: DynamicFourier2d, v :: Vector{Float64})
    # This function gets an initial value for the location
    vals = Float64[dot(v,psi(model, [x; y; vel_x; vel_z])) for x = model.static.approx_grid_x, y = model.static.approx_grid_z, vel_x = model.grid_v, vel_z = model.grid_v]
    ind = indmin(vals);
    i,j,k,l = ind2sub(vals, ind)
    return [model.static.approx_grid_x[i]; model.static.approx_grid_z[j]; model.grid_v[k]; model.grid_v[l]]
end

parameterBounds(model :: DynamicFourier2d) =
    model.bounds
