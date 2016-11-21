immutable Conv1d <: SuperRes
    filter :: Function
    filter_grad :: Function
    eval_grid :: Array{Float64}
    approx_grid :: Array{Float64}
    psf_grid :: Matrix{Float64}
    x_max :: Float64
    Conv1d(filter, filter_grad, eval_grid, approx_grid, x_max) =
        new(filter,
            filter_grad,
            eval_grid,
            approx_grid,
            [filter(y - x) for x in approx_grid, y in eval_grid],
            x_max)
end

function psi(model :: Conv1d, theta :: Vector{Float64})
    # This function computes the direct problem for a single point theta
    return Float64[model.filter(x - theta[1]) for x in model.eval_grid]
end

function dpsi(model :: Conv1d, theta :: Vector{Float64})
    # This function computes the gradient of psi
    return Float64[-model.filter_grad(x - theta[1]) for x in model.eval_grid]
end

function getStartingPoint(model :: Conv1d, v :: Vector{Float64})
    # This function gets an initial value for the location
    return [model.approx_grid[indmin(model.psf_grid * v)]]
end

parameterBounds(model :: Conv1d) = [0.0], [model.x_max] # Bounds for the parameters
dim(model :: Conv1d) = 1

immutable DynamicConv1d<: DynamicSuperRes
    static :: Conv1d
    v_max :: Float64
    grid_v :: Vector{Float64}
    times :: Vector{Float64}
    DynamicConv1d(static, v_max, τ, K, num_v = 10) = new(static,
                                                          v_max,
                                                          linspace(-v_max, v_max, num_v),
                                                          -τ*K:τ:τ*K)
end
function getStartingPoint(model :: DynamicConv1d, v :: Vector{Float64})
    # This function gets an initial value for the location
    vals = Float64[dot(v,psi(model, [x; vel])) for x = model.static.eval_grid, vel = model.grid_v]
    ind = indmin(vals);
    i,j = ind2sub(vals, ind)
    return [model.static.eval_grid[i]; model.grid_v[j]]
end

parameterBounds(model :: DynamicConv1d) = [0.0; -model.v_max], [model.static.x_max; model.v_max] # Bounds for the parameters
