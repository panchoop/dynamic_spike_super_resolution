type Conv2dParameters
    x_max :: Float64
    y_max :: Float64
    filter :: Function
    filter_dx :: Function
    filter_dy :: Function
    pix_size_x :: Float64
    pix_size_y :: Float64
    guess_error_x :: Float64
    guess_error_y :: Float64
end
type Conv2d <: SuperRes
    x_max :: Float64
    y_max :: Float64
    filter :: Function
    filter_dx :: Function
    filter_dy :: Function
    eval_grid_x :: Array{Float64}
    eval_grid_y :: Array{Float64}
    guess_grid_x :: Array{Float64}
    guess_grid_y :: Array{Float64}
    eval_array :: Array{Float64}
    eval_array_grad :: Array{Float64}
    guess_array :: Array{Float64}
    n_x :: Int64
    n_y :: Int64
    function Conv2d(params :: Conv2dParameters)
        eval_grid_x_tmp = 0:params.pix_size_x:params.x_max
        eval_grid_y_tmp = 0:params.pix_size_y:params.y_max
        eval_grid_x = [x for x in eval_grid_x_tmp, y in eval_grid_y_tmp][:]
        eval_grid_y = [y for x in eval_grid_x_tmp, y in eval_grid_y_tmp][:]
        guess_grid_x_tmp = 0:params.guess_error_x:params.x_max
        guess_grid_y_tmp = 0:params.guess_error_y:params.y_max
        guess_grid_x = [x for x in guess_grid_x_tmp, y in guess_grid_y_tmp][:]
        guess_grid_y = [y for x in guess_grid_x_tmp, y in guess_grid_y_tmp][:]
        eval_array = zeros(length(eval_grid_x))
        eval_array_grad = zeros(length(eval_grid_x), 2)
        guess_array = zeros(length(guess_grid_x))
        return new(params.x_max, params.y_max,
                   params.filter, params.filter_dx, params.filter_dy,
                   eval_grid_x, eval_grid_y,
                   guess_grid_x, guess_grid_y,
                   eval_array, eval_array_grad, guess_array,
                   length(eval_grid_x_tmp), length(eval_grid_y_tmp))
    end
end
function psi(model :: Conv2d, theta :: Vector{Float64}, w = 1.0, acc = false)
    # This function computes the direct problem for a single point theta
    if acc
        for i in 1:length(model.eval_grid_x)
            model.eval_array[i] += w * model.filter(model.eval_grid_x[i] - theta[1], model.eval_grid_y[i] - theta[2])
        end
    else
        for i in 1:length(model.eval_grid_x)
            model.eval_array[i] = w * model.filter(model.eval_grid_x[i] - theta[1], model.eval_grid_y[i] - theta[2])
        end
    end
    return model.eval_array
end

function dpsi(model :: Conv2d, theta :: Vector{Float64})
    # This function computes the gradient of psi and adds it to eval_array_grad field
    for i in 1:length(model.eval_grid_x)
        model.eval_array_grad[i, 1] = - model.filter_dx(model.eval_grid_x[i] - theta[1], model.eval_grid_y[i] - theta[2])
        model.eval_array_grad[i, 2] = - model.filter_dy(model.eval_grid_x[i] - theta[1], model.eval_grid_y[i] - theta[2])
    end
    return model.eval_array_grad
end

function phi(model :: Conv2d, parameters :: Matrix{Float64}, weights :: Vector{Float64})
    # Evaluates the direct problem in place
    fill!(model.eval_array, 0.0)
    for i in 1:length(weights)
        psi(model, vec(parameters[:, i]), weights[i], true) 
    end
    return model.eval_array
end

function getStartingPoint(model :: Conv2d, v :: Vector{Float64})
    # This function gets an initial value for the location
    map!((x,y) -> dot(v, psi(model, [x;y])), model.guess_array, model.guess_grid_x, model.guess_grid_y)
    ind = indmin(model.guess_array);
    res = [model.guess_grid_x[ind]; model.guess_grid_y[ind]]
    return res
end

parameterBounds(model :: Conv2d) = [0.0; 0.0], [model.x_max; model.y_max] # Bounds for the parameters
dim(model :: Conv2d) = 2

type DynamicConv2dParameters
    K :: Int64
    tau :: Float64
    v_max :: Float64
    num_v :: Int64
end

type DynamicConv2d<: SuperRes
    static :: Conv2d
    v_max :: Float64
    eval_grid_x :: Array{Float64}
    eval_grid_y :: Array{Float64}
    eval_grid_t :: Array{Float64}
    guess_grid_x :: Array{Float64}
    guess_grid_y :: Array{Float64}
    guess_grid_vx :: Array{Float64}
    guess_grid_vy :: Array{Float64}
    eval_array :: Vector{Float64}
    eval_array_grad :: Array{Float64}
    guess_array :: Array{Float64}
    function DynamicConv2d(static :: Conv2d, params::DynamicConv2dParameters)
        times = linspace(-params.K*params.tau, params.K*params.tau, 2*params.K + 1)
        eval_grid_x = [x for x in static.eval_grid_x, t in times][:]
        eval_grid_y = [y for y in static.eval_grid_y, t in times][:]
        eval_grid_t = [t for x in static.eval_grid_y, t in times][:]
        velocities = linspace(-params.v_max, params.v_max, params.num_v)
        guess_grid_x = [x for x in static.guess_grid_x, vx in velocities, vy in velocities][:]
        guess_grid_y = [y for y in static.guess_grid_y, vx in velocities, vy in velocities][:]   
        guess_grid_vx = [vx for y in static.guess_grid_y, vx in velocities, vy in velocities][:] 
        guess_grid_vy = [vy for y in static.guess_grid_y, vx in velocities, vy in velocities][:] 
        eval_array = zeros(length(eval_grid_x))
        eval_array_grad = zeros(length(eval_grid_x), 4)
        guess_array = zeros(length(guess_grid_x))
        return new(static, params.v_max,
                   eval_grid_x, eval_grid_y, eval_grid_t,
                   guess_grid_x, guess_grid_y, guess_grid_vx, guess_grid_vy,
                   eval_array, eval_array_grad, guess_array)
    end
end
function psi(model :: DynamicConv2d, theta :: Vector{Float64}, w = 1.0, acc=false)
    # This function computes the direct problem for a single point theta
    if acc
        for i in 1:length(model.eval_grid_x)
            model.eval_array[i] += w * model.static.filter(model.eval_grid_x[i] - theta[1] - theta[3]*model.eval_grid_t[i],
                                                           model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
        end
    else
        for i in 1:length(model.eval_grid_x)
            model.eval_array[i] = w * model.static.filter(model.eval_grid_x[i] - theta[1] - theta[3]*model.eval_grid_t[i],
                                                          model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
        end
    end
    return model.eval_array
end


function dpsi(model :: DynamicConv2d, theta :: Vector{Float64}, w = 1.0, acc=false)
    # This function computes the gradient of psi and adds it to eval_array_grad field
    for i in 1:length(model.eval_grid_x)
        model.eval_array_grad[i, 1] = - model.static.filter_dx(model.eval_grid_x[i] - theta[1] - theta[3]*model.eval_grid_t[i],
                                                             model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
        model.eval_array_grad[i, 2] = - model.static.filter_dy(model.eval_grid_x[i] - theta[2] - theta[3]*model.eval_grid_t[i],
                                                             model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
        model.eval_array_grad[i, 3] = - model.eval_grid_t[i] * model.static.filter_dx(model.eval_grid_x[i] - theta[1] - theta[3]*model.eval_grid_t[i],
                                                                                      model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
        model.eval_array_grad[i, 4] = - model.eval_grid_t[i] * model.static.filter_dy(model.eval_grid_x[i] - theta[1] - theta[3]*model.eval_grid_t[i],
                                                                                      model.eval_grid_y[i] - theta[2] - theta[4]*model.eval_grid_t[i])
    end
    return model.eval_array_grad
end

function phi(model :: DynamicConv2d, parameters :: Matrix{Float64}, weights :: Vector{Float64})
    # Evaluates the direct problem in place
    fill!(model.eval_array, 0.0)
    for i in 1:length(weights)
        psi(model, parameters[:, i], weights[i], true) 
    end
    return model.eval_array
end

function getStartingPoint(model :: DynamicConv2d, v :: Vector{Float64})
    # This function gets an initial value for the location
    map!((x,y, vx, vy) -> dot(v, psi(model, [x;y;vx;vy])), model.guess_array, model.guess_grid_x, model.guess_grid_y, model.guess_grid_vx, model.guess_grid_vy)
    ind = indmin(model.guess_array);
    return vec([model.guess_grid_x[ind]; model.guess_grid_y[ind]; model.guess_grid_vx[ind]; model.guess_grid_vy[ind]])
end

parameterBounds(model :: DynamicConv2d) = [0.0; 0.0; -model.v_max; -model.v_max], [model.static.x_max; model.static.y_max; model.v_max; model.v_max] # Bounds for the parameters
