abstract type SuperRes <: BoxConstrainedDifferentiableModel end

function to_static(thetas, t)
    d = div(size(thetas, 1),2)
    pts = thetas[1:d,:]
    velocities = thetas[d+1:2*d,:]
    pts_t = pts + velocities * t
    return pts_t
end

function is_in_bounds(model::SuperRes, thetas)
    bounds = parameterBounds(model)
    is_good = true
    for i = 1:size(thetas,1)
        for j = 1:size(thetas,2)
            if thetas[i,j] < bounds[1][i] || thetas[i,j] > bounds[2][i]
                return false
            end
        end
    end
    return true
end
