abstract DynamicSuperRes <: SuperRes

dim(model :: DynamicSuperRes) = dim(model.static)

function psi(model :: DynamicSuperRes, theta :: Vector{Float64})
        # This function computes the direct problem for a single point theta
        d = dim(model)
        return vcat([psi(model.static, theta[1:d] + t * theta[d+1:end]) for t in model.times]...)
end

function dpsi(model :: DynamicSuperRes, theta :: Vector{Float64})
        # This function computes the gradient of psi
        d = dim(model)
        dx = vcat([dpsi(model.static, theta[1:d] + t * theta[d+1:end]) for t in model.times]...)
        dv = vcat([t * dpsi(model.static, theta[1:d] + t * theta[d+1:end]) for t in model.times]...)
        return [dx dv]
end
