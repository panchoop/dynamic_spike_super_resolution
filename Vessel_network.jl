push!(LOAD_PATH, "./models")
push!(LOAD_PATH, ".")
module Vessel_network
# Module that implements methods to facilitate the generation and
# manipulation of blood vessel structures. 

using Interpolations
using Roots
using QuadGK

export Tree, Absolute_Vessel, displacement, evaluation

type TreeNode
    parent::Int
    children::Vector{Int}
    curveData::Array{Any,1}
end
type Tree
    nodes::Vector{TreeNode}
end
Tree(data) = Tree([TreeNode(0, Vector{Int}(), data)])
function addchild(tree::Tree, id::Int, curveData::Array{Any,1})
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))
    push!(tree.nodes, TreeNode(id, Vector{}(), curveData))
    child = length(tree.nodes)
    push!(tree.nodes[id].children, child)
    child
end
children(tree, id) = tree.nodes[id].children
parent(tree,id) = tree.nodes[id].parent
data(tree,id) = tree.nodes[id].curveData

function angle_direction(direction)
    return pi/2 - atan(direction[2],direction[1])
end

function segment_generator(Vessel, id)
    parameters = data(Vessel,id)
    x0 = parameters[1]
    tf = parameters[2]
    angle = parameters[3]
    speed = parameters[4]
    curve_type = parameters[5]
    params = parameters[6]
    if curve_type == "straight"
        f  = t -> [0 ; t]
        df = t -> [0 ; 1]
    elseif curve_type == "ellipse"
        if length(params)!= 3
            error("Incorrect number of input parameters")
        end
        side = params[1]
        x_semiax = params[2]
        y_semiax = params[3]
        g(t) = [x_semiax*(1-cos(t))*side; y_semiax*sin(t)]
        dg(t) = [x_semiax*side*sin(t); y_semiax*cos(t) ]
        # now we need to obtain the natural parametrization
            norm_derivative(t) = sqrt(dg(t)[1]^2 + dg(t)[2]^2)
            arclength(t) = quadgk(norm_derivative, 0, t)
            # invert the arc length
            max_sample = 1
            while arclength(max_sample)[1]-tf < 0
                max_sample = max_sample*2
            end
            t_sample = [t for t = linspace(0,max_sample,1000)]
            arclength_sample = [arclength(t)[1] for t in t_sample]
            inv_interp = LinearInterpolation(arclength_sample, t_sample)
        f = t -> g(inv_interp(t))
        df = t-> dg(inv_interp(t))/sqrt( dg(inv_interp(t))[1]^2 + dg(inv_interp(t))[2]^2 )
    else
        error("Not a valid curve type, find the correct names")
    end
    # Rotation matrix
    rotation_mat = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    # Translation, speed scalation, rotation
    segment(t) = rotation_mat*f(t*speed) + x0
    dsegment(t) = rotation_mat*df(t*speed)
    return [segment , dsegment]
end

function Absolute_Vessel(Vessel,id, x0, angle)
    datta = data(Vessel,id)
    datta[2] = datta[2]-angle
    prepend!(datta,[x0])
    Vessel.nodes[id].curveData = datta
    segt, dsegt = segment_generator(Vessel, id)
    new_angle = angle_direction(dsegt(datta[2]))
    new_position = segt(datta[2])
    for j in children(Vessel,id)
        Absolute_Vessel(Vessel,j, new_position, new_angle)
    end
end

# Function to move along the vessels, from a starting position
# (id, t) and a displacement dt
function displacement(Vessel, id, t, dt)
    # This method is not robust to big displacements ! 
    tf = data(Vessel,id)[2]
    if 0 <= t+dt && t+dt <= tf
        return (id, t+dt)
    elseif t+dt > tf
        next_vessel = children(Vessel,id)
        if isempty(next_vessel)
            return NaN
        else
        # Choose some children at random, assuming at most 2
            new_id = next_vessel[rand(1:2)]
            return (new_id, t+dt-tf)
        end
    else
        # backward case, t+dt < 0 
        new_id = parent(Vessel,id)
        if new_id == 0
            return NaN
        else
            new_tf = data(Vessel,new_id)[2]
            return (new_id, new_tf+t+dt)
        end
    end
end

#function to convert the position (id,t) to a point in R2.
function evaluation(Vessel, id, t)
    segt, _ = segment_generator(Vessel, id)
    return segt(t)
end

# Uniform samples on the network
function Uniform_sampler(Vessel)
    times = []
    for id =1:length(Vessel.nodes)
        append!(times, data(Vessel,id)[2])
    end
    tot_time = sum(times)
    selector = rand()*tot_time
    for i=1:length(times)
        if selector < times[i]
            return (i, selector)
        else
            selector = selector - times[i]
        end
    end
end
