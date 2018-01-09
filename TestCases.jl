module TestCases
export three_points_1d, two_points_1d, three_points_2d, two_points_2d, random_point_cloud, cloud_1d
function three_points_1d(x_max, dx, dv)
    pts = [0.5*x_max-0.5*dx, 0.5*x_max, 0.5*x_max + 0.5*dx]
    weights = [1.0; 1.0; 1.0]
    velocities = [0.0 ; dv; 0.0]
    thetas = [pts'; velocities']
    return (thetas, weights)
end
function two_points_1d(x_max, dx, dv)
    pts = [0.5*x_max - 0.5*dx, 0.5*x_max + 0.5 * dx]
    weights = [0.5, 0.5]
    velocities = [-dv/2, dv/2]
    thetas = [pts'; velocities']
    return (thetas, weights)
end
function three_points_2d(x_max, dx, dv, direction="x")
    eps = dx / 10
    pts_x = [0.5*x_max-0.5*dx, 0.5*x_max, 0.5*x_max + 0.5*dx]
    pts_y = [0.5*x_max, 0.5*x_max, 0.5*x_max]
    weights = [1.0; 1.0; 1.0]
    velocities_x = [0.0 ; dv; 0.0]
    velocities_y = [0.0 ; 0.0; 0.0]
    if direction == "x"
        thetas = [pts_x'; pts_y'; velocities_x'; velocities_y']
    else
        thetas = [pts_y'; pts_x'; velocities_y'; velocities_x']
    end
    return (thetas, weights)
end
function two_points_2d(x_max, dx, dv)
    pts_x = [0.5*x_max - 0.5*dx, 0.5*x_max + 0.5 * dx]
    pts_y = [0.5*x_max, 0.5*x_max]
    weights = [1.0, 1.0]
    velocities_x = [-dv/2 ; dv/2]
    velocities_y = [0.0 ; 0.0]
    thetas = [pts_x'; pts_y'; velocities_x'; velocities_y']
    return (thetas, weights)
end
function random_point_cloud(dx)
    pts = dx + (1.0-3*dx)*rand()
    pts = [0.0; pts; pts+dx; 1.0]
    is_valid = (pt, pts) -> pt > dx &&
    pt < 1-dx &&
    minimum(abs(pts[2:end-1] - pt)) > dx
    while maximum(mod(pts - circshift(pts, 1), 1.0)) > 2*dx
        new_pt = rand()
        while !is_valid(new_pt, pts)
            new_pt = rand()
        end
        pts = sort([pts; new_pt])
    end
    return pts[2:end-1]
end
function random_position_and_velocity(dx, dv)
    pts = dx + (1.0 - 2*dx) * rand()
    velocities = (1.0 - dv) * rand()
    velocities = velocities
    is_valid = (pt, vel, pts, vels) -> pt > dx &&
    pt < 1-dx &&
    (minimum(abs(pts - pt)) > dx ||
     minimum(abs(vels - vel)) > dv)
    while((maximum([abs(v1-v2) for v1 in velocities, v2 in velocities]) > 2*dv &&
           maximum([abs(x1-x2) for x1 in pts, x2 in pts]) > 2*dx) || length(velocities) == 1)
        new_pt = rand()
        new_vel = rand()
        iter = 0
        notFound = false;
        while !is_valid(new_pt, new_vel, pts, velocities)
            new_pt = rand()
            new_vel = rand()
            iter = iter + 1
            if (iter > 100)
                notFound = true;
                break;
            end
        end
        if (notFound)
            break;
        end
        pts = [pts; new_pt]
        velocities = [velocities; new_vel]
        if (length(pts) > 10)
            break;
        end
    end
    return (pts, velocities)
end
function cloud_1d(x_max, v_max, dx, dv)
    (pts, velocities) = random_position_and_velocity(dx/x_max, dv/(2*v_max))
    weights = ones(size(pts))
    thetas = [x_max * pts'; v_max - 2*v_max*velocities']
    return (thetas, weights)
end
function cloud_1d(x_max, v_max, n)
    pts = x_max * rand(n)
    velocities =  v_max - 2*v_max*rand(n)
    thetas = [pts'; velocities']
    weights = ones(size(pts))
    return (thetas, weights)
end
function cloud_1d_full(x_max,v_max,K,tau,n)
# We generate random particles, uniform on the rectangle [0,x_max]x[-v_max,v_max]
# under the condition of belonging to the set Omega.
    pts = rand(0)
    velocities = rand(0)
    while size(pts)[1]<n
	newPts = rand()*x_max
	newVel = (rand()-0.5)*2*v_max
	if newPts + K*tau*abs(newVel) <= x_max && newPts - K*tau*abs(newVel) >= 0
	   push!(pts,newPts)
	   push!(velocities,newVel)
	end
    end
    thetas = [pts'; velocities']
    weights = ones(size(pts))
    return (thetas, weights)
end
function cloud_1d_full(x_max,v_max,w_min,w_max,K,tau,n)
# Same as the before one, but we are considering random weights.
    pts = rand(0)
    velocities = rand(0)
    while size(pts)[1]<n
	newPts = rand()*x_max
	newVel = (rand()-0.5)*2*v_max
	if newPts + K*tau*abs(newVel) <= x_max && newPts - K*tau*abs(newVel) >= 0
	   push!(pts,newPts)
	   push!(velocities,newVel)
	end
    end
    thetas = [pts'; velocities']
    weights = w_min + rand(size(pts))*(w_max-w_min)
    return (thetas, weights)
end
function two_groups_1d(x_max, dx, dv)
    pts_1 = x_max * random_point_cloud(dx/x_max)
    pts_2 = x_max * random_point_cloud(dx/x_max)
    pts = [pts_1; pts_2]
    velocities = [-dv/2*ones(size(pts_1)); dv/2*ones(size(pts_2))]
    weights = ones(size(pts))
    thetas = [pts'; velocities']
    return (thetas, weights)
end
function two_groups_2d(x_max, dx, dy, dv)
    pts_1 = x_max * random_point_cloud(dx/x_max)
    pts_2 = x_max * random_point_cloud(dx/x_max)
    pts_x = [pts_1; pts_2]
    pts_y = [0.5*x_max * ones(size(pts_1)) + dy/2; 0.5*x_max * ones(size(pts_2)) - dy/2]
    velocities_y = [zeros(size(pts_1)); zeros(size(pts_2))]
    velocities_x = [-dv/2*ones(size(pts_1)); dv/2*ones(size(pts_2))]
    weights = ones(size(pts_x))
    thetas = [pts_x'; pts_y'; velocities_x'; velocities_y']
    return (thetas, weights)
end
function aligned_1d(x_max, dx, n)
    pts = dx * (1:n)
    velocities = zeros(size(pts))
    thetas = [pts'; velocities']
    weights = ones(size(pts))
    return (thetas, weights)
end
end
