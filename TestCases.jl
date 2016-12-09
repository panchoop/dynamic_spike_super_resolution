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
function two_points_2d_ortho(x_max, dx, dv)
    pts_x = [0.5*x_max - 0.5*dx, 0.5*x_max + 0.5 * dx]
    pts_y = [0.5*x_max, 0.5*x_max]
    weights = [1.0, 1.0]
    velocities_y = [-dv/2 ; dv/2]
    velocities_x = [0.0 ; 0.0]
    thetas = [pts_x'; pts_y'; velocities_x'; velocities_y']
    return (thetas, weights)
end
function two_points_2d_ortho_same(x_max,dx,dv)
    pts_x = [0.5*x_max - 0.5*dx, 0.5*x_max + 0.5 * dx]
    pts_y = [0.5*x_max, 0.5*x_max]
    weights = [1.0, 1.0]
    velocities_y = [dv/2 ; dv]
    velocities_x = [0.0 ; 0.0]
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
function two_groups_1d(x_max, dx, dv)
    pts_1 = x_max * random_point_cloud(dx/x_max)
    pts_2 = x_max * random_point_cloud(dx/x_max)
    pts = [pts_1; pts_2]
    velocities = [-dv/2*ones(size(pts_1)); dv/2*ones(size(pts_2))]
    weights = ones(size(pts))
    thetas = [pts'; velocities']
    return (thetas, weights)
end
