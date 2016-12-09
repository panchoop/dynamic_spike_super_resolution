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
