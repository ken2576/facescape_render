import numpy as np

def gen_ring_poses(radius, angle, height, h_cams=2, w_cams=10):
    """Generate camera poses in a cylinder

    Args:
        radius: radius of the cylinder
        angle: arc angle of the cylinder (in degree)
        h_cams: how many cameras along the vertical axis
        w_cams: how many cameras along the arc
    Returns:
        A list of positions [#cameras, 3]
    """

    rad = np.deg2rad(angle/2)
    rads = np.linspace(-rad, rad, w_cams)
    z = np.linspace(-height/2, height/2, h_cams)[..., None]
    z = np.tile(z, [1, w_cams])
    z = z.reshape([1, -1])
    pos = np.array([radius * np.sin(rads), -radius * np.cos(rads)])
    pos = np.tile(pos, [1, h_cams])
    pos = np.concatenate([pos, z], 0)
    return pos.transpose()

def gen_random_frontal_pose(rmin, rmax, n_cams):
    """Generate random frontal poses

    Args:
        rmin: minimum radius away from the origin
        rmax: maximum radius away from the origin
        n_cams: number of cameras
    Returns:
        A list of positions [#cameras, 3]
    """
    radii = np.random.uniform(rmin, rmax, n_cams)
    pos = np.stack([np.zeros(n_cams),
                    -radii,
                    np.zeros(n_cams)], 1)
    return (pos, radii)

def gen_random_poses(rmin, rmax, theta_max, phi_max, n_cams):
    """Generate camera poses randomly on a sphere

    Args:
        rmin: minimum radius away from the origin
        rmax: maximum radius away from the origin
        theta_max: maximum azimuthal angle to the left and the right
        phi_max: maximum elevational angle
        n_cams: number of cameras
    Returns:
        A list of positions [#cameras, 3]
    """
    radii = np.random.uniform(rmin, rmax, n_cams)
    theta = np.random.uniform(-theta_max, theta_max, n_cams)
    phi = np.random.uniform(-phi_max, phi_max, n_cams)
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    pos = np.stack([radii * np.sin(theta) * np.cos(phi),
                    -radii * np.cos(theta)* np.cos(phi),
                    radii * np.sin(phi)], 1)
    return (pos, radii)

def gen_multiview_poses(rmin, rmax, theta_max, phi_max, n_cams):
    """Generate camera poses randomly on a sphere for multiview cameras
       (Multiview cameras are around the same radius, 5% difference)
    Args:
        rmin: minimum radius away from the origin
        rmax: maximum radius away from the origin
        theta_max: maximum azimuthal angle to the left and the right
        phi_max: maximum elevational angle
        n_cams: number of cameras
    Returns:
        A list of positions [#cameras, 3]
    """
    dist = np.random.uniform(rmin, rmax)
    radii = np.random.uniform(dist*0.95, dist*1.05, n_cams)
    theta = np.random.uniform(-theta_max, theta_max, n_cams-1)
    phi = np.random.uniform(-phi_max, phi_max, n_cams-1)
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    theta = np.concatenate([np.array([0.0]), theta])
    phi = np.concatenate([np.array([0.0]), phi])

    pos = np.stack([radii * np.sin(theta) * np.cos(phi),
                    -radii * np.cos(theta)* np.cos(phi),
                    radii * np.sin(phi)], 1)
    return (pos, radii)

if __name__ == '__main__':
    # poses = gen_ring_poses(300, 20, 50, 5, 5)
    # poses = gen_random_poses(1, 3, 45, 30, 100)
    poses, _ = gen_random_frontal_pose(300, 500, 5)
    print(poses.shape)
    print(poses)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')      
    for pose in poses:
        ax.scatter(pose[0], pose[1], pose[2], color='black')

    s = 600
    ax.set_xlim3d(-s,s)
    ax.set_ylim3d(-s,s)
    ax.set_zlim3d(-s,s)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()