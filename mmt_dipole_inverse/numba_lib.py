import numba as nb
import numpy as np


@nb.jit(nopython=True)
def populate_matrix_numba(G, scan_domain, scan_height, cuboids, Npart,
                          Ny, Nx, scan_spacing_x, scan_spacing_y,
                          scan_deltax, scan_deltay, Origin, verbose=True):
    """
    Main function to populate the G matrix

    Notes
    -----
    The outer while loop will last until reaching the total number of cuboids
    in the sample. Adjacent cuboids belong to a single particle, which is
    indexed in the 6th element of the cuboids array. The population of the G
    matrix is performed column wise for every particle. For each cuboid
    belonging to a particle, their contribution to the magnetic flux is summed
    up for every sensor measurement in steps of delta in the xy plane, which
    are given by the loops with the i-j indexes. The flux is stored column
    wise.

    Parameters
    ----------
    scan_domain
        Array of size 2x2 with the lower left and upper right coordinates of
        the scan surface
    Origin
        If True the scan data is set to the QDM lower left corner coordinates.
        If False, the scan data origin is set at (0., 0.)
    """

    Cm = 1e-7
    if Origin is True:
        xi0, eta0 = scan_domain[0, :]
    else:
        xi0, eta0 = 0., 0.
    zeta0 = (-1) * scan_height
    sensor_pos = np.zeros(3)
    sensor_pos[2] = zeta0

    # Definitions
    particle_flux = np.zeros(3)
    get_flux = np.zeros(3)
    cuboid_center = np.zeros(3)
    dr_cuboid = np.zeros(3)
    cuboid_size = np.zeros(3)

    i_cuboid = 0
    i_particle_prev = int(cuboids[0, 6])
    i_particle = i_particle_prev

    # print('max cub =', Npart)
    # print('G matrix', G.shape)
    # If grains are not numbered in order this always works
    i_particle_0_N = 0

    while i_cuboid < len(cuboids):
        if verbose:
            # print(f'Particle = {i_particle}  Cuboid = {i_cuboid}')
            print('Particle =', i_particle, 'Cuboid =', i_cuboid)
            # print(particle =)

        i_cuboid_old = i_cuboid

        # Loop over sensor measurements. Each sensor is in the xy
        # plane and has area delta^2
        for j in range(Ny):
            sensor_pos[1] = eta0 + scan_spacing_y * j
            for i in range(Nx):
                sensor_pos[0] = xi0 + scan_spacing_x * i

                # The contribution of the flux for mx, my, mz
                particle_flux[:] = 0

                # Start from the index of the particle being analysed
                i_particle = int(cuboids[i_cuboid_old, 6])
                i_cuboid = i_cuboid_old

                # While the cuboid has particle index of the
                # particle being analysed
                while i_particle == i_particle_prev:
                    #                     print(i_particle, i, j, i_cuboid)
                    cuboid_center[:] = cuboids[i_cuboid, :3]
                    dr_cuboid[:] = cuboid_center - sensor_pos
                    # Cuboid sizes:
                    cuboid_size[:] = cuboids[i_cuboid, 3:6]

                    # calculate flux per cuboid
                    get_flux[:] = 0.
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            for s3 in [-1, 1]:
                                for s4 in [-1, 1]:
                                    for s5 in [-1, 1]:
                                        x = dr_cuboid[0] + s1 * cuboid_size[0] - s4 * scan_deltax
                                        y = dr_cuboid[1] + s2 * cuboid_size[1] - s5 * scan_deltay
                                        z = dr_cuboid[2] + s3 * cuboid_size[2]
                                        sign = s1 * s2 * s3 * s4 * s5
                                        x2, y2, z2 = x ** 2, y ** 2, z ** 2
                                        r2 = x2 + y2 + z2
                                        r = np.sqrt(r2)
                                        Az = np.arctan2(x * y, z * r)
                                        if r != 0.0:
                                            Lx = np.log(x + r)
                                            Ly = np.log(y + r)
                                        else:
                                            Lx, Ly = 0., 0.
                                            print('Error at p = ', i_particle)

                                        F120 = 0.5 * ((y2 - z2) * Lx - r * x) - y * (z * Az - x * Ly)
                                        F210 = 0.5 * ((x2 - z2) * Ly - r * y) - x * (z * Az - y * Lx)
                                        F22m = -x * y * Az - z * (x * Lx + y * Ly - r)

                                        get_flux[0] += sign * F120
                                        get_flux[1] += sign * F210
                                        get_flux[2] += sign * F22m

                    # Finish cuboidsloop in the particle i_particle_prev
                    # and continue with the next sensor measurement

                    # scale flux measurement:
                    particle_flux[:] += -Cm * get_flux
                    i_cuboid += 1
                    i_particle = int(cuboids[i_cuboid, 6])

                # print(i + j * Nx, 3 * i_particle_prev)
                # Populate G matrix column wise
                G[i + j * Nx, 3 * i_particle_0_N] = particle_flux[0]
                G[i + j * Nx, 3 * i_particle_0_N + 1] = particle_flux[1]
                G[i + j * Nx, 3 * i_particle_0_N + 2] = particle_flux[2]

        i_particle_prev = i_particle
        i_particle_0_N += 1
