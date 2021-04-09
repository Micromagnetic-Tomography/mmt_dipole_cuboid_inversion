import numpy as np
import matplotlib.pyplot as plt
import textwrap


def dipole_Bz(dip_r, dip_m, pos_r, Bz_grid):
    """
    Compute the z-component of the dipole field at the pos_r position(s), from
    a group of particles located at the dip_r dipole_positions, and which have
    magnetic dipole moments given in the dip_m array.

    For these arrays, N > 1

    Parameters
    ----------

    dip_r
        N x 3 array with dipole positions (m)
    dip_m
        N x 3 array with dipole moments (Am^2)
    pos_r
        M X P x 3 array (grid) with coordinates of measurement point (m)

    """
    # For every row of dip_r (Nx3 array), subtract pos_r (1x3 array)
    for j in range(pos_r.shape[0]):
        for i in range(pos_r.shape[1]):
            r = pos_r[j, i] - dip_r
            x, y, z = r[:, 0], r[:, 1], r[:, 2]

            rho2 = np.sum(r ** 2, axis=1)
            rho = np.sqrt(rho2)

            mx, my, mz = dip_m[:, 0], dip_m[:, 1], dip_m[:, 2]
            m_dot_r = mx * x + my * y + mz * z
            f = 3e-7 * z * m_dot_r / (rho2 * rho2 * rho)
            g = -1e-7 * mz / (rho2 * rho)

            # Only return Bz
            res = f + g


            Bz_grid[j, i] = np.sum(res)

    return None


Lx, Ly = 40.0, 40.0
res = 21
X, Y = np.linspace(0, Lx, res) * 1e-6, np.linspace(0, Ly, res) * 1e-6
scan_grid_X, scan_grid_Y = np.meshgrid(X, Y)
scan_height = 2e-6
scan_grid_coords = np.stack((scan_grid_X,
                             scan_grid_Y,
                             np.ones_like(scan_grid_X) * scan_height
                             ), axis=2)

Bz_grid = np.zeros((res, res))

# Put a dipole at the middle of the sample
# Save this single dipole at different depths:

# dipole_depth = 30
for dipole_depth in [6, 8, 10, 12, 14, 16, 20, 30, 40, 60]:

    dipole_pos = np.array([[Lx * 0.5, Ly * 0.5, -dipole_depth]]) * 1e-6
    Ms = 4.8e5
    vols = np.array([1 * 1 * 1.]) * 1e-18
    dipole_mus = Ms * vols[:, np.newaxis] * np.array([[0., 1., 0.]])
    # Bzgrid is rewritten with this function
    dipole_Bz(dipole_pos, dipole_mus, scan_grid_coords, Bz_grid)
    np.savetxt(f'single_dipole_depth_{dipole_depth:02d}_Bzgrid.txt',
               Bz_grid, fmt='%.18e')

    # f, ax = plt.subplots()
    # ax.imshow(Bz_grid, cmap='RdBu_r', origin='lower')
    # plt.show()

    # Save a cuboid with volume 1x1x1 micrometre^3 representing a dipole
    cuboid_file = textwrap.dedent(
        f"""
        {Lx * 0.5:.1f} {Ly * 0.5:.1f} {dipole_depth:.1f} 0.5 0.5 0.5 4
        """
        )

    with open(f'single_dipole_depth_{dipole_depth:02d}_cuboids.txt', 'w') as cf:
        cf.write(cuboid_file)

