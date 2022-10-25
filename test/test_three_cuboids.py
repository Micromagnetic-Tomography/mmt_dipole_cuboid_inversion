from mmt_dipole_cuboid_inversion_config import set_max_num_threads
set_max_num_threads(4)
import numpy as np
from pathlib import Path
# import time
# import matplotlib.pyplot as plt
import mmt_dipole_cuboid_inversion as dci  # noqa: E402
# import mmt_dipole_inverse.tools as dcit  # noqa: E402
try:
    # the cuda populate_matrix function
    from mmt_dipole_inverse.cython_cuda_lib import pop_matrix_cudalib
    HASCUDA = True
except ImportError:
    HASCUDA = False

# Get this script location
thisloc = Path(__file__).resolve().parent
dataloc = thisloc / 'test_three_cub_data/'


def test_three_cuboids():
    """
    Test the inversion of 3 cuboidal particles by comparing the inverted
    magnetizations with the magnetization from an analytical description of the
    particles and their demag field, which is saved in `npy` format. The field
    and particle theoretical values are located in the `test_three_cub_data/`
    directory. The scan surface properties are specified here.

    Here we use a RHS, so z is pointing towards depth (scanning surf negative pos)
    """

    ScanMatrix = dataloc / 'bz_cuboids_three-particles.npy'
    cuboidfile = dataloc / 'cuboids.txt'
    cuboid_data = np.loadtxt(cuboidfile, skiprows=0)
    # cuboid_data[:, 2] *= -1

    # Scan surface properties (from the semi-analytical model)
    scan_domain = np.array([[-2.5, -2.5], [2.5, 2.5]]) * 1e-6
    scan_spacing = 50e-9
    scan_deltax = 25e-9
    scan_deltay = 25e-9
    scan_area = 2500e-18
    scan_height = -500e-9

    dip_inversion = dci.DipoleCuboidInversion(scan_domain, scan_spacing, scan_deltax,
                                 scan_deltay, scan_area, scan_height)

    # Cuboid file units in nm
    dip_inversion.read_files(ScanMatrix, cuboid_data,
                             cuboid_scaling_factor=1e-9)
    # print(dip_inversion.cuboids)

    dip_inversion.prepare_matrix(method='numba', verbose=True)
    # FG_copy = np.copy(dip_inversion.Forward_G)

    # dip_inversion.calculate_inverse(method='scipy_pinv', atol=1e-30)
    dip_inversion.calculate_inverse(method='scipy_pinv', rtol=1e-30)
    print(dip_inversion.Mag)

    cube_props_theory = np.loadtxt(thisloc / './test_three_cub_data/cuboids_properties.dat')
    # Mas values at the final column
    MagTheory = cube_props_theory[:, -1]
    MagInversion = np.linalg.norm(dip_inversion.Mag.reshape(-1, 3), axis=1)

    # Check relative error is less than 1% = 0.01
    print('-' * 10)
    for i in range(3):
        print(f'Cuboid {i + 1}')
        RelErr = np.abs(MagInversion[i] - MagTheory[i]) / np.abs(MagTheory[i])
        print(f'Minv = {MagInversion[i]:.4f}  Mtheory = {MagTheory[i]:.4f}')
        print(f'RelErr |Minv - Mtheory| = {RelErr * 100:.5f} %')
        print('-' * 10)
        assert(RelErr < 0.01)

    # assert abs(dip_inversion.Mag[1] - Ms) / Ms < 1e-2
    dip_inversion.Inverse_G

    # dcit.plot.set_grain_geometries(dip_inversion)
    #
    # f, ax = plt.subplots()
    # dcit.plot.plot_grain_boundaries(dip_inversion, ax)
    # im = dcit.plot.plot_scan_field(dip_inversion, ax,
    #                                  imshow_args=dict(cmap='RdBu'))
    # ax.set_title('Forward field')
    # plt.colorbar(im)
    # plt.show()

    # f, ax = plt.subplots()
    # dcit.plot.plot_grain_boundaries(dip_inversion, ax)
    # # im = dcit.plot.plot_scan_field(dip_inversion, ax,
    # #                                  imshow_args=dict(cmap='RdBu'))
    # im = dcit.plot.plot_inversion_field(dip_inversion, ax,
    #                                       imshow_args=dict(cmap='RdBu'))
    # # im = dcit.plot.plot_residual(dip_inversion, ax,
    # #                                imshow_args=dict(cmap='RdBu'))

    # plt.colorbar(im)
    # plt.show()


if __name__ == "__main__":
    test_three_cuboids()
