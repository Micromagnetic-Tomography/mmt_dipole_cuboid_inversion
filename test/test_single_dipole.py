from mmt_dipole_cuboid_inversion_config import set_max_num_threads
set_max_num_threads(4)
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import mmt_dipole_cuboid_inversion as dci  # noqa: E402
try:
    # the cuda populate_matrix function
    from mmt_dipole_cuboid_inversion.cython_cuda_lib import pop_matrix_cudalib
    HASCUDA = True
except ImportError:
    HASCUDA = False

# Get this script location
thisloc = Path(__file__).resolve().parent


def test_dipole_class():

    ScanFile = thisloc / Path('./single_dipole_depth_30_Bzgrid.txt')
    cuboidfile = thisloc / Path('./single_dipole_depth_30_cuboids.txt')

    # size of QDM domain
    scan_sensor_domain = np.array([[0, 0], [40., 40.]]) * 1e-6
    # Spacing between two QDM sample points
    scan_spacing = 2e-6
    # half length of QDM sensor
    scan_deltax = 1e-6
    # half width of QDM sensor
    scan_deltay = 1e-6
    # area of QDM sensor -> necessary? --> use deltax * deltay
    scan_area = 4e-12
    # thickness of sample -> Unnecessary
    # sample_height = 30e-6
    # distance between QDM and top sample
    scan_height = 2e-6

    dip_inversion = dci.DipoleCuboidInversion(
        None, scan_sensor_domain, scan_spacing, scan_deltax, scan_deltay, scan_area,
        scan_height, verbose=True)

    dip_inversion.read_files(ScanFile, cuboidfile, cuboid_scaling_factor=1e-6)
    dip_inversion.set_scan_domain(gen_sd_mesh_from='sensor_center_domain')

    # Since scan height is positive, z-pos of cuboids must be negative:
    dip_inversion.cuboids[:, 2] *= -1.0

    assert(dip_inversion.scan_matrix.shape[0]) == 21
    assert(dip_inversion.scan_matrix.shape[1]) == 21

    print('Testing Numba pop matrix')
    dip_inversion.prepare_matrix(method='numba')
    FG_copy = np.copy(dip_inversion.Forward_G)

    print('\nComparing Cython pop matrix to Numba code')
    dip_inversion.prepare_matrix(method='cython')
    for j, i in [(5, 1), (100, 0), (195, 2), (368, 1)]:
        assert abs(FG_copy[j, i] - dip_inversion.Forward_G[j, i]) < 1e-8
    # print(dip_inversion.Forward_G[j, i])

    dip_inversion.calculate_inverse(method='scipy_pinv2', atol=1e-25)

    Ms = 4.8e5
    # Check relative error is less than 1% = 0.01
    assert (np.linalg.norm(dip_inversion.Mag) - Ms) / Ms < 1e-2

    if HASCUDA:
        print('\nComparing NVIDIA cuda pop matrix to Numba code')
        dip_inversion.prepare_matrix(method='cuda')
        for j, i in [(5, 1), (100, 0), (195, 2), (368, 1)]:
            assert abs(FG_copy[j, i] - dip_inversion.Forward_G[j, i]) < 1e-8
        print(dip_inversion.Forward_G[j, i])

        dip_inversion.calculate_inverse(method='scipy_pinv2', rcond=1e-25)

        Ms = 4.8e5
        # Check relative error is less than 1% = 0.01
        assert abs(dip_inversion.Mag[1] - Ms) / Ms < 1e-2


def test_coord_system_single_dipole():

    ScanFile = thisloc / Path('./single_dipole_depth_30_Bzgrid.txt')
    cuboidfile = thisloc / Path('./single_dipole_depth_30_cuboids.txt')
    cuboid_data = np.loadtxt(cuboidfile, skiprows=0, ndmin=2)
    cuboid_data[:, 2] *= -1.
    # print(cuboid_data)

    sensor_domain = np.array([[0, 0], [40., 40.]]) * 1e-6
    scan_spacing = 2e-6
    scan_deltax = 1e-6
    scan_deltay = 1e-6
    scan_area = 4e-12
    # distance between QDM and top sample
    scan_height = 2e-6

    dip_inversion = dci.DipoleCuboidInversion(
        None, sensor_domain, scan_spacing, scan_deltax, scan_deltay, scan_area,
        scan_height, verbose=True)

    dip_inversion.read_files(ScanFile, cuboid_data, cuboid_scaling_factor=1e-6)
    dip_inversion.set_scan_domain(gen_sd_mesh_from='sensor_center_domain')

    print('Testing Numba pop matrix')
    dip_inversion.prepare_matrix(method='cython')
    dip_inversion.calculate_inverse(method='scipy_pinv', rtol=1e-25)
    # Save the mag from the default right handed system of coords:
    Mag_RHS = np.copy(dip_inversion.Mag)

    Ms = 4.8e5
    # Check relative error is less than 1% = 0.01
    rel_err = (np.linalg.norm(dip_inversion.Mag) - Ms) / Ms
    assert rel_err < 1e-2

    # Check abs and rel error by components:
    msol = np.array([0., 1., 1.]) / np.sqrt(2.)
    for i in range(1, 3):
        minv = dip_inversion.Mag / np.linalg.norm(dip_inversion.Mag)
        ae = np.abs(minv[i] - msol[i])
        re = np.abs(minv[i] - msol[i]) / np.abs(msol[i])
        # print(re)
        assert ae < 1e-3
        assert re < 1e-3

    # Now do inversion using LHS with positive z downwards (towards grain depth)
    dip_inversion.cuboids[:, 2] *= -1.
    dip_inversion.scan_height = -scan_height
    dip_inversion.prepare_matrix(method='cython')
    dip_inversion.calculate_inverse(method='scipy_pinv', rtol=1e-25)
    Mag_LHS = np.copy(dip_inversion.Mag)

    Ms = 4.8e5
    # Check relative error is less than 1% = 0.01
    rel_err = (np.linalg.norm(dip_inversion.Mag) - Ms) / Ms
    assert rel_err < 1e-2

    # Check by components: (LHS will have the opposite direction in z)
    msol = np.array([0., 1., -1.]) / np.sqrt(2.)
    for i in range(3):
        minv = dip_inversion.Mag / np.linalg.norm(dip_inversion.Mag)
        ae = np.abs(minv[i] - msol[i])
        assert ae < 1e-3

    print("RHS:")
    print(Mag_RHS)
    print(Mag_RHS / np.linalg.norm(Mag_RHS))
    print("LHS:")
    print(Mag_LHS)
    print(Mag_LHS / np.linalg.norm(Mag_LHS))


if __name__ == "__main__":
    test_dipole_class()
    test_coord_system_single_dipole()
