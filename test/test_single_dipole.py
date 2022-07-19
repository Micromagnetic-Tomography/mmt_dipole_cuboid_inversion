from mmt_dipole_inverse_tools import set_max_num_threads
set_max_num_threads(4)
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import mmt_dipole_inverse as dpinv  # noqa: E402
try:
    # the cuda populate_matrix function
    from mmt_dipole_inverse.cython_cuda_lib import pop_matrix_cudalib
    HASCUDA = True
except ImportError:
    HASCUDA = False

# Get this script location
thisloc = Path(__file__).resolve().parent


def test_dipole_class():

    QDMfile = thisloc / Path('./single_dipole_depth_30_Bzgrid.txt')
    cuboidfile = thisloc / Path('./single_dipole_depth_30_cuboids.txt')

    # size of QDM domain
    QDM_domain = np.array([[0, 0], [40., 40.]]) * 1e-6
    # Spacing between two QDM sample points
    QDM_spacing = 2e-6
    # half length of QDM sensor
    QDM_deltax = 1e-6
    # half width of QDM sensor
    QDM_deltay = 1e-6
    # area of QDM sensor -> necessary? --> use deltax * deltay
    QDM_area = 4e-12
    # thickness of sample -> Unnecessary
    sample_height = 30e-6
    # distance between QDM and top sample
    scan_height = 2e-6

    dip_inversion = dpinv.Dipole(
        QDM_domain, QDM_spacing,
        QDM_deltax, QDM_deltay, QDM_area, sample_height, scan_height)

    dip_inversion.read_files(QDMfile, cuboidfile, cuboid_scaling_factor=1e-6)

    assert(dip_inversion.QDM_matrix.shape[0]) == 21
    assert(dip_inversion.QDM_matrix.shape[1]) == 21

    print('Testing Numba pop matrix')
    dip_inversion.prepare_matrix(method='numba', verbose=True)
    FG_copy = np.copy(dip_inversion.Forward_G)

    print('\nComparing Cython pop matrix to Numba code')
    dip_inversion.prepare_matrix(method='cython', verbose=False)
    for j, i in [(5, 1), (100, 0), (195, 2), (368, 1)]:
        assert abs(FG_copy[j, i] - dip_inversion.Forward_G[j, i]) < 1e-8
    print(dip_inversion.Forward_G[j, i])

    dip_inversion.calculate_inverse(method='scipy_pinv2', rcond=1e-25)

    Ms = 4.8e5
    # Check relative error is less than 1% = 0.01
    assert abs(dip_inversion.Mag[1] - Ms) / Ms < 1e-2

    if HASCUDA:
        print('\nComparing NVIDIA cuda pop matrix to Numba code')
        dip_inversion.prepare_matrix(method='cuda', verbose=False)
        for j, i in [(5, 1), (100, 0), (195, 2), (368, 1)]:
            assert abs(FG_copy[j, i] - dip_inversion.Forward_G[j, i]) < 1e-8
        print(dip_inversion.Forward_G[j, i])

        dip_inversion.calculate_inverse(method='scipy_pinv2', rcond=1e-25)

        Ms = 4.8e5
        # Check relative error is less than 1% = 0.01
        assert abs(dip_inversion.Mag[1] - Ms) / Ms < 1e-2


if __name__ == "__main__":
    test_dipole_class()
