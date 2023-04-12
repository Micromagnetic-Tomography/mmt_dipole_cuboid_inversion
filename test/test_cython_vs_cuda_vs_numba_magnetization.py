from mmt_dipole_cuboid_inversion_config import set_max_num_threads
import numpy as np
from pathlib import Path
import copy
import pytest
set_max_num_threads(8)
import mmt_dipole_cuboid_inversion as dci  # noqa: E402
try:
    from .cython_cuda_lib import populate_matrix as pop_matrix_CUDA  # the cuda populate_matrix function
    HASCUDA = True
except ImportError:
    HASCUDA = False


# Get this script location
thisloc = Path(__file__).resolve().parent
data_path = thisloc / Path('../chest/tutorial_qdm_data/')

BACKENDS = ['numba', 
            'cython',
             pytest.param('cuda', marks=pytest.mark.skipif(not HASCUDA, reason="CUDA not found"))
            ]

@pytest.mark.parametrize("backend", BACKENDS, ids=['num', 'cyt', 'cud'])
def test_magnetization_result(backend):
    # location and name of QDM and cuboid file
    QDMfile = data_path / 'class_QDM_result2.txt'
    cuboidfile = data_path / 'class_cuboid_result2.txt'
    magn_file = data_path / 'grain_mag.txt'

    # size of QDM domain
    QDM_sensor_domain = np.array([[300, 1250], [450, 1400]]) * 1e-6
    QDM_spacing = 1.2e-6
    QDM_deltax = 0.6e-6
    QDM_deltay = 0.6e-6
    QDM_area = 1.44e-12
    sample_height = 80e-6
    scan_height = 6e-6

    print('Setting up class')
    inverse_model = dci.DipoleCuboidInversion(
        None, QDM_sensor_domain, QDM_spacing, QDM_deltax, QDM_deltay, QDM_area,
        scan_height, verbose=True)

    inverse_model.read_files(QDMfile, cuboidfile, 1e-6)
    inverse_model.set_scan_domain()
    inverse_model.verbose = False

    # Inversion using specific backend:
    print(f'Prepare matrix using {backend}')
    inverse_model.prepare_matrix(method=backend)
    inverse_model.calculate_inverse()
    mag_result = copy.deepcopy(inverse_model.Mag).reshape(-1, 3)

    magn_sol = np.linalg.norm(np.loadtxt(magn_file)[:, 1:], axis=1)
    assert all((np.linalg.norm(mag_result, axis=1) - magn_sol) / magn_sol < 1e-3)
    print('Mag assertions passed')


if __name__ == '__main__':
    test_magnetization_result('numba')
    test_magnetization_result('cython')
    if HASCUDA:
        test_magnetization_result('cuda')
