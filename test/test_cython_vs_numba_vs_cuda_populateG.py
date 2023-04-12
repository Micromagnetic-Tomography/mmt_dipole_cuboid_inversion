# Test performance of matrix population for three different methods
#
import timeit
from mmt_dipole_cuboid_inversion_config import set_max_num_threads
import numpy as np
from pathlib import Path
import pytest
set_max_num_threads(8)  # Use  8 threads for the Cython calculation
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

@pytest.mark.skip(reason="Long time test")
@pytest.mark.parametrize("backend", BACKENDS, ids=['num', 'cyt', 'cud'])
def test_populate_G(backend):
    # location and name of QDM and cuboid file
    QDMfile = data_path / 'class_QDM_result2.txt'
    cuboidfile = data_path / 'class_cuboid_result2.txt'

    # size of QDM domain
    QDM_sensor_domain = np.array([[300, 1250], [450, 1400]]) * 1e-6
    QDM_spacing = 1.2e-6
    QDM_deltax = 0.6e-6
    QDM_deltay = 0.6e-6
    QDM_area = 1.44e-12
    sample_height = 80e-6
    scan_height = 6e-6

    mag_svd = dci.DipoleCuboidInversion(
        None, QDM_sensor_domain, QDM_spacing, QDM_deltax, QDM_deltay, QDM_area, scan_height,
        verbose=False)

    mag_svd.read_files(QDMfile, cuboidfile, 1e-6)
    mag_svd.set_scan_domain(gen_sd_mesh_from='sensor_center_domain')

    t_method = timeit.timeit(lambda: mag_svd.prepare_matrix(method=backend),
                             number=10)

    # For now just test all values are not NaNs or infinites
    assert np.all(np.isfinite(mag_svd.Forward_G))

    return t_method


if __name__ == '__main__':
    with open('results_comparison_populate_matrix_performance.txt', 'w') as F:
        msg = ('Timing results for matrix populate function (10 runs)\n' +
               'Matrix: 125 x 125 x 99 x 3  entries. Nx, Ny = 125, Npart=99\n' +
               '-' * 80 + '\n')
        print('\n' + msg)
        F.write(msg)
        for k in ['cython', 'numba', 'cuda']:
            print(k)
            if k == 'cuda' and not HASCUDA:
                continue
            res = test_populate_G(k)
            print(res)
            outp = f'{k:<18}: {res:.6f} s\n'
            print(outp)
            F.write(outp)
    print('Results written to: results_comparison_populate_matrix_performance.txt')
