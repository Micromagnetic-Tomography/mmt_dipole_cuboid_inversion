# Test performance of matrix population for three different methods
#
import timeit
from dipole_inverse_tools import set_max_num_threads
import numpy as np
from pathlib import Path
set_max_num_threads(8)
import dipole_inverse as dpinv  # noqa: E402

# Get this script location
thisloc = Path(__file__).resolve().parent
data_path = thisloc / Path('../chest/tutorial_qdm_data/')

# location and name of QDM and cuboid file
QDMfile = data_path / 'class_QDM_result2.txt'
cuboidfile = data_path / 'class_cuboid_result2.txt'

# size of QDM domain
QDM_domain = np.array([[300, 1250], [450, 1400]]) * 1e-6
QDM_spacing = 1.2e-6
QDM_deltax = 0.6e-6
QDM_deltay = 0.6e-6
QDM_area = 1.44e-12
sample_height = 80e-6
scan_height = 6e-6

mag_svd = dpinv.Dipole(
    QDM_domain, QDM_spacing, QDM_deltax,
    QDM_deltay, QDM_area, sample_height,
    scan_height)

mag_svd.read_files(QDMfile, cuboidfile, 1e-6)

results = {}

t_cython_8 = timeit.timeit("mag_svd.prepare_matrix(method='cython', verbose=False)",
                           globals=globals(), number=10)

t_numba = timeit.timeit("mag_svd.prepare_matrix(method='numba', verbose=False)",
                        globals=globals(), number=10)

t_cuda = timeit.timeit("mag_svd.prepare_matrix(method='cuda', verbose=False)",
                       globals=globals(), number=10)

results['cython_8_threads'] = t_cython_8
results['numba'] = t_numba
results['cuda'] = t_cuda

print('\nTiming results for matrix populate function (10 runs)')
print('Matrix: 125 x 125 x 99 x 3  entries. Nx, Ny = 125, Npart=99')
print('-' * 80)
with open('results_comparison_populate_matrix_performance.txt', 'w') as F:
    for k in results.keys():
        outp = f'{k:<18}: {results[k]:.6f} s\n'
        print(outp)
        F.write(outp)
