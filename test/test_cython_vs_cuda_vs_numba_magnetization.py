from dipole_inverse_tools import set_max_num_threads
import numpy as np
from pathlib import Path
import copy
set_max_num_threads(8)
import dipole_inverse as dpinv  # noqa: E402

# Get this script location
thisloc = Path(__file__).resolve().parent
data_path = thisloc / Path('../chest/tutorial_qdm_data/')

# location and name of QDM and cuboid file
QDMfile = data_path / 'class_QDM_result2.txt'
cuboidfile = data_path / 'class_cuboid_result2.txt'
magn_file = data_path / 'grain_mag.txt'

# size of QDM domain
QDM_domain = np.array([[300, 1250], [450, 1400]]) * 1e-6
QDM_spacing = 1.2e-6
QDM_deltax = 0.6e-6
QDM_deltay = 0.6e-6
QDM_area = 1.44e-12
sample_height = 80e-6
scan_height = 6e-6

print('Setting up class')
inverse_model = dpinv.Dipole(
    QDM_domain, QDM_spacing, QDM_deltax,
    QDM_deltay, QDM_area, sample_height,
    scan_height)

inverse_model.read_files(QDMfile, cuboidfile, 1e-6)

# cython
print('Prepare matrix using cython')
inverse_model.prepare_matrix(method='cython', verbose=False)
inverse_model.calculate_inverse()
cython = copy.deepcopy(inverse_model.Mag).reshape(-1, 3)
# cuda
# print('Prepare matrix using cuda')
# inverse_model.prepare_matrix(method='cuda', verbose=False)
# inverse_model.calculate_inverse()
# cuda = copy.deepcopy(inverse_model.Mag).reshape(-1, 3)
# numba
print('Prepare matrix using numba')
inverse_model.prepare_matrix(method='numba', verbose=False)
inverse_model.calculate_inverse()
numba = copy.deepcopy(inverse_model.Mag).reshape(-1, 3)

magn_sol = np.linalg.norm(np.loadtxt(magn_file)[:, 1:], axis=1)
assert all((np.linalg.norm(cython, axis=1) - magn_sol) / magn_sol < 1e-3)
# assert all((np.linalg.norm(cuda, axis=1) - magn_sol) / magn_sol < 1e-3)
assert all((np.linalg.norm(numba, axis=1) - magn_sol) / magn_sol < 1e-3)
print('Assertions passed')