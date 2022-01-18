# from dotenv import load_dotenv
# load_dotenv()
from dipole_inverse_tools import set_max_num_threads
import numpy as np
from pathlib import Path
set_max_num_threads(6)
import dipole_inverse as dpinv  # noqa: E402

data_path = Path('../chest/tutorial_qdm_data/')

# location and name of QDM and cuboid file
QDMfile = data_path / 'class_QDM_result2.txt'
cuboidfile = data_path / 'class_cuboid_result2.txt'

# size of QDM domain
QDM_domain = np.array([[300, 1250], [450, 1400]]) * 1e-6
# Spacing between two QDM sample points
QDM_spacing = 1.2e-6
# half length of QDM sensor
QDM_deltax = 0.6e-6
# half width of QDM sensor
QDM_deltay = 0.6e-6
# area of QDM sensor
QDM_area = 1.44e-12
# thickness of sample
sample_height = 80e-6
# distance between QDM and top sample
scan_height = 6e-6
# file to write magnetization grains to
Magfile = data_path / "grain_mag.txt"

mag_svd = dpinv.Dipole(
    QDM_domain, QDM_spacing, QDM_deltax,
    QDM_deltay, QDM_area, sample_height,
    scan_height)

mag_svd.obtain_magnetization(QDMfile, cuboidfile, 1e-6, method_populate='cython', verbose=False)
