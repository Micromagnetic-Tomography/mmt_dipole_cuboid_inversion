# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Synthetic Sample

import numpy as np
import matplotlib.pyplot as plt
import mmt_dipole_inverse as dpinv
import mmt_dipole_inverse.tools as dpinv_tools

import requests, zipfile, io
from pathlib import Path

data_dir = Path('deGroot2018_data')
data_dir.mkdir(exist_ok=True)

if not any(data_dir.iterdir()):
    data_url = 'https://store.pangaea.de/Publications/deGroot-etal_2018/Micro-grain-data.zip'
    r = requests.get(data_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)

# Now open the ZIP file with formatted data:
z = zipfile.ZipFile(data_dir / 'V2_2021-04.zip')
z.extractall(data_dir)

# ## Area 1: Unknown Magnetic State

# +
data_dir = Path('deGroot2018_data/PDI-16803')

# location and name of QDM and cuboid file
ScanFile = data_dir / 'Area1-90-fig2MMT.txt'
CuboidFile = data_dir / 'FWInput-FineCuboids-A1.txt'

# 
SQUID_domain = np.array([[0, 0], [350, 200]]) * 1e-6
SQUID_spacing = 1e-6
SQUID_deltax = 0.5e-6
SQUID_deltay = 0.5e-6
SQUID_area = 1e-12
scan_height = 2e-6
# -

mag_inv = dpinv.Dipole(SQUID_domain, SQUID_spacing,
                       SQUID_deltax, SQUID_deltay, SQUID_area, scan_height)

mag_inv.read_files(ScanFile, CuboidFile, cuboid_scaling_factor=1e-6)

mag_inv.prepare_matrix(method='cython')

mag_inv.calculate_inverse(method='scipy_pinv2', atol=1e-25)

mag_inv.Mag.reshape(-1, 3)

MagNorms = np.linalg.norm(mag_inv.Mag.reshape(-1, 3), axis=1)
for m, mag in enumerate(MagNorms):
    print(f'Grain {m + 1}  |M| = {mag:>9.3f} A/m')

dpinv.tools.


