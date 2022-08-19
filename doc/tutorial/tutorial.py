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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Tutorial

# ## Import base libraries

# Before importing the main library we can set the maximum number of threads used by the library and the external libs such as Python and Scipy

from mmt_dipole_inverse_config import set_max_num_threads
set_max_num_threads(6)

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mmt_dipole_inverse as dpinv

# ## Using the Dipole class

# We first specify the location of the scan data and the tomographic data with the cuboid positions and dimensions.

# +
data_path = Path('../../chest/tutorial_qdm_data/')

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
# Magfile = data_path / "grain_mag.txt"
# -

# Now we can instantiate the class using the above parameters

mag_svd = dpinv.Dipole(QDM_domain, QDM_spacing, QDM_deltax, QDM_deltay, 
                       QDM_area, scan_height)

# Alternatively, we can load the scan surface parameters from a `json` file, which can be useful if we want to keep a record of the inversion parameters. Let's take a look at the `json` file for this tutorial first:

# !cat tutorial_scan_params.json

# We can now use this configuration file to load our class:

mag_svd = dpinv.Dipole.from_json(data_path / 'tutorial_scan_params.json')

# Now we can read the cubooid and QDM files:

mag_svd.read_files(QDMfile, cuboidfile, cuboid_scaling_factor=1e-6)

# ## Inversion

# To compute the magnetization we can use the shortcut fuction `obtain_magnetization` which calls three internal methods in the class. To populate the matrix we choose to use the `cython` method which fills the forward matrix `G` in parallel (much faster). The method to perform the numerical inversion can also be specified, in this case we use `pinv2` from Scipy which uses a Singular Value Decomposition for the pseudo-inverse:

# + tags=[]
mag_svd.obtain_magnetization(QDMfile, cuboidfile, cuboid_scaling_factor=1e-6,
                             method_populate='cython', method_inverse='scipy_pinv2')

# +
# An alternative method is to populate the matrix using the Numba optimised
# function
# mag_svd.obtain_magnetization(method_populate='numba', method_inverse='scipy_pinv2')
# -

# The forward matrix has the following form:

mag_svd.Forward_G

# We can directly plot the original scan data but in the next section we will use the more powerful plot tools from this library

plt.imshow(mag_svd.QDM_matrix, origin='lower', cmap='magma')
plt.show()

# This is the field but inverted, we will plot this in the next section using the correct limits

plt.imshow((mag_svd.Forward_G @ mag_svd.Mag).reshape(mag_svd.Ny, -1),
           origin='lower', cmap='magma')
plt.show()

# ## Plots

# + [markdown] tags=[]
# The `mmt_dipole_inverse` library includes a powerful and modular way of generating plots from the inversion results and the cuboid/grain information.
# -

from mmt_dipole_inverse.tools import plot as dpi_pt

# We first generate the corresponding arrays in the instance of the Dipole class, in this case we called it `mag_svd`. The `set_grain geometries` will create these arrays with information about the grain position and their dimensions. An useful argument is to scale the space by a factor, in this case we will set everything in micrometre units:

dpi_pt.set_grain_geometries(mag_svd, spatial_scaling=1e6)

# We can, for example, plot the grain boundaries:

f, ax = plt.subplots()
dpi_pt.plot_grain_boundaries(mag_svd, ax)
plt.show()

# But more useful is to combine this information with other plots. For example, we can first pot the grain boundaries and on top of it, we can plot the grains colored by their magnetization, which was computed with the Dipole class methods.
# Moreover, we plot the inverted forward field in the background. We can do these operations pasing the same Matplotlib `axis`:

f, ax = plt.subplots(figsize=(8, 8))
dpi_pt.plot_magnetization_on_grains(mag_svd, ax, grain_labels=False)
dpi_pt.plot_grain_boundaries(mag_svd, ax, 
                             labels_args=dict(ha='center', va='center', fontsize=14))
dpi_pt.plot_inversion_field(mag_svd, ax)
plt.show()
