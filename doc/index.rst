.. dipole_inverse documentation master file, created by
   sphinx-quickstart on Thu Nov  4 16:33:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MMT Dipole Cuboid Inversion - Cuboid model numerical inversion
==============================================================

.. image:: ./_static/area1_grains_scan_sample.jpg

|

This is the documentation for the `dipole_cuboid_inversion` library. This
Python module performs numerical inversions from magnetometry or microscopy
scan data into grains modelled as cuboids. Grain geometries and locations are
obtained from micro or nano X-ray computed tomography. The main class of this
library is the `Dipole` class that accepts the measurement data and has the
necessary methods to obtain inverted magnetizations.

.. autoclass:: mmt_dipole_cuboid_inversion.DipoleCuboidInversion
   :noindex:
   
The `scan_data` and `cuboid_data` parameters are passed as text files. The data
from the scan measurement is formatted as a :math:`NxN` matrix where every
element is the out of plane component of the total field passing through a scan
sensor whose position and size is defined by the `scan_*` parameters. The scan
sensors are assumed as rectangular sensors. The forward field of every sensor
in the forward matrix is modelled as a field flux integrated within the sensor
area, hence it is computed in units of Tesla per meter square. Accordingly, the
total field of every entry in the `scan_data` file is multiplied by the
`scan_area` variable. This means the inverted field is also given in units of
Tesla per meter square.

The data from tomography is given as a 6 column text table referring to the
geometry and location of the cuboids defining the grain profiles. The cuboids
are generated from a voxel-aggregation algorithm applied to the raw tomographic
data, which results in the largest possible cuboids to describe a grain
profile. The `cuboid_data` files requires the following entries::

    x y z dx dy dz index

Columns 1-3 are the cuboid positions, 4-6 are the cuboid edge lengths and the
`index` is an integer number referring to the grain (label) where the cuboid
belongs.

An introductory tutorial is provided in the :doc:`tutorial/tutorial` section
and more complex functionality can be found in the other notebooks

Detailed documentation of every method in the `Dipole` class and functions to
plot the data analyzed with this library can be found in the API section.

.. toctree::
   :maxdepth: 2
   :hidden:

   Introduction <self>
   installation.rst
   tutorial/tutorial.ipynb
   tutorial/synthetic_sample_analysis.ipynb
   tutorial/cuboid_decomposition.ipynb
   autoapi/index.rst


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
