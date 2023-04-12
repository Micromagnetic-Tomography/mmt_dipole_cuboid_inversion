Installation
============

The `mmt_dipole_cuboid_inversion` library can be installed directly via PyPI
using `pip` (notice PyPI changed `_` to `-`). Only Linux and Windows builds are
currently available.

.. code-block:: sh

    pip install mmt-dipole-cuboid-inversion

Alternatively, you can clone the Github `repository`_ and install via
`pip`

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion.git
    cd mmt_dipole_cuboid_inversion
    pip install .

Or using Poetry (recommended for development):

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion.git
    cd mmt_dipole_cuboid_inversion
    poetry install

CUDA
----

To install the CUDA library you can clone the repository and define the
`CUDAHOME` or `CUDA_PATH` environment variables pointing to your CUDA
directory. For example

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion.git
    cd mmt_dipole_cuboid_inversion
    export CUDA_PATH=/usr/local/cuda-11.7/
    poetry install

and you should see some compilation outputs from Cython. The build will also
work if the directory of the `nvcc` compiler is defined in your `PATH`
variable. 

Using Poetry will install the `mmt_dipole_cuboid_inversion` in a new Python
environment. If you need it in your base environment, you can use `poetry
build` and then `pip install` the wheel (`.whl`) file that is generated in the
`dist` directory.

.. _repository: https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion
