[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7823450.svg)](https://doi.org/10.5281/zenodo.7823450)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mmt-dipole-cuboid-inversion/badge/?version=latest)](https://mmt-dipole-cuboid-inversion.readthedocs.io/en/latest/?badge=latest)

# MMT Numerical Libraries: Dipole Cuboid Inversion

![](doc/_static/area1_grains_scan_sample.jpg)

Python library for the calculation of magnetizations of magnetic grain sources,
which are modelled as homogeneously magnetized particles, using an aggregation
of cuboids. The method requires both the position of grains and magnetic
surface data from the Micromagnetic Tomography project. This is achieved by:

- Modelling the grains as aggregation of cuboids. This data is input into this
  library's main class called `DipoleCuboidInversion`.

- Creating a forward matrix, which is also known as Green's matrix, that is
  obtained from the analytical formulation of the demagnetizing field of the
  cuboids. This matrix is multiplied by the degrees of freedom of the system
  which are the magnetizations of all the grains in the sample. The
  multiplication results into the magnetic field signal imprinted into the scan
  surface. The `DipoleCuboidInversion` class accepts the scan surface data as
  a text input file or Numpy matrix and has methods to calculate the Green's
  matrix.

- Numerically inverting the scan surface data into the grains to obtain their
  individual magnetizations. The inversion is obtained by calculating the
  pseudo-inverse of the Green's matrix using Numpy or Scipy.

The `mmt_dipole_cuboid_inversion` library is optimized to populate the Green's
matrix using either: Numba (compiled function), C parallelized with OpenMP
(parallelization in the number of particles) or NVidia CUDA (high performance
parallelization via the number of sensors in the scan surface).

# Installation

Via PyPI and `pip` (note PyPI names use `-` instead of `_`)

```console
pip install mmt-dipole-cuboid-inversion
```

Or you can use Poetry (recommended for development and CUDA, see below)

```console
poetry install
```
# Documentation

Please visit the official ![Documentation](https://mmt-dipole-cuboid-inversion.readthedocs.io/en/latest/?badge=latest)

## CUDA

To build the code with the `cuda` option to populate the Green's matrix, it is
necessary to define the `CUDAHOME` variable pointing to the `cuda` folder
(assuming you have a functional `cuda` installation), e.g.

```console
export CUDAHOME=/usr/local/cuda-11.5/
```

Then you can compile the code using `poetry install`.

## Poetry

This library is built using the `poetry` library. After cloning the repository
a `virtualenv` will be created automatically when running `poetry install`,
unless you are already in a `virtualenv`, for example, creating one via
`conda`. Within this environment it is possible to run and test the code using
`poetry run`:

```
git clone https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion
cd mmt_dipole_cuboid_inversion
poetry install
poetry run python test/generate_single_dipole.py
```

For more information see this
![link](https://python-poetry.org/docs/managing-environments/). If the package
requires to be built for publication (in the PyPI repository for example) or to
be installed via `pip`, you can run

```console
poetry build
```

This will produce a `dist` folder containing a `tar.gz` file and a `wheel`
file. These files can be installed via `pip`. 

## PyPI

Install from PyPI. Only Linux and Windows builds are currently available.

```console
pip install mmt-dipole-cuboid-inversion
```

## How to use

You can call the class by doing:

```
import mmt_dipole_cuboid_inversion as dci

data = dci.DipoleCuboidInversion(...)
...
```

An Example notebook file with example data is included.

# Cite

If you find this library useful please cite us (you might need LaTeX's `url`
package). Check Github's Citation in the About menu, or the following Bibtex
entries:

    @Misc{Out2023,
      author       = {Out, Frenk and Cortés-Ortuño, David and Kosters, Martha and Fabian, Karl and de Groot, Lennart V.},
      title        = {{MMT Numerical Libraries: Dipole Cuboid Inversion}},
      publisher    = {Zenodo},
      note         = {Github: \url{https://github.com/Micromagnetic-Tomography/mmt_dipole_cuboid_inversion}},
      year         = {2023},
      doi          = {10.5281/zenodo.7823450},
      url          = {https://doi.org/10.5281/zenodo.7823450},
    }

If you have a new version of `biblatex` you can also use `@Software` instead of 
`@Misc`, and add a `version={}` entry. You can also cite the paper with the
theoretical framework of this library:

    @article{Out2023,
    ...
    }
