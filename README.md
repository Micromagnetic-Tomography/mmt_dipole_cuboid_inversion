[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# MMT Numerical Libraries: Dipole Inverse

Python library for the calculation of dipole magnetizations from magnetic grain
sources using both the position of grains and magnetic surface data from the
Micromagnetic Tomography project. This is achieved by:

- Modelling the grains as aggregation of cuboids. This data is input into this
  library's main class called `Dipole`.

- Creating a forward matrix, which is also known as Green's matrix, that is
  obtained from the analytical formulation of the demagnetizing field of the
  cuboids. This matrix is multiplied by the degrees of freedom of the system
  which are the magnetizations of all the grains in the sample. The
  multiplication results into the magnetic field signal imprinted into the scan
  surface. The `Dipole` class accepts the scan surface data as a text input
  file or Numpy matrix and has methods to calculate the Green's matrix.

- Numerically inverting the scan surface data into the grains to obtain their
  individual magnetizations. The inversion is obtained by calculating the
  pseudo-inverse of the Green's matrix using Numpy or Scipy.

The `mmt_dipole_inverse` library is optimized to populate the Green's matrix
using either: Numba (compiled function), C parallelized with OpenMP
(parallelization in the number of particles) or NVidia CUDA (high performance
parallelization via the number of sensors in the scan surface).

# Installation

Via PyPI and `pip` (note PyPI names use `-` instead of `_`)

```console
pip install mmt-dipole-inverse
```

Or you can use Poetry (recommended for development and CUDA, see below)

```console
poetry install
```

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
git clone https://github.com/Micromagnetic-Tomography/mmt_dipole_inverse
cd mmt_dipole_inverse
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

that will produce a `dist` folder containing a `tar.gz` file and a `wheel`
file. These files can be installed via `pip`. 

### Test

After committing changes please run the test `test/test_single_dipole.py` using
`pytest` to check your changes. More tests are required to make the code more
robust.

## How to use

You can call the class by doing:

```
import dipole_inverse as dpinv

data = dpinv.Dipole(...)
...
```

An Example notebook file with example data is included.

# Cite

If you find this library useful please cite us (you might need LaTeX's
`url` package)

    @Misc{Out2023,
      author       = {Out, Frenk and Cortés-Ortuño, David and Kosters, Martha and Fabian, Karl and de Groot, Lennart V.},
      title        = {{MMT Numerical Libraries: Dipole Inversion}},
      publisher    = {Zenodo},
      note         = {Github: \url{https://github.com/Micromagnetic-Tomography/mmt_dipole_inversion}},
      year         = {2022},
      doi          = {10.5281/zenodo.XXXXX},
      url          = {https://doi.org/10.5281/zenodo.XXXXX},
    }

If you have a new version of `biblatex` you can also use `@Software` instead of 
`@Misc`, and add a `version={}` entry. You can also cite the paper with the
theoretical framework of this library:

    @article{Out2023,
    ...
    }
