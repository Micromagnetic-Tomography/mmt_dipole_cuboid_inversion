# Dipole Inverse

Alpha version of a Python library for the calculation of dipole magnetizations using both the position of grains and magnetic surface data from the Micromagnetic Tomography project. This is achieved by:

- Modelling the grains as aggregation of cuboids. This data is inputted into this library's main class called `Dipole`.

- Creating a forward matrix, also known as Green's matrix, that is obtained from the analytical formulation of the demagnetizing field of the cuboids. This matrix is multiplied by the degrees of freedom of the system which are the magnetizations of all the grains in the sample. The multiplication results into the magnetic field signal imprinted into the scan surface. The `Dipole` class accepts the scan surface data as a text input file or Numpy matrix and has methods to calculate the Green's matrix.

- Numerically inverting the scan surface data into the grains to obtain their individual magnetizations. The inversion is obtained by calculating the pseudo-inverse of the Green's matrix using Numpy or Scipy.

The `dipole_inverse` library is optimized to populate/generate the Green's matrix using either: Numba (compiled function), C parallelized with OpenMP (parallelization in the number of particles) or NVidia cuda (high performance parallelization via the number of sensors in the scan surface).

## Install

(TO BE UPDATED) Clone this repository and then you can `pip` install this library:

```console
git clone https://github.com/Micromagnetic-Tomography/dipole_inverse
cd dipole_inverse
pip install . -U
```

###

To build the code with the `cuda` option to populate the Green's matrix, it is necessary to define the `CUDAHOME` variable pointing to the `cuda` folder (assuming you have a functional `cuda` installation), e.g.

```console
export CUDAHOME=/usr/local/cuda-11.5/
```

### Develop

This library is built using the `poetry` library. After cloning the repository a `virtualenv` will be created automatically when running `poetry install`, unless you are already in a `virtualenv`, for example, creating one via `conda`. Within this environment it is possible to run and test the code using `poetry run`:

```
git clone https://github.com/Micromagnetic-Tomography/dipole_inverse
cd dipole_inverse
poetry install
poetry run python test/generate_single_dipole.py
```

For more information see ![](https://python-poetry.org/docs/managing-environments/). If the package requires to be built for publication (in the PyPI repository for example) or to be installed via `pip`, you can run

```console
poetry build
```

that will produce a `dist` folder containing a `tar.gz` file and a `wheel` file. These files can be installed via `pip`. 

### Test

After committing changes please run the test `test/test_single_dipole.py` using `pytest` to check your changes. More tests are required to make the code more robust.

TODO: configure Github actions to run tests automatically. This can also be used to produce packages for different platforms such as 
Windows, MacOS and Linux.

## How to use

You can call the class by doing:

```
import dipole_inverse as dpinv

data = dpinv.Dipole(...)
...
```

An Example notebook file with example data is included.
