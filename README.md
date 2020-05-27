# Dipole Inverse

First version of a Python library for the calculation of dipole magnetization using position of grains and magnetic surface data
from the Micromagnetic Tomography project.

This class is made by combining David's numba code for setting up the G matrix with the codes Frenk used for dipole magnetization. So no Fortran is needed!

## Install

Clone this repository and then you can `pip` install this library:

```
git clone https://github.com/Micromagnetic-Tomography/dipole_inverse
cd dipole_inverse
pip install . -U
```

## How to use

You can call the class by doing:

```
import dipole_inverse as dpinv

data = dpinv.Dipole(...)
...
```
An Example notebook file with example data is included.