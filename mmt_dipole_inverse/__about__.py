import importlib.metadata

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

__title__ = "mmt_dipole_inverse"
__summary__ = (
    "Library to compute magnetizations from numerical inversions of surface "
    "scan data into grain sources modelled as cuboid aggregations"
)
__uri__ = "https://github.com/Micromagnetic-Tomography/mmt_dipole_inverse"

__version__ = importlib.metadata.version("mmt_dipole_inverse")

__author__ = "F. Out, D. Cortés-Ortuño, M. Kosters, K. Fabian, L. V. de Groot"
__email__ = "f.out@uu.nl"

__license__ = "MIT"
__copyright__ = "Copyright 2022 {}".format(__author__)
