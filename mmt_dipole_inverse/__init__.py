import importlib.metadata
from .dipole_inverse import Dipole
from . import cython_lib
from . import plot_tools


__version__ = importlib.metadata.version("dipole_inverse")
