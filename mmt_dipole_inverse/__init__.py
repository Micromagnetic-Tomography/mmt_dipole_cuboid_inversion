# Show only the main class:
from .dipole_inverse import Dipole
from . import cython_lib

from .__about__ import (
    __author__,
    __copyright__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __uri__,
    __version__,
)

__all__ = [
    "Dipole",
    "cython_lib",
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]
