from mmt_dipole_inverse.tools import CuboidDecomposition
from pathlib import Path
import numpy as np


def test_cuboid_decomposition():
    parent = Path(__file__).resolve().parent
    my_data = parent / 'grain_voxel_data.txt'

    CuboidDecomposition(my_data, 'cuboid_decomposed_data.txt',
                        format_output=True)


if __name__ == "__main__":
    test_cuboid_decomposition()
