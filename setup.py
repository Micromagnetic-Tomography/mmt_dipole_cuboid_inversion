import setuptools
from setuptools.extension import Extension
# import sys
# cython and python dependency is handled by pyproject.toml
from Cython.Build import cythonize
import numpy


# -----------------------------------------------------------------------------
# Compilation of C module in c_lib
com_args = ['-std=c99', '-O3', '-fopenmp']
link_args = ['-fopenmp']
extensions = [
    Extension("dipole_inverse.cython_lib.pop_matrix_lib",
              ["dipole_inverse/cython_lib/pop_matrix_lib.pyx",
               "dipole_inverse/cython_lib/pop_matrix_C_lib.c"],
              extra_compile_args=com_args,
              extra_link_args=link_args,
              include_dirs=[numpy.get_include()]
              ),
    #
    Extension("dipole_inverse.cython_cuda_lib.pop_matrix_cudalib",
              ["dipole_inverse/cython_cuda_lib/pop_matrix_cudalib.pyx",
               "dipole_inverse/cython_cuda_lib/pop_matrix_cuda_lib.c"],
              extra_compile_args=com_args,
              extra_link_args=link_args,
              include_dirs=[numpy.get_include()]
              ),
]

# -----------------------------------------------------------------------------

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    # setup_requires=['cython'],  # not working (see the link at top)
    name='dipole_inverse',
    version='1.9',
    description=('Python lib to calculate dipole magnetization'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='F. Out, D. Cortes, M. Kosters, K. Fabian, L. V. de Groot',
    author_email='f.out@students.uu.nl',
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=['matplotlib',
                      'numpy>=1.20',
                      'scipy>=1.6',
                      'numba>=0.51',
                      'descartes',
                      'pathlib',
                      'shapely',
                      # The following is a dependency in a private repository:
                      'grain_geometry_tools @ git+ssh://git@github.com/Micromagnetic-Tomography/grain_geometry_tools'
                      ],

    # TODO: Update license
    classifiers=['License :: BSD2 License',
                 'Programming Language :: Python :: 3 :: Only',
                 ],
)
