import setuptools
# from setuptools.extension import Extension
import sys

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    # setup_requires=['cython'],  # not working (see the link at top)
    name='dipole_inverse',
    version='1.0',
    description=('Python lib to calculate dipole magnetization'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='F. Out, D. Cortes, M. Kosters, K. Fabian, L. V. de Groot',
    author_email='f.out@students.uu.nl',
    packages=setuptools.find_packages(),
    install_requires=['matplotlib',
                      'numpy',
                      'scipy',
                      'numba',
                      'descartes',
                      'pathlib',
                      'shapely'
                      ],
    # TODO: Update license
    classifiers=['License :: BSD2 License',
                 'Programming Language :: Python :: 3 :: Only',
                 ],
)
