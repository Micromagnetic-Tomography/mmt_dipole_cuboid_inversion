import setuptools
from setuptools.extension import Extension
from Cython.Distutils import build_ext
# import sys
# cython and python dependency is handled by pyproject.toml
from Cython.Build import cythonize
import numpy
import os
from os.path import join as pjoin

# -----------------------------------------------------------------------------
# CUDA SPECIFIC FUNCTIONS

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()
print(CUDA)

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# -----------------------------------------------------------------------------

# Compilation of C module in c_lib
com_args = ['-std=c99', '-O3', '-fopenmp']
link_args = ['-fopenmp']
extensions = [
    Extension("dipole_inverse.cython_lib.pop_matrix_lib",
              ["dipole_inverse/cython_lib/pop_matrix_lib.pyx",
               "dipole_inverse/cython_lib/pop_matrix_C_lib.c"],
              extra_compile_args={'gcc': com_args},
              extra_link_args=link_args,
              include_dirs=[numpy.get_include()]
              ),
    #
    # Extension("dipole_inverse.cython_cuda_lib.pop_matrix_cudalib",
    #           ["dipole_inverse/cython_cuda_lib/pop_matrix_cudalib.pyx",
    #            "dipole_inverse/cython_cuda_lib/pop_matrix_cuda_lib.c"],
    #           extra_compile_args=com_args,
    #           extra_link_args=link_args,
    #           include_dirs=[numpy.get_include()]
    #           ),

    Extension("dipole_inverse.cython_cuda_lib.pop_matrix_cudalib",
              sources=["dipole_inverse/cython_cuda_lib/pop_matrix_cudalib.pyx",
                       "dipole_inverse/cython_cuda_lib/pop_matrix_cuda_lib.cu"],
              # library_dirs=[CUDA['lib64']],
              libraries=['cudart'],
              language='c++',
              # This syntax is specific to this build system
              # We're only going to use certain compiler args with nvcc and not with gcc
              # the implementation of this trick is in customize_compiler() below
              # For nvcc we use the Turing architecture: sm_75
              # See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
              # FMAD (floating-point multiply-add): turning off helps for numerical precission (useful
              #                                     for graphics) but this might slightly affect performance
              extra_compile_args={'gcc': com_args,
                                  'nvcc': ['-arch=sm_75', '--fmad=false', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
              include_dirs = [numpy.get_include(), CUDA['include'], '.'],
              library_dirs=[CUDA['lib64']],
              runtime_library_dirs=[CUDA['lib64']]
              )
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

    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},

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

    # since the package has c code, the egg cannot be zipped
    zip_safe=False
)
