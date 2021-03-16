import numpy as np
import numba as nb
from pathlib import Path
import scipy.linalg as spl
from .cython_lib import pop_matrix_lib    # the cython populate_matrix function
from typing import Literal  # Working with Python >3.8
from typing import Union    # Working with Python >3.8
import os


@nb.jit(nopython=True)
def populate_matrix_numba(G, QDM_domain, scan_height, cuboids, Npart,
                          Ny, Nx, QDM_spacing, QDM_deltax, QDM_deltay,
                          Origin, verbose=True):
    """
    Main function to populate the G matrix

    Notes
    -----
    The outer while loop will last until reaching the total number of cuboids
    in the sample. Adjacent cuboids belong to a single particle, which is
    indexed in the 6th element of the cuboids array. The population of the G
    matrix is performed column wise for every particle. For each cuboid
    belonging to a particle, their contribution to the magnetic flux is summed
    up for every sensor measurement in steps of delta in the xy plane, which
    are given by the loops with the i-j indexes. The flux is stored column
    wise.

    Parameters
    ----------
    QDM_domain
        Array of size 2x2 with the lower left and upper right coordinates of
        the scan surface
    Origin
        If True the scan data is set to the QDM lower left corner coordinates.
        If False, the scan data origin is set at (0., 0.)
    """

    Cm = 1e-7
    if Origin is True:
        xi0, eta0 = QDM_domain[0, :]
    else:
        xi0, eta0 = 0., 0.
    zeta0 = (-1) * scan_height
    sensor_pos = np.zeros(3)
    sensor_pos[2] = zeta0

    # Definitions
    particle_flux = np.zeros(3)
    get_flux = np.zeros(3)
    cuboid_center = np.zeros(3)
    dr_cuboid = np.zeros(3)
    cuboid_size = np.zeros(3)

    i_cuboid = 0
    i_particle_prev = int(cuboids[0, 6])
    i_particle = i_particle_prev

    # print('max cub =', Npart)
    # print('G matrix', G.shape)
    # If grains are not numbered in order this always works
    i_particle_0_N = 0

    while i_cuboid < len(cuboids):
        if verbose:
            # print(f'Particle = {i_particle}  Cuboid = {i_cuboid}')
            print('Particle =', i_particle, 'Cuboid =', i_cuboid)
            # print(particle =)

        i_cuboid_old = i_cuboid

        # Loop over sensor measurements. Each sensor is in the xy
        # plane and has area delta^2
        for j in range(Ny):
            sensor_pos[1] = eta0 + QDM_spacing * j
            for i in range(Nx):
                sensor_pos[0] = xi0 + QDM_spacing * i

                # The contribution of the flux for mx, my, mz
                particle_flux[:] = 0

                # Start from the index of the particle being analysed
                i_particle = int(cuboids[i_cuboid_old, 6])
                i_cuboid = i_cuboid_old

                # While the cuboid has particle index of the
                # particle being analysed
                while i_particle == i_particle_prev:
                    #                     print(i_particle, i, j, i_cuboid)
                    cuboid_center[:] = cuboids[i_cuboid, :3]
                    dr_cuboid[:] = cuboid_center - sensor_pos
                    # Cuboid sizes:
                    cuboid_size[:] = cuboids[i_cuboid, 3:6]

                    # calculate flux per cuboid
                    get_flux[:] = 0.
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            for s3 in [-1, 1]:
                                for s4 in [-1, 1]:
                                    for s5 in [-1, 1]:
                                        x = dr_cuboid[0] + s1 * cuboid_size[0] - s4 * QDM_deltax
                                        y = dr_cuboid[1] + s2 * cuboid_size[1] - s5 * QDM_deltay
                                        z = dr_cuboid[2] + s3 * cuboid_size[2]
                                        sign = s1 * s2 * s3 * s4 * s5
                                        x2, y2, z2 = x ** 2, y ** 2, z ** 2
                                        r2 = x2 + y2 + z2
                                        r = np.sqrt(r2)
                                        Az = np.arctan2(x * y, z * r)
                                        if r != 0.0:
                                            Lx = np.log(x + r)
                                            Ly = np.log(y + r)
                                        else:
                                            Lx, Ly = 0., 0.
                                            print('Error at p = ', i_particle)

                                        F120 = 0.5 * ((y2 - z2) * Lx - r * x) - y * (z * Az - x * Ly)
                                        F210 = 0.5 * ((x2 - z2) * Ly - r * y) - x * (z * Az - y * Lx)
                                        F22m = -x * y * Az - z * (x * Lx + y * Ly - r)

                                        get_flux[0] += sign * F120
                                        get_flux[1] += sign * F210
                                        get_flux[2] += sign * F22m

                    # Finish cuboidsloop in the particle i_particle_prev
                    # and continue with the next sensor measurement

                    # scale flux measurement:
                    particle_flux[:] += -Cm * get_flux
                    i_cuboid += 1
                    i_particle = int(cuboids[i_cuboid, 6])

                # print(i + j * Nx, 3 * i_particle_prev)
                # Populate G matrix column wise
                G[i + j * Nx, 3 * i_particle_0_N] = particle_flux[0]
                G[i + j * Nx, 3 * i_particle_0_N + 1] = particle_flux[1]
                G[i + j * Nx, 3 * i_particle_0_N + 2] = particle_flux[2]

        i_particle_prev = i_particle
        i_particle_0_N += 1
    return G


class Dipole(object):

    def __init__(self,
                 QDM_data: str,
                 cuboid_data: str,
                 QDM_domain: np.ndarray,
                 QDM_spacing: float,
                 QDM_deltax: float,
                 QDM_deltay: float,
                 QDM_area: float,
                 sample_height: float,
                 scan_height: float,
                 tol: float = 1e-7,
                 max_num_threads: int = -1
                 ):
        """
        This class calculates the magnetization of a group of magnetic grains
        from a surface with magnetic field scan data.

        Parameters
        ----------
        QDM_data
            Matrixfile (Nx columns, Ny rows) containing the QDM/scan data in T
        cuboid_data
            File (x, y, z, dx, dy, dz, index) containing location and size
            grains in microm
        QDM_domain
            (2x2 numpy matrix) : Size (metres) of the QDM domain as
             np.array([[x1, y1], [x2, y2]])
        QDM_spacing
            Distance between two adjacent scanning points in metres
        QDM_deltax
            Half length of QDM sensor
        QDM_deltay
            Half width of QDM sensor
        QDM_area
            Area of QDM sensor in square metres
        sample_height
            Thickness of sample in metres
        scan_height
            Distance between sample and QDM scanner in metres
        tol
            Tolerance for checking QDM_domain. Default is 1e-7
        max_num_threads
            Limits the maximum number of threads used by
            parallel/multi-threaded functions. This is done by setting multiple
            environment variables. This can be updated at any time.
            See the docstring of the self.max_num_threads variable for details.

        Attributes
        ----------
        QDM_data
        cuboid_data
        QDM_domain
        QDM_spacing
        QDM_deltax
        QDM_deltay
        QDM_area
        sample_height
        scan_height
        Nx, Ny
        QDM_domain

        cuboids


        """

        self.QDM_data = Path(QDM_data)
        self.cuboid_data = Path(cuboid_data)
        self.QDM_domain = QDM_domain
        self.QDM_spacing = QDM_spacing
        self.QDM_deltax = QDM_deltax
        self.QDM_deltay = QDM_deltay
        self.QDM_area = QDM_area
        self.sample_height = sample_height
        self.scan_height = scan_height

        self._max_num_threads = "-1"
        # somehow setting str manually (instead of the setter doing this)
        # avoids mypy to complain -> TODO: make minimal example
        self.max_num_threads = str(max_num_threads)

        self.Ny, self.Nx = np.loadtxt(QDM_data).shape
        new_domain = self.QDM_domain[0, 0] \
            + (self.Nx - 1) * self.QDM_spacing
        if abs(new_domain - self.QDM_domain[1, 0]) > tol:
            print(f'QDM_domain[1, 0] has been reset from '
                  f'{self.QDM_domain[1, 0]} to {new_domain}.')
            self.QDM_domain[1, 0] = new_domain

        new_domain = self.QDM_domain[0, 1] \
            + (self.Ny - 1) * self.QDM_spacing
        if abs(new_domain - self.QDM_domain[1, 1]) > tol:
            print(f'QDM_domain[1, 1] has been reset from '
                  f'{self.QDM_domain[1, 1]} to {new_domain}.')
            self.QDM_domain[1, 1] = new_domain

        if abs(self.QDM_deltax * self.QDM_deltay * 4
               - self.QDM_area) > tol**2:
            print('The sensor is not a rectangle. '
                  'Calculation will probably go wrong here!')

    @property
    def max_num_threads(self) -> str:
        """Get the maximum number of threads and according to this value set
        multiple environment variables to limit the threads used in
        parallel/multi-threaded functions. This property will limit:
            OPENMP, OPENBLAS, MKL, VECLIB and NUMEXPR

        If max_num_threads is set to a value < 0, environment variables are not
        updated
        """
        return self._max_num_threads

    @max_num_threads.setter
    def max_num_threads(self, max_threads: Union[int, str]):
        self._max_num_threads = str(max_threads)
        if int(max_threads) > 0:
            os.environ["OMP_NUM_THREADS"] = self._max_num_threads
            os.environ["OPENBLAS_NUM_THREADS"] = self._max_num_threads
            os.environ["MKL_NUM_THREADS"] = self._max_num_threads
            os.environ["VECLIB_MAXIMUM_THREADS"] = self._max_num_threads
            os.environ["NUMEXPR_NUM_THREADS"] = self._max_num_threads

    def read_files(self, factor: float = 1e-6):
        """ Reads in QDM_data and cuboid_data
        """

        self.QDM_matrix = np.loadtxt(self.QDM_data) * self.QDM_area
        self.cuboids = np.loadtxt(self.cuboid_data)
        self.cuboids[:, :6] = self.cuboids[:, :6] * factor
        self.Npart = len(np.unique(self.cuboids[:, 6]))
        self.Ncub = len(self.cuboids[:, 6])

    _PrepMatOps = Literal['cython', 'numba']

    def prepare_matrix(self,
                       Origin: bool = True,
                       verbose: bool = True,
                       method: _PrepMatOps = 'cython'
                       ):
        """ Allocates/instatiates the Numpy arrays to populate the forward
        matrix

        Parameters
        ----------
        Origin
            If True, use the QDM_domain lower left coordinates as the scan grid
            origin. If False, set scan grid origin at (0., 0.)
        verbose
            Set to True to print log information when populating the matrix
        method
            Populating the matrix can be done using either 'numba' or 'cython'
            optimisation.
            The cython function is parallelized with OpenMP thus the number of
            threads is specified from the OMP_NUM_THREADS system variable. This
            can be limited by setting self.max_num_threads
        """

        self.Forward_G = np.zeros((self.Nx * self.Ny, 3 * self.Npart))

        if method == 'cython':
            pop_matrix_lib.populate_matrix_cython(
                self.Forward_G.T, self.QDM_domain[0], self.scan_height,
                np.ravel(self.cuboids),
                self.Ncub, self.Npart, self.Ny, self.Nx,
                self.QDM_spacing, self.QDM_deltax, self.QDM_deltay,
                Origin, int(verbose))

        elif method == 'numba':
            self.Forward_G = populate_matrix_numba(
                self.Forward_G, self.QDM_domain, self.scan_height,
                self.cuboids, self.Npart, self.Ny, self.Nx,
                self.QDM_spacing, self.QDM_deltax, self.QDM_deltay,
                Origin=Origin, verbose=verbose)

    _MethodOps = Literal['scipy_lapack',
                         'scipy_pinv',
                         'scipy_pinv2'
                         'numpy_pinv']

    def calculate_inverse(self,
                          method: _MethodOps = 'scipy_pinv',
                          **method_kwargs):
        """
        Calculates the inverse and computes the magnetization.  The solution is
        generated in the self.Mag variable

        Parameters
        ----------
        method
            The numerical inversion can be done using the SVD algorithms or the
            least squares method. The options available are:

            scipy_lapack    :: Uses scipy lapack wrappers for dgetrs and dgetrf
            scipy_pinv      :: Least squares method
            scipy_pinv2     :: SVD method
            numpy_pinv      :: SVD method

        Notes
        -----
        Additional keyword arguments are passed to the solver, e.g.

            calculate_inverse(method='numpy_pinv', rcond=1e-15)

        """
        SUCC_MSG = 'Inversion has been carried out'

        QDM_flatten = self.QDM_matrix.flatten()
        if self.Forward_G.shape[0] >= self.Forward_G.shape[1]:
            print(f'Start inversion with {self.Forward_G.shape[0]} '
                  f'knowns and {self.Forward_G.shape[1]} unknowns')
            # probably there is a more efficient way to write these options
            if method == 'scipy_pinv':
                Inverse_G = spl.pinv(self.Forward_G, **method_kwargs)
                self.Mag = np.matmul(Inverse_G, QDM_flatten)
                print(SUCC_MSG)
            elif method == 'scipy_pinv2':
                Inverse_G = spl.pinv2(self.Forward_G, **method_kwargs)
                self.Mag = np.matmul(Inverse_G, QDM_flatten)
                print(SUCC_MSG)
            elif method == 'numpy_pinv':
                Inverse_G = np.linalg.pinv(self.Forward_G, **method_kwargs)
                self.Mag = np.matmul(Inverse_G, QDM_flatten)
                print(SUCC_MSG)

            elif method == 'scipy_lapack':
                GtG = np.matmul(self.Forward_G.transpose,
                                self.Forward_G)
                GtG_shuffle, IPIV, INFO1 = spl.lapack.dgetrf(GtG)
                if INFO1 == 0:
                    print(SUCC_MSG)
                    GtQDM = np.matmul(self.Forward_G, QDM_flatten)
                    self.Mag, INFO2 = spl.lapack.dgetrs(GtG_shuffle, IPIV, GtQDM)
                    if INFO2 != 0:
                        self.Mag = None
                        print(f'{INFO2}th argument has an'
                              'illegal value. self.Mag deleted')
                    else:
                        print(SUCC_MSG)
                else:
                    print(f'{INFO1}th argument has an illegal value')

            else:
                print(f'Method {method} is not recognized')
        else:
            print(f'Problem is underdetermined with '
                  f'{self.Forward_G.shape[0]} knowns and '
                  f'{self.Forward_G.shape[1]} unknowns')

    def obtain_magnetization(self,
                             verbose: bool = True,
                             method: _MethodOps = 'scipy_pinv',
                             **method_kwargs):
        """
        A shortcut method to call three functions to compute the magnetization
        of the grains. See self.calculate_inverse docstring for details about
        the method parameter.
        """

        self.read_files()
        self.prepare_matrix(verbose=verbose)
        self.calculate_inverse(method=method, **method_kwargs)

    def save_results(self,
                     Magfile: str,
                     keyfile: str,
                     path_to_plot: str = None,
                     colormap: str = 'coolwarm'):
        """
        Saves the magnetization to a specified Magfile file and the keys of the
        index of the particles in the keyfile file.

        (to be removed)
        An optional plot is produced if path_to_plot (string) is set.
        colormap default set at coolwarm
        """

        # WARNING: the old version did not save the indexes as 1st column:
        # np.savetxt(Magfile, self.Mag.reshape(self.Npart, 3))

        # Sort indexes
        _, sort_idx = np.unique(self.cuboids[:, 6], return_index=True)
        p_idxs = self.cuboids[:, 6][np.sort(sort_idx)]
        # Sort the np.unique indexes to get the cuboid idxs in the orig order
        # Check:
        # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
        data = np.column_stack((p_idxs, self.Mag.reshape(self.Npart, 3)))

        np.savetxt(Magfile, data)
        np.savetxt(keyfile, p_idxs)

    def forward_field(self, filepath, snrfile=None, tol=0.9):

        """ Calculates the forward field and signal to noise ratio and saves
        them (SNR saving is optional)

        Parameters
        ----------
        filepath
            Path to file to save the forward field
        snrfile
            If specified, saves the SNR
        tol
            Stands for percentage of signal used (0.9 is 90% default)
        """

        Forward_field = np.matmul(self.Forward_G, self.Mag)  # flux field
        np.savetxt(filepath, Forward_field.reshape(self.Ny, self.Nx)
                   / self.QDM_area)
        if snrfile is not None:
            org_field = self.QDM_matrix.flatten()  # flux field
            residual = org_field - Forward_field
            snr = np.zeros(self.Forward_G.shape[1])
            el_signal = np.zeros((self.Forward_G.shape[0], 2))
            for column in range(self.Forward_G.shape[1]):
                el_signal[:, 0] = self.Forward_G[:, column] * self.Mag[column]
                el_signal[:, 1] = residual
                el_sum = np.sqrt(np.sum((el_signal[:, 0])**2))
                el_signal = el_signal[np.argsort(abs(el_signal[:, 0]))]
                res2_sum = 0
                forw2_sum = 0
#                 forw_sum = 0
                for item in range(1, len(el_signal[:, 0]) + 1):
                    res2_sum += el_signal[-item, 1]**2
                    forw2_sum += el_signal[-item, 0]**2
#                     forw_sum += abs(el_signal[-item, 0])
                    if np.sqrt(forw2_sum) / el_sum > tol:
                        res2_sum = np.sqrt(res2_sum)
                        forw2_sum = np.sqrt(forw2_sum)
                        snr[column] = forw2_sum / res2_sum
                        break
            np.savetxt(snrfile, snr)
