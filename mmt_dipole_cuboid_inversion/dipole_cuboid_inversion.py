# Allow class annotations in classmethod
from __future__ import annotations
import numpy as np
from pathlib import Path
import scipy.linalg as spl
from .numba_lib import populate_matrix_numba
from .cython_lib import pop_matrix_lib   # the cython populate_matrix function
try:
    from .cython_cuda_lib import pop_matrix_cudalib   # the cuda populate_matrix function
    HASCUDA = True
except ImportError:
    HASCUDA = False
from typing import Literal, Union, Tuple, Optional
import numpy.typing as npt
# from typing import Type      # Working with Python >3.8
import json
import warnings
# Make a proper logging system if we grow this library:
# import logging  # def at __init__ file
# logging.getLogger(__name__)


def loadtxt_iter(txtfile, delimiter=None, skiprows=0, dtype=np.float64):
    """Reads a simply formatted text file using Numpy's `fromiter` function.
    This function should perform faster than the `loadtxt` function.

    Parameters
    ----------
    txtfile
        Path to text file
    delimiter
        Passed to `split(delimiter=)` in every line of the text file.
        `None` means any number of white spaces
    skiprows
    dtype

    Notes
    -----
    Based on N. Schlomer function at:

    https://stackoverflow.com/questions/18259393/numpy-loading-csv-too-slow-compared-to-matlab

    and J. Kington at

    https://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy
    """

    def iter_func():
        line = ''
        with open(txtfile, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                # Might not be necessary to strip characters at start and end:
                line = line.strip().split(delimiter)
                # line = line.split(delimiter)
                # As a general solution we can also use regex:
                # re.split(" +", line)
                for item in line:
                    yield dtype(item)
        if len(line) == 0:
            raise Exception(f'Empty file: {txtfile}')
        loadtxt_iter.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype).flatten()
    data = data.reshape((-1, loadtxt_iter.rowlength))

    return data


class DipoleCuboidInversion(object):

    def __init__(self,
                 scan_domain: Optional[npt.NDArray[np.float64]] = None,
                 sensor_center_domain: Optional[npt.NDArray[np.float64]] = None,
                 scan_spacing: float | Tuple[float, float] = 1e-6,
                 scan_deltax: float = 0.5e-6,
                 scan_deltay: float = 0.5e-6,
                 scan_area: float = 1e-12,
                 scan_height: float = 1e-6,
                 verbose: bool = True
                 ):
        """Class to obtain the magnetization from grains modelled as cuboids

        The magnetization is computed via numerical inversion from a surface
        with magnetic field scan data (from microscopy, such as Quantum Diamond
        Microscopy), and both grain locations and geometries from tomographic
        data. Grains are modelled as aggregations of cuboids with dipole order
        magnetic moments.

        The scan surface data and cuboid data are loaded using the `read_files`
        array. Then, the scan domain coordinates are defined when calling the
        `set_scan_domain` method. The scan domain is defined as a regular mesh
        of sensors whose centers are spaced according to `scan_spacing`.

        The sensors in the scan surface are modelled with rectangular geometry
        with side lengths given by the `scan_delta*` variables.

        Parameters
        ----------
        scan_domain
            (2x2 numpy matrix) : Coordinates, in meters, of the lower left
            and upper right corners of the scanning surface domain as
            `np.array([[x1, y1], [x2, y2]])`
        sensor_center_domain
            (2x2 numpy matrix) : Coordinates, in meters, of the corners of the
            lower left and upper right sensors, of the scanning surface as
            `np.array([[x_center0, y_center0], [x_center1, y_center1]])`
        scan_spacing
            Distance between two adjacent scanning points in metres. Can be
            passed as a float, if the spacing is the same in x and y, or as a tuple
        scan_deltax
            Half length of scan sensor (can be different to scan spacing)
        scan_deltay
            Half width of scan sensor (can be different to scan spacing)
        scan_area
            Area of scan sensor in square metres
        scan_height
            Distance between sample and scan surface in meters. If this
            parameter is defined negative, it is assumed that we have a left
            handed coordinate system with the z-direction pointing downwards,
            i.e. towards depth, so cuboids must be defined with positive
            z-positions. If `scan_height` is positive, then we have a right
            handed system and cuboid's z-positions must have negative values
        verbose
            Print extra information about the functions to populate the matrix,
            the inversions and other methods. Can be changed at any time.

        Attributes
        ----------
        scan_data
        cuboid_data
        scan_domain
        sensor_center_domain
        scan_spacing
        scan_deltax
        scan_deltay
        scan_area
        scan_height
        Nx, Ny
        scan_domain

        Notes
        -----
        The `read_files` and `set_scan_domain` function set::

            scan_matrix
            cuboids
            Nx, Ny
            Npart
            Ncub
            scan_domain (*)
            sensor_center_domain (*)
            scan_spacing (*)

        (*) Depending on the `gen_sd_mesh_from` method

        """

        self.scan_domain = np.zeros((2, 2)) if scan_domain is None else scan_domain
        self.sensor_center_domain = np.zeros((2, 2)) if sensor_center_domain is None else sensor_center_domain

        if isinstance(scan_spacing, Tuple):
            self.scan_spacing = np.array(scan_spacing)
        else:
            self.scan_spacing = np.array([scan_spacing, scan_spacing])

        self.scan_deltax = scan_deltax
        self.scan_deltay = scan_deltay
        self.scan_area = scan_area
        self.scan_height = scan_height

        self.Nx, self.Ny = 0, 0
        self.Inverse_G = np.empty([])
        self.Mag = np.empty([])
        self.scan_matrix = np.empty([])
        self.verbose = verbose

    @classmethod
    def from_json(cls, file_path: Union[Path, str], verbose: bool = True) -> DipoleCuboidInversion:
        """Instantiate the class using scanning surface params from a JSON file

        The required JSON keys are::

            'Scan domain LL-x-y'
            'Scan domain UR-x-y'
            'Sensor center domain LL-x-y'
            'Sensor center domain UR-x-y'
            'Scan spacing'
            'Scan delta-x'
            'Scan delta-y'
            'Scan area'
            'Scan height'
        """
        # Load metadata
        with open(file_path, 'r') as f:
            metadict = json.load(f)

        scan_domain = np.array([[metadict.get('Scan domain LL-x-y', 0.0)[0],
                                 metadict.get('Scan domain LL-x-y', 0.0)[1]],
                                [metadict.get('Scan domain UR-x-y', 0.0)[0],
                                 metadict.get('Scan domain UR-x-y', 0.0)[1]]])

        sensor_domain = np.array([[metadict.get('Sensor center domain LL-x-y', 0.0)[0],
                                   metadict.get('Sensor center domain LL-x-y', 0.0)[1]],
                                  [metadict.get('Sensor center domain UR-x-y', 0.0)[0],
                                   metadict.get('Sensor center domain UR-x-y', 0.0)[1]]])

        return cls(scan_domain,
                   sensor_domain,
                   metadict.get('Scan spacing'),
                   metadict.get('Scan delta-x'),
                   metadict.get('Scan delta-y'),
                   metadict.get('Scan area'),
                   metadict.get('Scan height'),
                   verbose=verbose)

    def read_files(self,
                   scan_data: Union[Path, str, npt.NDArray[np.float64]],
                   cuboid_data: Union[Path, str, npt.NDArray[np.float64]],
                   cuboid_scaling_factor: float,
                   scan_matrix_reader_kwargs={},
                   cuboids_reader_kwargs={}):
        """Reads in scan data and cuboid data from text/csv files

        Parameters
        ----------
        scan_data
            File path to a text or `npy` file, `np.ndarray` with
            (`Nx` columns, `Ny` rows), containing the scan data in Tesla
        cuboid_data
            File path, `np.ndarray` containing the location and size of the
            grains in micrometer, with format `(x, y, z, dx, dy, dz, index)`
        cuboid_scaling_factor
            Scaling factor for the cuboid positions and lengths
        scan_matrix_reader_kwargs
            Extra arguments to the reader of the scan file, e.g. `delimiter=','`
        cuboids_reader_kwargs
            Extra arguments to the reader of cuboid files, e.g. `skiprows=2`
        """

        if isinstance(scan_data, (np.ndarray)):
            self.scan_matrix = np.copy(scan_data)
        else:
            try:
                data_path = Path(scan_data)
                if data_path.__str__().endswith('.npy'):
                    self.scan_matrix = np.load(data_path)

                # self.scan_matrix = np.loadtxt(self.scan_data) * self.scan_area
                # Use a faster reader, assuming the scan file is separated by
                # white spaces or another delimiter specified by reader_kwargs
                else:
                    self.scan_matrix = loadtxt_iter(data_path,
                                                    **scan_matrix_reader_kwargs)
            except TypeError:
                print(f'{scan_data} is not a valid file name and cannot be '
                      'loaded. You can also try an np.ndarray')
                raise

        np.multiply(self.scan_matrix, self.scan_area, out=self.scan_matrix)

        # ---------------------------------------------------------------------

        # Read cuboid data in a 2D array
        if isinstance(cuboid_data, (np.ndarray)):
            self.cuboids = np.copy(cuboid_data)
        else:
            try:
                cuboid_path = Path(cuboid_data)
                # self.cuboids = np.loadtxt(self.cuboid_data, ndmin=2)
                # We are assuming here that cuboid file does not have comments
                self.cuboids = loadtxt_iter(cuboid_path, **cuboids_reader_kwargs)
            except TypeError:
                print(f'{cuboid_data} is not a valid file name and cannot be '
                      'loaded. You can also try an np.ndarray')

        self.cuboids[:, :6] = self.cuboids[:, :6] * cuboid_scaling_factor
        self.Npart = len(np.unique(self.cuboids[:, 6]))
        self.Ncub = len(self.cuboids[:, 6])

    _SetScanDomainOps = Literal['sensor_center_domain', 'scan_domain', 'sd_partitioned']

    def set_scan_domain(self,
                        gen_sd_mesh_from: _SetScanDomainOps = 'sensor_center_domain',
                        tol_sd_limits: float = 1e-7):
        """Sets the scan domain limits and spacings

        This function might also correct the limits of the `scan_domain`
        attribute according to the size of the scan data matrix.

        Parameters
        ----------
        gen_sd_mesh_from
            Method to set the limits or spacings of the scan domain. The
            `sensor_center_domain` will use the centers of the lower left and
            upper right sensors of the scan surface, which are defined in the
            variable of the same name. Similarly, The `scan_domain` uses
            the lower left and upper right coordinates of the scan surface, from
            the `scan_domain` variable. These two methods require the scan
            spacings, and the new domain limit is compared to the limits
            given by the upper right point in `*_domain`. The option
            `sd_partitioned` uses the matrix dimensions of `scan_data` to
            partition the `scan_domain` limits by re-defining the
            `scan_spacing` variable.
        tol_sd_limits
            Tolerance for checking `scan_domain` in two of the `gen_sd_mesh_from`
            methods
        """
        if self.scan_matrix.ndim > 1:
            self.Ny, self.Nx = self.scan_matrix.shape
        else:
            raise RuntimeError('Scan domain requires that scan_matrix is defined. '
                               'Run the read_files method or set the 2D scan_matrix array')

        new_domain = np.zeros(2, dtype=np.float64)

        if gen_sd_mesh_from == 'sensor_center_domain':

            new_domain[0] = self.sensor_center_domain[0, 0] + (self.Nx - 1) * self.scan_spacing[0]
            new_domain[1] = self.sensor_center_domain[0, 1] + (self.Ny - 1) * self.scan_spacing[1]
            for i in range(2):
                if abs(new_domain[i] - self.sensor_center_domain[1, i]) > tol_sd_limits:
                    print(f'Domain limit {i} has been reset from {self.sensor_center_domain[1, i]} to {new_domain[i]}.')
                    self.sensor_center_domain[1, i] = new_domain[i]

            self.scan_domain[0] = self.sensor_center_domain[0] - self.scan_spacing
            self.scan_domain[1] = self.sensor_center_domain[1] + self.scan_spacing

        elif gen_sd_mesh_from == 'scan_domain':

            new_domain[0] = self.scan_domain[0, 0] + self.Nx * self.scan_spacing[0]
            new_domain[1] = self.scan_domain[0, 1] + self.Ny * self.scan_spacing[1]
            for i in range(2):
                if abs(new_domain[i] - self.scan_domain[1, i]) > tol_sd_limits:
                    print(f'Domain limit {i} has been reset from {self.scan_domain[1, i]} to {new_domain[i]}.')
                    self.scan_domain[1, i] = new_domain[i]

            self.sensor_center_domain[0] = self.scan_domain[0] + self.scan_spacing
            self.sensor_center_domain[1] = self.scan_domain[1] - self.scan_spacing

        elif gen_sd_mesh_from == 'sd_partitioned':
            self.scan_spacing = ((self.scan_domain[1, 0] - self.scan_domain[0, 0]) / (self.Nx + 1),
                                 (self.scan_domain[1, 1] - self.scan_domain[0, 1]) / (self.Ny + 1))
            if self.verbose:
                print(f'Scan spacing x defined as: {self.scan_spacing[0]}')
                print(f'Scan spacing y defined as: {self.scan_spacing[1]}')

            self.sensor_center_domain[0] = self.scan_domain[0] + self.scan_spacing
            self.sensor_center_domain[1] = self.scan_domain[1] - self.scan_spacing

        else:
            raise TypeError('Specify a valid option to generate the measurement mesh geometry')

        if abs(self.scan_deltax * self.scan_deltay * 4 - self.scan_area) > tol_sd_limits ** 2:
            warnings.warn('The sensor geometry is not a rectangle')

        # ---------------------------------------------------------------------

    _PrepMatOps = Literal['cython', 'numba', 'cuda']

    def prepare_matrix(self,
                       Origin: bool = True,
                       method: _PrepMatOps = 'cython'
                       ):
        """ Allocates/instantiates the Numpy arrays to populate the forward
        matrix

        Parameters
        ----------
        Origin
            If True, use the scan_domain lower left coordinates as the scan grid
            origin. If False, set scan grid origin at (0., 0.)
        method
            Populating the matrix can be done using either `numba` or `cython`
            or (nvidia) `cuda` optimisation.
            The cython function is parallelized with OpenMP thus the number of
            threads is specified from the `OMP_NUM_THREADS` system variable.
            This can be limited using set_max_num_threads in the tools module
        """

        self.Forward_G = np.zeros((self.Nx * self.Ny, 3 * self.Npart),
                                  dtype=np.double)

        if method == 'cython':
            # The Cython function populates the matrix column-wise via a 1D arr
            pop_matrix_lib.populate_matrix_cython(
                self.Forward_G, self.scan_domain[0], self.scan_height,
                np.ravel(self.cuboids), self.Ncub,
                self.Npart, self.Ny, self.Nx,
                self.scan_spacing[0], self.scan_spacing[1],
                self.scan_deltax, self.scan_deltay, Origin, int(self.verbose))

        if method == 'cuda':
            if HASCUDA is False:
                raise Exception('The cuda method is not available. Stopping calculation')

            pop_matrix_cudalib.populate_matrix_cython(
                self.Forward_G, self.scan_domain[0], self.scan_height,
                np.ravel(self.cuboids), self.Ncub,
                self.Npart, self.Ny, self.Nx,
                self.scan_spacing[0], self.scan_spacing[1],
                self.scan_deltax, self.scan_deltay,
                Origin, int(self.verbose))

        elif method == 'numba':
            populate_matrix_numba(
                self.Forward_G, self.scan_domain, self.scan_height,
                self.cuboids, self.Npart, self.Ny, self.Nx,
                self.scan_spacing[0], self.scan_spacing[1],
                self.scan_deltax, self.scan_deltay,
                Origin=Origin, verbose=self.verbose)

    _MethodOps = Literal['scipy_lapack',
                         'scipy_pinv',
                         'scipy_pinv2',
                         'numpy_pinv']

    def calculate_inverse(self,
                          method: _MethodOps = 'scipy_pinv',
                          store_inverse_G_matrix: bool = False,
                          **method_kwargs
                          ) -> None:
        r"""
        Calculates the inverse and computes the magnetization. The solution is
        generated in the self.Mag variable. Optionally, the covariance matrix
        can be established.

        Parameters
        ----------
        method
            The numerical inversion can be done using the SVD algorithms or the
            least squares method. The options available are:

                * scipy_lapack    :: Uses scipy lapack wrappers for dgetrs and
                                     dgetrf to compute :math:`\mathbf{M}` by
                                     solving the matrix least squares problem:
                                     :math:`Gᵀ * G * M = Gᵀ * ϕ_{scan}`
                * scipy_pinv      :: SVD method
                * scipy_pinv2     :: (Deprecated) SVD method, calls pinv
                * numpy_pinv      :: SVD method

        Notes
        -----
        Additional keyword arguments are passed to the solver, e.g::

            calculate_inverse(method='numpy_pinv', rcond=1e-15)
        """
        SUCC_MSG = 'Inversion has been carried out'
        scan_flatten = self.scan_matrix.flatten()
        if self.Forward_G.shape[0] >= self.Forward_G.shape[1]:
            if self.verbose:
                print(f'Start inversion with {self.Forward_G.shape[0]} '
                      f'knowns and {self.Forward_G.shape[1]} unknowns')
            # Probably there is a more efficient way to write these options
            if method == 'scipy_pinv' or 'scipy_pinv2':
                if method == 'scipy_pinv2':
                    # Not shown in Jupyter somehow: (make a simple print?)
                    warnings.warn('pinv2 is deprecated, using pinv instead',
                                  DeprecationWarning)
                Inverse_G = spl.pinv(self.Forward_G, **method_kwargs)
                self.Mag = np.matmul(Inverse_G, scan_flatten)  # type: ignore
                if self.verbose: print(SUCC_MSG)
            elif method == 'numpy_pinv':
                Inverse_G = np.linalg.pinv(self.Forward_G, **method_kwargs)
                self.Mag = np.matmul(Inverse_G, scan_flatten)
                if self.verbose:
                    print(SUCC_MSG)

            elif method == 'scipy_lapack':
                # Solve G^t * phi = G^t * G * M
                # where: M -> magnetization ; phi -> scan measurements (1D arr)
                # 1. Get LU decomp for G^t * G
                # 2. Solve the linear equation using the LU dcomp as required
                #    by the dgesrs solver
                GtG = np.matmul(self.Forward_G.T, self.Forward_G)
                GtG_shuffle, IPIV, INFO1 = spl.lapack.dgetrf(GtG)
                if INFO1 == 0:
                    if self.verbose:
                        print('LU decomposition of G * G^t succeeded')
                    GtScan = np.matmul(self.Forward_G.T, scan_flatten)
                    self.Mag, INFO2 = spl.lapack.dgetrs(GtG_shuffle, IPIV, GtScan)
                    if INFO2 != 0:
                        self.Mag = np.empty([])
                        raise RuntimeError(f'{INFO2}th argument has an illegal value. self.Mag deleted')
                    else:
                        if self.verbose:
                            print(SUCC_MSG)
                else:
                    print(f'{INFO1}th argument has an illegal value')

            else:
                raise TypeError(f'Method {method} is not recognized')

            if store_inverse_G_matrix:
                if method == 'scipy_lapack':
                    raise TypeError('LAPACK method does not compute G inverse')
                else:
                    # Warning: Inverse_G might be an unbound variable:
                    self.Inverse_G = Inverse_G

        else:
            raise RuntimeError(f'Problem is underdetermined with '
                               f'{self.Forward_G.shape[0]} knowns and '
                               f'{self.Forward_G.shape[1]} unknowns')

        return None

    def obtain_magnetization(
            self,
            scan_data: Path | str | npt.NDArray[np.float64],
            cuboid_data: Path | str | npt.NDArray[np.float64],
            cuboid_scaling_factor: float,
            method_scan_domain: _SetScanDomainOps = 'sensor_center_domain',
            method_populate: _PrepMatOps = 'cython',
            method_inverse: _MethodOps = 'scipy_pinv',
            **method_inverse_kwargs):
        """Shortcut method to compute the magnetization of the grains

        It calls three methods: `read_files`, `prepare_matrix` and
        `calculate_inverse`

        Parameters
        ----------
        scan_data
            Matrix file or `np.ndarray` `(Nx columns, Ny rows)` containing the
            scan data in T
        cuboid_data
            File or np.ndarray `(x, y, z, dx, dy, dz, index)` containing location
            and size grains in micrometers
        cuboid_scaling_factor
            Scaling factor for the cuboid positions and lengths
        method_scan_domain
            Method to set the scan domain. See: `set_scan_domain`
        method_populate
            Method to populate the forward matrix. See: `populate_matrix`
        method_inverse
            Method to calculate the numerical inversion. See the docstring of
            `calculate_inverse` for details about the method parameters
        """

        self.read_files(scan_data, cuboid_data, cuboid_scaling_factor)
        self.set_scan_domain(gen_sd_mesh_from=method_scan_domain)
        self.prepare_matrix(method=method_populate)
        self.calculate_inverse(method=method_inverse, **method_inverse_kwargs)

    def save_results(self,
                     Magfile: Path or str,
                     keyfile: Path or str,
                     ):
        """
        Saves the magnetization to a specified `Magfile` file and the keys of
        the index of the particles in the key file file.

        Parameters
        ----------
        Magfile
            Path to file to save the magnetization
        keyfile
            Path to file to save the identification (key) of all grains
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

        np.save(Magfile, data)
        np.save(keyfile, p_idxs)

    def forward_field(self,
                      filepath: Optional[str] = None,
                      sigma: Optional[float] = None):

        """ Calculates the forward field

        Parameters
        ----------
        filepath
            Optional path to file to save the forward field
        sigma
            Standard deviation of Gaussian noise to be added in Tesla

        Returns
        -------
        Forward_field
            Optionally return forward magnetic field if no file path is input
        """

        Forward_field = np.matmul(self.Forward_G, self.Mag) / self.scan_area  # mag field
        if sigma is not None:  # add Gaussian noise to the forward field
            error = np.random.normal(0, sigma, len(Forward_field))
            self.sigma = sigma * 4 * self.scan_deltax * self.scan_deltay  # originally it is a flux
            Forward_field = Forward_field + error
        if filepath is not None:
            np.save(filepath, Forward_field.reshape(self.Ny, self.Nx))
        else:
            return Forward_field.reshape(self.Ny, self.Nx)
