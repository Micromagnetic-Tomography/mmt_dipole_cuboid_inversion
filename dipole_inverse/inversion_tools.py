import numpy as np
from . import Dipole
from typing import Literal   # Working with Python >3.8
from typing import Union     # Working with Python >3.8
from typing import Tuple     # Working with Python >3.8
from typing import Optional  # Working with Python >3.8


def calculate_covariance_matrix(DipoleClassInstance: Dipole,
                                sigma: float,
                                norm: bool = True,
                                std_dev_file: str = None,
                                norm_covar_file: str = None,
                                resol_matrix_file: str = None
                                ) -> Optional[Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]]:
    r"""
    Calculates the covariance matrix

    Parameters
    ----------
    DipoleClassInstance
        An instance of the Dipole class that was used to calculate the
        inversion matrices
    sigma
        The standard deviation of the error of the magnetic field
    norm
        normalize the covariance matrix (True) or not (False)
    std_dev_file
        File path to where the standard deviation is written
    norm_covar_file
        File path to where the normalized covariance matrix is written
    resol_matrix_file
        File path to where the resolution matrix is written

    Returns
    -------
    standard_deviation
    normalized_covariance_matrix
    resolution_matrix
    """

    self = DipoleClassInstance

    if self.Inverse_G is None:
        print('This method requires calling calculate_inverse with the '
              'store_inverse_G_matrix=True option. Stopping calculation.')
        return None

    covar = (sigma ** 2) * (self.Inverse_G @ self.Inverse_G.transpose())

    standard_deviation = np.sqrt(np.diag(covar)).reshape(self.Npart, 3)

    if norm == True:
        normcovar = covar.copy()
        for row in range(covar.shape[0]):
            for column in range(covar.shape[1]):
                normcovar[row, column] = covar[row, column]
                normcovar[row, column] /= np.sqrt(covar[row, row] * covar[column, column])
    else:
        normcovar = covar

    resolution_matrix = self.Inverse_G @ self.Forward_G

    if std_dev_file is not None:
        np.save(std_dev_file, standard_deviation)
    if norm_covar_file is not None:
        np.save(norm_covar_file, normcovar)
    if resol_matrix_file is not None:
        np.save(resol_matrix_file, resolution_matrix)

    return (standard_deviation, normcovar, resolution_matrix)


_normOps = Literal[None, 1, -1, 2, -2, 'inf', '-inf', 'fro']


def calculate_condition_number(DipoleClassInstance: Dipole,
                               matrix_norm: _normOps = None
                               ) -> Union[None, float]:
    r"""
    Returns the condition number of the forward matrix. The cond number is
    defined as the product of the matrix norms of the forward matrix and
    its pseudo inverse: :math:`|Q| * |Q^\dagger|`.

    Parameters
    ----------
    matrix_norm
        The kind of matrix `norm` to be used, which is determined by the
        `ord` parameter in `numpy.linalg.norm`.  For instance,
        return_pinv_and_cnumber='fro'.  The Numpy's `inf` values in this
        case are replaced by strings. Notice that the condition number will
        be determined by the cutoff value for the singular values of the
        forward matrix.
    """

    self = DipoleClassInstance

    if self.Inverse_G is None:
        print('This method requires calling calculate_inverse with the '
              'store_inverse_G_matrix=True option. Stopping calculation.')
        return None

    if matrix_norm in ('inf', '-inf'):
        norm_order = np.inf if matrix_norm == 'inf' else -np.inf
    else:
        norm_order = matrix_norm
    cond_number = np.linalg.cond(self.Forward_G, p=norm_order)

    return cond_number


def calculate_forward_field(DipoleClassInstance: Dipole,
                            sigma: Union[None, float] = None,
                            forward_field_file: Optional[str] = None,
                            ) -> np.ndarray:

    """
    Calculates the forward field from the inverted magnetisation. If a standard
    deviation value is passed a Gaussian noise is estimated and added to the
    field.

    Parameters
    ----------
    DipoleClassInstance
        An instance of the Dipole class that was used to calculate the
        inversion matrices
    sigma
        Standard deviation of Gaussian noise to be added in T.
    forward_field_file
        Path to file to save the inverse forward field
    """
    self = DipoleClassInstance

    # Approxim Magnetic field from the inversion
    Forward_field = np.matmul(self.Forward_G, self.Mag) / self.QDM_area

    if sigma is not None:  # add Gaussian noise to the forward field
        error = np.random.normal(0, sigma, len(Forward_field))
        # originally it is a flux (we dont store sigma for now)
        # self.sigma = sigma * 4 * self.QDM_deltax * self.QDM_deltay
        Forward_field = Forward_field + error

    if forward_field_file is not None:
        np.save(forward_field_file,
                Forward_field.reshape(self.Ny, self.Nx))

    return Forward_field
