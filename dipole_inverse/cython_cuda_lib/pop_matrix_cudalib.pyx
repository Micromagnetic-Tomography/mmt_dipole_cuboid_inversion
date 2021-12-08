cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free 

# -----------------------------------------------------------------------------

cdef extern from "pop_matrix_cuda_lib.h":

    void populate_matrix_cuda(double ** G,
                              double * QDM_domain, double scan_height,
                              double * cuboids,
                              unsigned long long N_cuboids, unsigned long long Npart,
                              unsigned long long Ny, unsigned long long Nx, 
                              double QDM_spacing,
                              double QDM_deltax, double QDM_deltay,
                              int Origin, int verbose
                              )

# -----------------------------------------------------------------------------

def populate_matrix_cython(double [:, :] G,
                           double [:] QDM_domain,
                           double scan_height,
                           double [:] cuboids, 
                           unsigned long long N_cuboids,
                           unsigned long long Npart, 
                           unsigned long long Ny, unsigned long long Nx,
                           double QDM_spacing,
                           double QDM_deltax, double QDM_deltay,
                           int Origin, int verbose):

    # cdef unsigned long long G_rows = Nx * Ny
    # cdef unsigned long long G_rows = Nx * Ny

    # Make sure the array a has the correct memory layout (here C-order)
    # cdef cnp.ndarray[double, ndim=2, mode="c"] G_cython = np.asarray(G, dtype=float, order="C")
    # cdef double[:,::1] G_view = np.ascontiguousarray(G, dtype=np.double)

    # cdef double** point_to_G = <double **>malloc(G_rows * sizeof(double*))
    cdef double** point_to_G = <double **> PyMem_Malloc(G.shape[0] * sizeof(double*))
    if not point_to_G: raise MemoryError

    try:
        for i in range(G.shape[0]): 
            point_to_G[i] = &G[i, 0]

        # Call the C function that expects a double**
        # G MUST be passed as a column-order (C) 2D array to the C code
        populate_matrix_cuda(&point_to_G[0],
                             &QDM_domain[0], scan_height,
                             &cuboids[0], N_cuboids, Npart,
                             Ny, Nx, QDM_spacing,
                             QDM_deltax, QDM_deltay,
                             Origin, verbose
                             )
    finally:
        # free(point_to_G)
        if point_to_G is not NULL:
            PyMem_Free(point_to_G)
