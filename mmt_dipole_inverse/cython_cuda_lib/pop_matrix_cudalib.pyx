cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free 

# -----------------------------------------------------------------------------

cdef extern from "pop_matrix_cuda_lib.h":

    void populate_matrix_cuda(double * G,
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

    # Call the C function
    populate_matrix_cuda(&G[0, 0],
                         &QDM_domain[0], scan_height,
                         &cuboids[0], N_cuboids, Npart,
                         Ny, Nx, QDM_spacing,
                         QDM_deltax, QDM_deltay,
                         Origin, verbose
                         )
