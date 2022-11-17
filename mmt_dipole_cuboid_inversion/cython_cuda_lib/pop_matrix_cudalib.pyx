cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free 

# -----------------------------------------------------------------------------

cdef extern from "pop_matrix_cuda_lib.h":

    void populate_matrix_cuda(double * G,
                              double * scan_domain, double scan_height,
                              double * cuboids,
                              unsigned long long N_cuboids, unsigned long long Npart,
                              unsigned long long Ny, unsigned long long Nx, 
                              double scan_spacing_x, double scan_spacing_y,
                              double scan_deltax, double scan_deltay,
                              int Origin, int verbose
                              )

# -----------------------------------------------------------------------------

def populate_matrix(double [:, :] G,
                    double [:] scan_domain,
                    double scan_height,
                    double [:] cuboids, 
                    unsigned long long N_cuboids,
                    unsigned long long Npart, 
                    unsigned long long Ny, unsigned long long Nx,
                    double scan_spacing_x, double scan_spacing_y,
                    double scan_deltax, double scan_deltay,
                    int Origin, int verbose):

    # Call the C function
    populate_matrix_cuda(&G[0, 0],
                         &scan_domain[0], scan_height,
                         &cuboids[0], N_cuboids, Npart,
                         Ny, Nx, scan_spacing_x, scan_spacing_y,
                         scan_deltax, scan_deltay,
                         Origin, verbose
                         )
