cimport numpy as np

# -----------------------------------------------------------------------------

cdef extern from "pop_matrix_C_lib.h":

    void populate_matrix_C(double * G,
                           double * scan_domain, double scan_height,
                           double * cuboids,
                           unsigned long long N_cuboids, unsigned long long Npart,
                           unsigned long long Ny, unsigned long long Nx, 
                           double scan_spacing_x, double scan_spacing_y,
                           double scan_deltax, double scan_deltay,
                           int Origin, int verbose
                           )

# -----------------------------------------------------------------------------

def populate_matrix_cython(double [:, :] G,
                           double [:] scan_domain,
                           double scan_height,
                           double [:] cuboids, 
                           unsigned long long N_cuboids,
                           unsigned long long Npart, 
                           unsigned long long Ny, unsigned long long Nx,
                           double scan_spacing_x, double scan_spacing_y,
                           double scan_deltax, double scan_deltay,
                           int Origin, int verbose):

    # I guess G is passed as a column-order (C) 1D array to the C code
    populate_matrix_C(&G[0, 0],
                      &scan_domain[0], scan_height,
                      &cuboids[0], N_cuboids, Npart,
                      Ny, Nx, scan_spacing_x, scan_spacing_y,
                      scan_deltax, scan_deltay,
                      Origin, verbose
                      )
