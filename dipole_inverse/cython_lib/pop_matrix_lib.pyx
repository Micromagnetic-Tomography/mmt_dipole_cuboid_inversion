cimport numpy as np

# -----------------------------------------------------------------------------

cdef extern from "pop_matrix_C_lib.h":

    void populate_matrix_C(double * G,
                           double * QDM_domain, double scan_height,
                           double * cuboids, int N_cuboids, int Npart,
                           int Ny, int Nx, double QDM_spacing,
                           double QDM_deltax, double QDM_deltay,
                           int Origin
                           )

# -----------------------------------------------------------------------------

def populate_matrix_cython(double [:, :] G,
                           double [:] QDM_domain,
                           double scan_height,
                           double [:] cuboids, int N_cuboids,
                           int Npart, int Ny, int Nx, double QDM_spacing,
                           double QDM_deltax, double QDM_deltay,
                           int Origin):

    # I guess G is passed as a column-order (C) 1D array to the C code
    populate_matrix_C(&G[0, 0],
                      &QDM_domain[0], scan_height,
                      &cuboids[0], N_cuboids, Npart,
                      Ny, Nx, QDM_spacing,
                      QDM_deltax, QDM_deltay,
                      Origin
                      )
