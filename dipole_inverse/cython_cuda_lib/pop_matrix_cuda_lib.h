void populate_matrix_cuda(double * G,
                          double * QDM_domain, double scan_height,
                          double * cuboids, 
                          unsigned long long N_cuboids, unsigned long long Npart,
                          unsigned long long Ny, unsigned long long Nx,
                          double QDM_spacing,
                          double QDM_deltax, double QDM_deltay,
                          int Origin, int verbose
                          );
