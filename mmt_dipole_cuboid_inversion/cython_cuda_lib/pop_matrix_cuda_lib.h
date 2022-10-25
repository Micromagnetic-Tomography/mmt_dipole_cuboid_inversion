void populate_matrix_cuda(double * G,
                          double * scan_domain, double scan_height,
                          double * cuboids, 
                          unsigned long long N_cuboids, unsigned long long Npart,
                          unsigned long long Ny, unsigned long long Nx,
                          double scan_spacing_x, double scan_spacing_y,
                          double scan_deltax, double scan_deltay,
                          int Origin, int verbose
                          );
