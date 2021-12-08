#include "pop_matrix_cuda_lib.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>


__global__ void pop_matrix_nv(double * G, double * cuboids, 
                              unsigned long long N_cuboids, 
                              unsigned long long Nx, unsigned long long Ny,
                              double QDM_deltax, double QDM_deltay, 
                              double xi0, double eta0, double zeta0) {

    // The thread's unique number 
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    int threadsInBlock = blockDim.x * blockDim.y;
    int globThread = threadsInBlock * (gridDim.x * blockIdx.y + blockIdx.x)
                     + (blockDim.x * threadIdx.y + threadIdx.x);

    for (unsigned long long sx = idx_x; sx < Nx; sx += stride_x) {
        for (unsigned long long sy = idx_y; sy < Ny; sy += stride_y) {

        printf("Glob Thread: %i \n", globThread);

        for(int i = idx; i < size; i += stride) {
            res[i] = a[i] + b[i];
        }
        // printf("Stop\n");
        

        // Loop over sensor measurements. Each sensor is in the xy
        // plane and has area delta^2
        // #pragma omp parallel for lastprivate(i_cuboid, i_particle) shared(i_cuboid_old, i_particle_prev)
        // for (unsigned long long n = 0; n < Nx * Ny; n++) {

        unsigned long long i_cuboid;
        unsigned long long i_cuboid_old;
        unsigned long long i_particle_prev;
        unsigned long long i_particle;

        // If grains are not numbered in order this always works
        int i_particle_0_N = 0;

        // Set scan positions in x and y direction
        unsigned long long i = n % Nx;
        unsigned long long j = n / Nx;

        // Definitions
        double x, y, z, x2, y2, z2, sign, r2, r, Az, Lx, Ly, F120, F210, F22m;
        double particle_flux[3] = {0};
        double get_flux[3]      = {0};
        double cuboid_center[3] = {0};
        double dr_cuboid[3]     = {0};
        double cuboid_size[3]   = {0};

        double sensor_pos[3] = {0};
        sensor_pos[2] = zeta0;
        sensor_pos[1] = eta0 + QDM_spacing * sy;
        sensor_pos[0] = xi0 + QDM_spacing * sx;

        i_cuboid = 0;
        i_cuboid_old = 0;
        i_particle_prev = (int) cuboids[6];
        i_particle = i_particle_prev;
        i_particle_0_N = 0;

        while (i_cuboid < N_cuboids) {
            if(verbose == 1) {
                printf("Particle = %lld   Cuboid = %lld i %lld j %lld\n", i_particle, i_cuboid, i, j);
            }
            // i_cuboid_old = i_cuboid;

            // The contribution of the flux for mx, my, mz
            for (int k = 0; k < 3; k++) particle_flux[k] = 0.0;

            // Start from the index of the particle being analysed
            // i_particle = (unsigned long long) cuboids[7 * i_cuboid_old + 6];
            // i_cuboid = i_cuboid_old;

            // While the cuboid has particle index of the
            // particle being analysed
            while (i_particle == i_particle_prev) {

                for (int k = 0; k < 3; k++) {
                    cuboid_center[k] = cuboids[7 * i_cuboid + k];
                    dr_cuboid[k] = cuboid_center[k] - sensor_pos[k];
                    cuboid_size[k] = cuboids[7 * i_cuboid + (k + 3)];
                }
                // Cuboid sizes:

                // calculate flux per cuboid
                for (int k = 0; k < 3; k++) get_flux[k] = 0.0;

                for (double s1 = -1; s1 < 1.1;  s1 += 2) {
                    for (double s2 = -1; s2 < 1.1;  s2 += 2) {
                        for (double s3 = -1; s3 < 1.1;  s3 += 2) {
                            for (double s4 = -1; s4 < 1.1;  s4 += 2) {
                                for (double s5 = -1; s5 < 1.1;  s5 += 2) {
                                    x = dr_cuboid[0] + s1 * cuboid_size[0] - s4 * QDM_deltax;
                                    y = dr_cuboid[1] + s2 * cuboid_size[1] - s5 * QDM_deltay;
                                    z = dr_cuboid[2] + s3 * cuboid_size[2];
                                    sign = s1 * s2 * s3 * s4 * s5;
                                    x2 = x * x; y2 = y * y; z2 = z * z;
                                    r2 = x2 + y2 + z2;
                                    r = sqrt(r2);
                                    Az = atan2(x * y, z * r);
                                    if (r != 0.0) {
                                        Lx = log(x + r);
                                        Ly = log(y + r);
                                    } else {
                                        Lx = Ly = 0.0;
                                        printf("Error at p = %lld", i_particle);
                                    }

                                    F120 = 0.5 * ((y2 - z2) * Lx - r * x) - y * (z * Az - x * Ly);
                                    F210 = 0.5 * ((x2 - z2) * Ly - r * y) - x * (z * Az - y * Lx);
                                    F22m = -x * y * Az - z * (x * Lx + y * Ly - r);

                                    get_flux[0] += sign * F120;
                                    get_flux[1] += sign * F210;
                                    get_flux[2] += sign * F22m;
                                } // s1
                            } // s2
                        } // s3
                    } // s4
                } // s5

                // Finish cuboidsloop in the particle i_particle_prev
                // and continue with the next sensor measurement

                // scale flux measurement:
                for (int k = 0; k < 3; k++) {
                    particle_flux[k] += -Cm * get_flux[k];
                }
                i_cuboid += 1;
                i_particle = (unsigned long long) cuboids[7 * i_cuboid + 6];

            }  // end while cuboids in i_particle

            // printf("%d %d", i + j * Nx, 3 * i_particle_prev);
            // Trying to populate G row wise:
            // printf("Pop: %lld\n", Nx * Ny * (3 * i_particle_0_N    ) + i + Nx * j);
            // printf("Part 0 N: %lld\n", i_particle_0_N);
            // printf("i_cuboid: %lld\n", i_cuboid);
            // printf("N_cuboids: %lld\n", N_cuboids);
            // G[sx + sy * Nx][3 * i_particle_0_N    ] = particle_flux[0];
            // G[sx + sy * Nx][3 * i_particle_0_N + 1] = particle_flux[1];
            // G[sx + sy * Nx][3 * i_particle_0_N + 2] = particle_flux[2];
            int sensor_idx = Nx * sy + sx;
            G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N    )] = particle_flux[0];
            G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N + 1)] = particle_flux[1];
            G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N + 2)] = particle_flux[2];

            // Transposed:
            // G[Nx * Ny * (3 * i_particle_0_N    ) + i + Nx * j] = particle_flux[0];
            // G[Nx * Ny * (3 * i_particle_0_N + 1) + i + Nx * j] = particle_flux[1];
            // G[Nx * Ny * (3 * i_particle_0_N + 2) + i + Nx * j] = particle_flux[2];

            // Move to next particle (last i_particle after ending while loop)
            i_particle_prev = i_particle;
            // Update particle number indexed from 0 to N
            i_particle_0_N += 1;

        } // end while i_cuboids < N_cuboids
    } // end sensor loop
}

/*
Main loop to populate the G matrix The outer while loop will last until
reaching the total number of cuboids in the sample. Adjacent cuboids belong to
a single particle, which is indexed in the 6th element of the cuboids array.
The population of the G matrix is performed column wise for every particle. For
each cuboid belonging to a particle, their contribution to the magnetic flux is
summed up for every sensor measurement in steps of delta in the xy plane, which
are given by the loops with the i-j indexes.  The flux is stored column wise.
If Origin is True (default), the cuboids are stored with their original
coordinates. If cuboids are shifted, Origin is False.
*/

// G matrix     -> 1D array that comes from the Python array: (N_parts, Nx * Ny)
//                 So it;s the transposed version of original G
// QDM_domain   -> array with 4 entries x1 y1 x2 y2
// cuboids      -> N_part * 6 array
void populate_matrix_cuda(double * G,
                          double * QDM_domain, double scan_height,
                          double * cuboids,
                          unsigned long long N_cuboids, unsigned long long Npart,
                          unsigned long long Ny, unsigned long long Nx,
                          double QDM_spacing,
                          double QDM_deltax, double QDM_deltay,
                          int Origin, int verbose
                          ) {

    double Cm = 1e-7;

    double xi0, eta0, zeta0;
    if (Origin == 1) {
        xi0 = QDM_domain[0];
        eta0 = QDM_domain[1];
        zeta0 = (-1) * scan_height;
    } else {
        xi0 = 0.0;
        eta0 = 0.0;
        zeta0 = (-1) * scan_height;
    }

    
    // Manual mem allocation:
    double *G_dev;
    cudaMalloc((void**)&G_dev, sizeof(double) * Nx * Ny);

    // Launch kernel
    // int size     = matrix_size;
    // Quadro RTX 6000: 4608 CUDA Cores
    int n_threads = 16; // 256 threads per block in 2D
    // Determine based on problem size: Nx * Ny (easy for a matrix)
    int n_blocks_x = (Nx / n_threads) + ((Nx % n_threads) != 0);
    int n_blocks_y = (Ny / n_threads) + ((Ny % n_threads) != 0);
    dim3 grid(n_blocks_x, n_blocks_y);
    dim3 block(n_threads, n_threads);

    pop_matrix_nv<<<grid, block>>>(** G_dev, double * cuboids_dev, 
                                   N_cuboids, Nx, Ny, QDM_deltax, QDM_deltay, 
                                   xi0, eta0, zeta0);

} // main function
