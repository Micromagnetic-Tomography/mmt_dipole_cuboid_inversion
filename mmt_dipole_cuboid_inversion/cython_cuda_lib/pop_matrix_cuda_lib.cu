#include "pop_matrix_cuda_lib.h"
#include <math.h>
#include <stdio.h>


__global__ void pop_matrix_nv(double * G, double * cuboids, 
                              unsigned long long N_cuboids, 
                              unsigned long long Nx, unsigned long long Ny, unsigned long long Npart,
                              double scan_deltax, double scan_deltay,
                              double scan_spacing_x, double scan_spacing_y,
                              double xi0, double eta0, double zeta0, double RefMult,
                              int verbose) {

    double Cm = 1e-7;

    // The thread's unique number 
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    int threadsInBlock = blockDim.x * blockDim.y;
    int globThread = threadsInBlock * (gridDim.x * blockIdx.y + blockIdx.x)
                     + (blockDim.x * threadIdx.y + threadIdx.x);


    // Loop over sensor measurements. Each sensor is in the xy plane and has area delta^2
    for (unsigned long long sy = idx_y; sy < Ny; sy += stride_y) {
        for (unsigned long long sx = idx_x; sx < Nx; sx += stride_x) {

            // if (sy == idx_y && sx == idx_x) { 
            //     printf("GlobThread : %d sx %ld sy %ld\n", globThread, sx, sy);
            // }

            unsigned long long i_cuboid;
            // unsigned long long i_cuboid_old;
            unsigned long long i_particle_prev;
            unsigned long long i_particle;

            // If grains are not numbered in order this always works
            int i_particle_0_N = 0;

            // Set scan positions in x and y direction
            // unsigned long long i = n % Nx;
            // unsigned long long j = n / Nx;

            // Definitions
            double x, y, z, x2, y2, z2, sign, r2, r, Az, Lx, Ly, F120, F210, F22m;
            double particle_flux[3] = {0};
            double get_flux[3]      = {0};
            double cuboid_center[3] = {0};
            double dr_cuboid[3]     = {0};
            double cuboid_size[3]   = {0};

            double sensor_pos[3] = {0};
            sensor_pos[2] = zeta0;
            sensor_pos[1] = eta0 + scan_spacing_y * sy;
            sensor_pos[0] = xi0 + scan_spacing_x * sx;

            i_cuboid = 0;
            // i_cuboid_old = 0;
            i_particle_prev = (int) cuboids[6];
            i_particle = i_particle_prev;
            i_particle_0_N = 0;

            while (i_cuboid < N_cuboids) {
                if(verbose == 1) {
                    printf("Particle = %lld   Cuboid = %lld i %lld j %lld\n", i_particle, i_cuboid, sx, sy);
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
                                        x = RefMult * (dr_cuboid[0] + s1 * cuboid_size[0] - s4 * scan_deltax);
                                        y = RefMult * (dr_cuboid[1] + s2 * cuboid_size[1] - s5 * scan_deltay);
                                        z = RefMult * (dr_cuboid[2] + s3 * cuboid_size[2]);
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
                    for (int k = 0; k < 3; k++) particle_flux[k] += -Cm * get_flux[k];
                    i_cuboid += 1;
                    i_particle = (unsigned long long) cuboids[7 * i_cuboid + 6];

                }  // end while cuboids in i_particle

                // printf("Pop: %lld\n", Nx * Ny * (3 * i_particle_0_N    ) + i + Nx * j);
                // printf("Part 0 N: %lld\n", i_particle_0_N);
                // G[sx + sy * Nx][3 * i_particle_0_N    ] = particle_flux[0];
                // G[sx + sy * Nx][3 * i_particle_0_N + 1] = particle_flux[1];
                // G[sx + sy * Nx][3 * i_particle_0_N + 2] = particle_flux[2];

                // Trying to populate G row wise:
                int sensor_idx = Nx * sy + sx;
                G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N    )] = particle_flux[0];
                G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N + 1)] = particle_flux[1];
                G[(3 * Npart) * sensor_idx + (3 * i_particle_0_N + 2)] = particle_flux[2];

                // Move to next particle (last i_particle after ending while loop)
                i_particle_prev = i_particle;
                // Update particle number indexed from 0 to N
                i_particle_0_N += 1;

            } // end while i_cuboids < N_cuboids
        } // end sensor loop sx
    } // end sensor loop sy
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

// G matrix     -> 1D array that comes from the Python array: (Nx * Ny, 3 * N_parts)
// scan_domain   -> array with 4 entries x1 y1 x2 y2
// cuboids      -> N_cuboids * 7 array
void populate_matrix_cuda(double * G,
                          double * scan_domain, double scan_height,
                          double * cuboids,
                          unsigned long long N_cuboids, unsigned long long Npart,
                          unsigned long long Ny, unsigned long long Nx,
                          double scan_spacing,
                          double scan_deltax, double scan_deltay,
                          int Origin, int verbose
                          ) {


    double RefMult = (scan_height < 0) ? -1.0 : 1.0;
    double xi0, eta0, zeta0;
    if (Origin == 1) {
        xi0 = scan_domain[0];
        eta0 = scan_domain[1];
    } else {
        xi0 = 0.0;
        eta0 = 0.0;
    }
    zeta0 = scan_height;

    size_t G_bytes = sizeof(double) * Nx * Ny * 3 * Npart;
    size_t cuboids_bytes = sizeof(double) * 7 * N_cuboids;

    // Manual mem allocation: G in GPU and cuboids_dev in GPU
    double *G_dev;
    // CUDA_ASSERT(cudaMalloc((void**)&G_dev, G_bytes));
    // (allocate in GPU if enough memory, see below)
    // cudaMalloc((void**)&G_dev, G_bytes);

    double *cuboids_dev;
    cudaMalloc((void**)&cuboids_dev, cuboids_bytes);
    // Copy cuboids array from the host to the GPU
    cudaMemcpy(cuboids_dev, cuboids, cuboids_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    // Quadro RTX 6000: 4608 CUDA Cores
    // More refined matrix allocation of blocks if we use smaller n_threads, e.g. 8
    int n_threads = 16; // 256 threads per block in 2D
    // Determine blocks and grid based on problem size: Nx * Ny (easy for a matrix)
    int n_blocks_x = (Nx / n_threads) + ((Nx % n_threads) != 0);
    int n_blocks_y = (Ny / n_threads) + ((Ny % n_threads) != 0);
    dim3 grid(n_blocks_x, n_blocks_y);
    dim3 block(n_threads, n_threads);

    // Checking available memory in GPU:
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double) free_byte / (1024. * 1024.);
    // Quadro RTX 6000: total mem should be 24220.3125 Mb
    double total_db = (double) total_byte / (1024. * 1024.);
    double used_db = total_db - free_db;
    double G_size_mb = (double) G_bytes / (1024. * 1024.);
    double cuboids_size_mb = (double) cuboids_bytes / (1024. * 1024.);

    // if(verbose == 0) {
    printf("------------ Nvidia GPU calculation info ------------\n");
    printf("GPU Memory      (MB): free  = %.4f\n", free_db);
    printf("                      used  = %.4f\n", used_db);
    printf("                      total = %.4f\n", total_db);
    printf("Size of G       (MB): %.4f\n", G_size_mb);
    printf("Size of cuboids (MB): %.4f\n", cuboids_size_mb);
    printf("Blocks grid = %d x %d\n", n_blocks_x, n_blocks_y);
    printf("Threads per block = %d x %d\n", n_threads, n_threads);
    printf("Sensor Matrix dims (rows x cols) = %d x %d\n", Ny, Nx);
    // }

    // Quadro RTX 6000: total mem should be 24220.3125 Mb
    // double MEM_THRESHOLD = 22000;

    // Calculate if there's enough memory in card
    if ((G_size_mb + cuboids_size_mb) > free_db) {

        printf("Not enough available memory in GPU\n");
        // printf("Calculation of large sensor matrices will be implemented in the future\n");
        printf("Stopping calculation\n");
        return;

        // -- WIP --
        // Estimate an optimal size for the sub-matrix. We will use a square m
        // unsigned int N = sqrt((free_db - cuboids_size_mb) / (3 * Npart));
        // G_bytes = sizeof(double) * N * N * 3 * Npart;
        // cudaMalloc((void**)&G_dev, G_bytes);

        // TODO:
        // - Use a for loop to populate the submatrix G_dev in device, using strides of
        //   size N in x-direction and y-direction
        //   + For this we have to find out a way to register the i,j sites that were
        //     calculated in the device in order to copy these sites into the
        //     corresponding sites of the full G matrix in the host device
        // - The pop_matrix_nv function in device might need to be modified in order
        //   to specify the i0,j0 pairs where the submatrix is located (starting point
        //   for populating G)
        //   + It might be easy if we just add i0,j0 to the indices sx,sy in the for loop
        //     of pop_matrix_nv
        // - To populate the full G matrix, we can use cudaMemcpy with (needs checking)
        //   &G_dev[i0 + N * j0] , to start pop from a different loc in the G array

    }
    else {
        // Allocate G matrix
        cudaMalloc((void**)&G_dev, G_bytes);

        // Populate matrix in GPU:
        pop_matrix_nv<<<grid, block>>>(G_dev, cuboids_dev, 
                                       N_cuboids, Nx, Ny, Npart,
                                       scan_deltax, scan_deltay, scan_spacing,
                                       xi0, eta0, zeta0, RefMult, verbose);
        cudaDeviceSynchronize();

        // Copy G from the GPU to the host
        cudaMemcpy(G, G_dev, G_bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(G_dev);
    cudaFree(cuboids_dev);

} // main function
