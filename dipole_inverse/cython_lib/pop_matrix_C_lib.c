#include "pop_matrix_C_lib.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>

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
void populate_matrix_C(double * G,
                       double * QDM_domain, double scan_height,
                       double * cuboids, int N_cuboids, int Npart,
                       int Ny, int Nx, double QDM_spacing,
                       double QDM_deltax, double QDM_deltay,
                       int Origin
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

    int i_cuboid = 0;
    int i_cuboid_old = 0;
    int i_particle_prev = (int) cuboids[6];
    int i_particle = i_particle_prev;


    // print('max cub =', Npart)
    // print('G matrix', G.shape)

    while (i_cuboid < N_cuboids) {
        printf("Particle = %d   Cuboid = %d\n", i_particle, i_cuboid);
        // print(particle =)
        i_cuboid_old = i_cuboid;

        // Loop over sensor measurements. Each sensor is in the xy
        // plane and has area delta^2
        #pragma omp parallel for lastprivate(i_cuboid, i_particle) shared(i_cuboid_old, i_particle_prev)
        for (int n = 0; n < Nx * Ny; n++) {

            // Definitions
            double x, y, z, x2, y2, z2, sign, r2, r, Az, Lx, Ly, F120, F210, F22m;
            double particle_flux[3] = {0};
            double get_flux[3]      = {0};
            double cuboid_center[3] = {0};
            double dr_cuboid[3]     = {0};
            double cuboid_size[3]   = {0};

            // Set scan positions in x and y direction
            int i = n % Nx;
            int j = n / Nx;

            double sensor_pos[3] = {0};
            sensor_pos[2] = zeta0;
            sensor_pos[1] = eta0 + QDM_spacing * j;
            sensor_pos[0] = xi0 + QDM_spacing * i;

            // The contribution of the flux for mx, my, mz
            for (int k = 0; k < 3; k++) particle_flux[k] = 0.0;

            // Start from the index of the particle being analysed
            i_particle = (int) cuboids[7 * i_cuboid_old + 6];
            i_cuboid = i_cuboid_old;

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
                                        printf("Error at p = %d", i_particle);
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
                i_particle = (int) cuboids[7 * i_cuboid + 6];

            }  // end while cuboids in i_particle

            // printf("%d %d", i + j * Nx, 3 * i_particle_prev);
            // Trying to populate G row wise:
            G[Nx * Ny * (3 * (i_particle_prev - 1)    ) + i + Nx * j] = particle_flux[0];
            G[Nx * Ny * (3 * (i_particle_prev - 1) + 1) + i + Nx * j] = particle_flux[1];
            G[Nx * Ny * (3 * (i_particle_prev - 1) + 2) + i + Nx * j] = particle_flux[2];
            // DO NOT forget to transpose G after calling this function!
            //
            // OLD CODE:
            // ------Populate G matrix column wise
            // G[i + j * Nx, 3 * (i_particle_prev - 1)]     = particle_flux[0];
            // G[i + j * Nx, 3 * (i_particle_prev - 1) + 1] = particle_flux[1];
            // G[i + j * Nx, 3 * (i_particle_prev - 1) + 2] = particle_flux[2];

        } // end for n

        i_particle_prev = i_particle;
    } // end while total cuboids
} // main function
