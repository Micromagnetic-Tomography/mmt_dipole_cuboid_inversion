import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.linalg.lapack import dgetrs
from scipy.linalg.lapack import dgetrf
from scipy.linalg import pinv
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.ops import cascaded_union

@nb.jit(nopython=True)
def populate_matrix(G, QDM_domain, scan_height, cuboids, Npart,
                    Ny, Nx, QDM_spacing, QDM_deltax, QDM_deltay,
                    Origin):
    """ Modified version of David's function
    Main loop to populate the G matrix
    The outer while loop will last until reaching the total number
    of cuboids in the sample. Adjacent cuboids belong to a single
    particle, which is indexed in the 6th element of the
    cuboids array. The population of the G matrix is
    performed column wise for every particle. For each cuboid
    belonging to a particle, their contribution to the magnetic
    flux is summed up for every sensor measurement in steps of
    delta in the xy plane, which are given by the loops with the
    i-j indexes. The flux is stored column wise.

    If Origin is True (default), the cuboids are stored with their
    original coordinates. If cuboids are shifted, Origin is False.
    """

    Cm = 1e-7
    if Origin is True:
        xi0, eta0 = QDM_domain[0, :]
        zeta0 = -1 * scan_height
    else:
        xi0, eta0, zeta0 = 0., 0., (-1) * scan_height
    sensor_pos = np.zeros(3)
    sensor_pos[2] = zeta0

    # Definitions
    particle_flux = np.zeros(3)
    get_flux = np.zeros(3)
    cuboid_center = np.zeros(3)
    dr_cuboid = np.zeros(3)
    cuboid_size = np.zeros(3)

    i_cuboid = 0
    i_particle_prev = int(cuboids[0, 6])
    i_particle = i_particle_prev

    # print('max cub =', Npart)
    # print('G matrix', G.shape)

    while i_cuboid < len(cuboids):
        # print(f'Particle = {i_particle}  Cuboid = {i_cuboid}')
        print(f'Particle =', i_particle,  'Cuboid =', i_cuboid)
        # print(particle =)
        i_cuboid_old = i_cuboid

        # Loop over sensor measurements. Each sensor is in the xy
        # plane and has area delta^2
        for j in range(Ny):
            sensor_pos[1] = eta0 + QDM_spacing * j
            for i in range(Nx):
                sensor_pos[0] = xi0 + QDM_spacing * i

                # The contribution of the flux for mx, my, mz
                particle_flux[:] = 0

                # Start from the index of the particle being analysed
                i_particle = int(cuboids[i_cuboid_old, 6])
                i_cuboid = i_cuboid_old

                # While the cuboid has particle index of the
                # particle being analysed
                while i_particle == i_particle_prev:
#                     print(i_particle, i, j, i_cuboid)
                    cuboid_center[:] = cuboids[i_cuboid, :3]
                    dr_cuboid[:] = cuboid_center - sensor_pos
                    # Cuboid sizes:
                    cuboid_size[:] = cuboids[i_cuboid, 3:6]

                    # calculate flux per cuboid
                    get_flux[:] = 0.
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            for s3 in [-1, 1]:
                                for s4 in [-1, 1]:
                                    for s5 in [-1, 1]:
                                        x = dr_cuboid[0] + s1 * cuboid_size[0] - s4 * QDM_deltax
                                        y = dr_cuboid[1] + s2 * cuboid_size[1] - s5 * QDM_deltay
                                        z = dr_cuboid[2] + s3 * cuboid_size[2]
                                        sign = s1 * s2 * s3 * s4 * s5
                                        x2, y2, z2 = x ** 2, y ** 2, z ** 2
                                        r2 = x2 + y2 + z2
                                        r = np.sqrt(r2)
                                        Az = np.arctan2(x * y, z * r)
                                        if r != 0.0:
                                            Lx = np.log(x + r)
                                            Ly = np.log(y + r)
                                        else:
                                            Lx, Ly = 0., 0.
                                            print('Error at p = ', i_particle)

                                        F120 = 0.5 * ((y2 - z2) * Lx - r * x) - y * (z * Az - x * Ly)
                                        F210 = 0.5 * ((x2 - z2) * Ly - r * y) - x * (z * Az - y * Lx)
                                        F22m = -x * y * Az - z * (x * Lx + y * Ly - r)

                                        get_flux[0] += sign * F120
                                        get_flux[1] += sign * F210
                                        get_flux[2] += sign * F22m

                    # Finish cuboidsloop in the particle i_particle_prev
                    # and continue with the next sensor measurement

                    # scale flux measurement:
                    particle_flux[:] += -Cm * get_flux
                    i_cuboid += 1
                    i_particle = int(cuboids[i_cuboid, 6])

                # print(i + j * Nx, 3 * i_particle_prev)
                # Populate G matrix column wise
                G[i + j * Nx, 3 * (i_particle_prev - 1)] = particle_flux[0]
                G[i + j * Nx, 3 * (i_particle_prev - 1) + 1] = particle_flux[1]
                G[i + j * Nx, 3 * (i_particle_prev - 1) + 2] = particle_flux[2]

        i_particle_prev = i_particle
    return G

class Dipole(object):
    """ This class calculates and plots magnetization
    """

    def __init__(self, QDM_data, cuboid_data,
                 QDM_domain, QDM_spacing, QDM_deltax,
                 QDM_deltay, QDM_area, sample_height,
                 scan_height, tol=1e-7):
        """ Initializes class

        Arguments:
        QDM_data (string) -- Matrixfile (Nx columns, Ny rows)
                    containing QDM data in T
        cuboid_data (string) -- File (x, y, z, dx, dy, dz, index)
                       containing location and size grains in microm
        QDM_domain (2x2 numpy matrix) -- Size (metres) of QDM domain as
                                         np.array([[x1, y1], [x2, y2]])
        QDM_spacing (float) -- Distance between two adjacent scanning
                               points in metres
        QDM_deltax (float) -- half length of QDM sensor
        QDM_deltay (float) -- half width of QDM sensor
        QDM_area (float) -- Area of QDM sensor in square metres
        sample_height (float) -- Thickness of sample in metres
        scan_height (float) -- Distance between sample and QDM scanner
                               in metres
        tol (float) -- Tolerance for checking QDM_domain.
                       Default is 1e-7
        """

        self.QDM_data = Path(QDM_data)
        self.cuboid_data = Path(cuboid_data)
        self.QDM_domain = QDM_domain
        self.QDM_spacing = QDM_spacing
        self.QDM_deltax = QDM_deltax
        self.QDM_deltay = QDM_deltay
        self.QDM_area = QDM_area
        self.sample_height = sample_height
        self.scan_height = scan_height

        self.Ny, self.Nx = np.loadtxt(QDM_data).shape
        new_domain = self.QDM_domain[0, 0] \
            + (self.Nx - 1) * self.QDM_spacing
        if abs(new_domain - self.QDM_domain[1, 0]) > tol:
            print(f'QDM_domain[1, 0] has been reset from '
                  f'{self.QDM_domain[1, 0]} to {new_domain}.')
            self.QDM_domain[1, 0] = new_domain

        new_domain = self.QDM_domain[0, 1] \
            + (self.Ny - 1) * self.QDM_spacing
        if abs(new_domain - self.QDM_domain[1, 1]) > tol:
            print(f'QDM_domain[1, 1] has been reset from '
                  f'{self.QDM_domain[1, 1]} to {new_domain}.')
            self.QDM_domain[1, 1] = new_domain

        if abs(self.QDM_deltax * self.QDM_deltay * 4
               - self.QDM_area) > tol**2:
            print('The sensor is not a rectangle. '
                  'Calculation will probably go wrong here!')

    def read_files(self, factor=1e-6):
        """ Reads in QDM_data and cuboid_data
        """

        self.QDM_matrix = np.loadtxt(self.QDM_data) * self.QDM_area
        self.cuboids = np.loadtxt(self.cuboid_data)
        self.cuboids[:, :6] = self.cuboids[:, :6] * factor
        self.Npart = len(np.unique(self.cuboids[:, 6]))
        self.Ncub = len(self.cuboids[:, 6])

    
    def prepare_matrix(self, Origin=True):
        """ prepares for populate_matrix
        """

        self.Forward_G = np.zeros((self.Nx * self.Ny, 3 * self.Npart))
        self.Forward_G = populate_matrix(
            self.Forward_G, self.QDM_domain, self.scan_height,
            self.cuboids, self.Npart, self.Ny, self.Nx,
            self.QDM_spacing, self.QDM_deltax, self.QDM_deltay,
            Origin=True)
        

    def calculate_inverse(self, method='svd'):
        """ Calculate the inverse and give solution
        method is either svd using scipy.linalg.pinv or
        dgetrf using scipy.linalg.dgetrs and dgetrf
        """

        QDM_flatten = self.QDM_matrix.flatten()
        if self.Forward_G.shape[0] >= self.Forward_G.shape[1]:
            print(f'Start inversion with {self.Forward_G.shape[0]} '
                  f'knowns and {self.Forward_G.shape[1]} unknowns')
            if method == 'svd':
                Inverse_G = pinv(self.Forward_G)
                self.Mag = np.matmul(Inverse_G, QDM_flatten)
                print("Inversion has been carried out")

            elif method == 'dgetrf':
                GtG = np.matmul(self.Forward_G.transpose,
                                self.Forward_G)
                GtG_shuffle, IPIV, INFO1 = dgetrf(GtG)
                if INFO1 == 0:
                    print('Inversion is carried out!')
                    GtQDM = np.matmul(self.Forward_G, QDM_flatten)
                    self.Mag, INFO2 = dgetrs(GtG_shuffle, IPIV, GtQDM)
                    if INFO2 != 0:
                        self.Mag = None
                        print(f'{INFO2}th argument has an'
                              'illegal value. self.Mag deleted')
                    else:
                        print("Inversion has been carried out")
                else:
                    print(f'{INFO1}th argument has an illegal value')

            else:
                print(f'Method {method} is not recognized')
        else:
            print(f'Problem is underdetermined with '
                  f'{self.Forward_G.shape[0]} knowns and '
                  f'{self.Forward_G.shape[1]} unknowns')

    def obtain_magnetization(self):
        """ Groups functions together needed for
            magnetizatiob
        """

        self.read_files()
        self.prepare_matrix()
        self.calculate_inverse()

    def plot_contour(self, ax, tol=1e-7):
        """ Plots contour of grains at ax (matplotlib axis)
        tolerance is used to enable merging cuboids
        of one grain. default is 1e-7
        """

        counter = 0
        grainnr = 1
        # plot grains with number
        for i in range(self.cuboids.shape[0]):
            xs, ys = self.cuboids[i, 0:2]
            dx, dy = self.cuboids[i, 3:5]

            # Create overlapping polygon
            polygon = Polygon([(xs - dx - tol, ys - dy - tol),
                               (xs - dx - tol, ys + dy + tol),
                               (xs + dx + tol, ys + dy + tol),
                               (xs + dx + tol, ys - dy - tol)])

            # If continuing with same grain
            if self.cuboids[i, 6] == grainnr:
                if counter == 0:  # No grain created before
                    united_polygons = polygon
                    counter += 1
                else:  # If a polygon has been made before
                    # Merge polygons into one
                    united_polygons = cascaded_union([polygon,
                                                      united_polygons])
                    counter += 1

            # Last particle
            if i == self.cuboids.shape[0] - 1:
                united_polygons = cascaded_union([polygon,
                                                  united_polygons])
                mean = united_polygons.representative_point().wkt
                mean = np.matrix(mean[7:-1])
                ax.add_patch(PolygonPatch(united_polygons,
                                          facecolor="None",
                                          edgecolor='black'))
                ax.text(mean[0, 0], mean[0, 1], int(grainnr),
                        fontsize=20,
                        horizontalalignment="center",
                        verticalalignment="center")

                # If new grain is started
            if self.cuboids[i, 6] != grainnr:
                ax.add_patch(PolygonPatch(united_polygons,
                                          facecolor="None",
                                          edgecolor='black'))
                # Get center united polygon to plot grain number there
                mean = united_polygons.representative_point().wkt
                mean = np.matrix(mean[7:-1])

                # Get grain number plotted with corresponding grain
                ax.text(mean[0, 0], mean[0, 1], int(grainnr),
                        fontsize=20,
                        horizontalalignment="center",
                        verticalalignment="center")
                grainnr = int(self.cuboids[i, 6])  # New grain number
                del united_polygons  # Delete old assembled polygon
                counter = 0  # Reset counter
                united_polygons = polygon  # Start with new polygon
                counter += 1
        return ax

    def plot_magnetization(self, ax, ax2=None,
                           colormap='coolwarm', tol=1e-7):
        """ Plots magnetization of grains with colorscale on mpl axis
        ax2 can be used for the colorbar
        default colormap is coolwarm
        tolerance is used to enable merging cuboids
        of one grain. default is 1e-7
        """

        counter = 0  # Keeps track of amount of cuboids per grain
        counter2 = 0  # Keeps track of amount of grains
        cvalmax = 0
        grainnr = 1

        # Get minimum/maximum magnetization to make proper colorscale
        cvalmin = np.sqrt(self.Mag[0] ** 2 + self.Mag[1] ** 2
                          + self.Mag[2] ** 2)
        partmin = 1
        # If more than one grain
        if self.cuboids.shape[0] > 1:
            for i in range(self.Npart):
                cvalnew = np.sqrt(np.power(self.Mag[3 * i], 2)
                                  + np.power(self.Mag[3 * i + 1], 2)
                                  + np.power(self.Mag[3 * i + 2], 2))
                if cvalnew > cvalmax:
                    cvalmax = cvalnew
                    partmax = i + 1
                if cvalmin > cvalnew:
                    cvalmin = cvalnew
                    partmin = i + 1
            print(f"Minimum is {cvalmin} at particle {partmin}."
                  f" Maximum is {cvalmax} at particle {partmax}.")
            # get colormap
            coolwarm = mpl.cm.get_cmap(colormap,
                                       int((np.log10(cvalmax)
                                            - np.log10(cvalmin))
                                           * 1000))
        else:
            cvalmax = cvalmin + 100
            cvalmin = cvalmin - 100
            print('Only one grain')
            coolwarm = mpl.cm.get_cmap(colormap, 200)

        # Plot grains now with number
        for i in range(self.cuboids.shape[0]):
            xs, ys = self.cuboids[i, 0:2]
            dx, dy = self.cuboids[i, 3:5]

            # Create polygon
            polygon = Polygon([(xs - dx - tol, ys - dy - tol),
                               (xs - dx - tol, ys + dy + tol),
                               (xs + dx + tol, ys + dy + tol),
                               (xs + dx + tol, ys - dy - tol)])

            # If continuing with same grain
            if self.cuboids[i, 6] == grainnr:
                # If cuboid part of same grain
                if counter == 0:  # If no polygon has been made before
                    united_polygons = polygon
                    counter += 1
                else:  # If a polygon has been made before
                    united_polygons = cascaded_union([polygon,
                                                      united_polygons])
                    # Merge polygons into one
                    counter += 1

            # If new grains is started
            if self.cuboids[i, 6] != grainnr:
                cvalue = np.sqrt(np.power(self.Mag[3 * counter2], 2)
                                 + np.power(self.Mag[
                                     3 * counter2 + 1], 2)
                                 + np.power(self.Mag[
                                     3 * counter2 + 2], 2))

                ax.add_patch(PolygonPatch(united_polygons,
                                          facecolor=coolwarm(int((
                                              np.log10(cvalue)
                                              - np.log10(cvalmin))
                                              * 1000)),
                                          edgecolor='black'))

                mean = united_polygons.representative_point().wkt
                mean = np.matrix(mean[7:-1])
                # Get grain numbers with grains
                ax.text(mean[0, 0], mean[0, 1], int(grainnr),
                        fontsize=20,
                        horizontalalignment="center",
                        verticalalignment="center")
                grainnr = int(self.cuboids[i, 6])  # Grain number
                del united_polygons  # Delete old assembled polygon
                counter = 0  # Reset counter
                united_polygons = polygon  # Start with new polygon
                counter += 1
                counter2 += 1

            # If last particle is reached
            if i == self.cuboids.shape[0] - 1:
                cvalue = np.sqrt(np.power(self.Mag[3 * counter2], 2)
                                 + np.power(self.Mag[
                                     3 * counter2 + 1], 2)
                                 + np.power(self.Mag[
                                     3 * counter2 + 2], 2))
                united_polygons = cascaded_union([polygon,
                                                  united_polygons])
                mean = united_polygons.representative_point().wkt
                mean = np.matrix(mean[7:-1])

                # If only one grain in total, go to else
                if self.cuboids.shape[0] > 1:
                    ax.add_patch(PolygonPatch(united_polygons,
                                              facecolor=coolwarm(int((
                                                  np.log10(cvalue)
                                                  - np.log10(cvalmin))
                                                  * 1000)),
                                              edgecolor='black'))
                    norm = mpl.colors.LogNorm(vmin=cvalmin,
                                              vmax=cvalmax)

                else:
                    ax.add_patch(PolygonPatch(united_polygons,
                                              facecolor=coolwarm(int(
                                                  cvalue - cvalmin)),
                                              edgecolor='black'))
                    norm = mpl.colors.Normalize(vmin=cvalmin,
                                                vmax=cvalmax)

                ax.text(mean[0, 0], mean[0, 1], int(grainnr),
                        fontsize=20,
                        horizontalalignment="center",
                        verticalalignment="center")

                if ax2 is not None:
                    cb1 = mpl.colorbar.ColorbarBase(
                        ax2, cmap=coolwarm, norm=norm,
                        orientation='horizontal')
                    cb1.set_label('M (A/m)')
                    return ax, ax2
                else:
                    return ax

    def save_results(self, Magfile,
                     path_to_plot=None, colormap='coolwarm'):
        """ Saves magnetization to a specified file
            and makes plots if path_to_plot (string)
            has been set. colormap default set at coolwarm
        """

        np.savetxt(Magfile, self.Mag.reshape(self.Npart, 3))
        if path_to_plot is not None:
            self.path_to_plot = Path(path_to_plot)
            # Original magnetic field with grains
            fig1, ax = plt.subplots(figsize = (25, 15))
            Bzplot = ax.imshow(self.QDM_matrix / self.QDM_area,
                               cmap = colormap,
                               extent = (self.QDM_domain[0, 0],
                                         self.QDM_domain[1, 0],
                                         self.QDM_domain[0, 1],
                                         self.QDM_domain[1, 1]))
            ax = self.plot_contour(ax)
            ax.set_title('Measured field with grains')
            ax.set_xlim(self.QDM_domain[0, 0], self.QDM_domain[1, 0])
            ax.set_ylim(self.QDM_domain[0, 1], self.QDM_domain[1, 1])
            cbar = plt.colorbar(Bzplot)
            cbar.set_label('B (T)')
            plt.savefig(self.path_to_plot / "Original_field.png")
            
            # Forward field with grains
            Forward_field = (np.matmul(
                self.Forward_G, self.Mag) / self.QDM_area)
            Forward_field = Forward_field.reshape(self.Ny, self.Nx)
            fig2, ax = plt.subplots(figsize = (25, 15))
            Bzforwplot = ax.imshow(Forward_field, cmap = colormap,
                                   extent = (self.QDM_domain[0, 0],
                                             self.QDM_domain[1, 0],
                                             self.QDM_domain[0, 1],
                                             self.QDM_domain[1, 1]))
            ax = self.plot_contour(ax)
            ax.set_title('Forward field with grains')
            ax.set_xlim(self.QDM_domain[0, 0], self.QDM_domain[1, 0])
            ax.set_ylim(self.QDM_domain[0, 1], self.QDM_domain[1, 1])
            cbar = plt.colorbar(Bzforwplot)
            cbar.set_label('B (T)')
            plt.savefig(self.path_to_plot / "Forward_field.png")

            # residual field with grains
            fig3, ax = plt.subplots(figsize = (25, 15))
            diffplot = ax.imshow(
                Forward_field - self.QDM_matrix / self.QDM_area,
                cmap = colormap, extent =
                (self.QDM_domain[0, 0],
                 self.QDM_domain[1, 0],
                 self.QDM_domain[0, 1],
                 self.QDM_domain[1, 1]))
            ax = self.plot_contour(ax)
            ax.set_title('Residual field with grains')
            ax.set_xlim(self.QDM_domain[0, 0], self.QDM_domain[1, 0])
            ax.set_ylim(self.QDM_domain[0, 1], self.QDM_domain[1, 1])
            cbar = plt.colorbar(diffplot)
            cbar.set_label('B (T)')
            plt.savefig(self.path_to_plot / "Residual_field.png")

            # Grain magnetization
            fig4, (ax, ax2) = plt.subplots(2, 1, figsize = (25, 15),
                                          gridspec_kw=
                                          {'height_ratios': [10, 1]})
            diffplot = ax.imshow(
                Forward_field - self.QDM_matrix / self.QDM_area,
                cmap = 'viridis', extent =
                (self.QDM_domain[0, 0],
                 self.QDM_domain[1, 0],
                 self.QDM_domain[0, 1],
                 self.QDM_domain[1, 1]))
            ax, ax2 = self.plot_magnetization(ax, ax2)
            ax.set_title('Residual field with magnetization grains')
            ax.set_xlim(self.QDM_domain[0, 0], self.QDM_domain[1, 0])
            ax.set_ylim(self.QDM_domain[0, 1], self.QDM_domain[1, 1])
#             cbar = plt.colorbar(diffplot)
#             cbar.set_label('B (T)')
            plt.savefig(self.path_to_plot / "Magnetization.png")

    def forward_field(self, Forwardfile, snrfile=None, tol=0.9):
        """ Calculates forward field and signal to noise ratio and saves it
        tol stands for percentage of signal used (0.9 is 90% default)
        if snrfile is None, no signal to noise ratio is calculated
        """

        Forward_field = np.matmul(self.Forward_G, self.Mag)
        np.savetxt(Forwardfile, Forward_field.reshape(self.Ny, self.Nx)
                   / self.QDM_area)
        if snrfile is not None:
            org_field = self.QDM_matrix.flatten()
            residual = org_field - Forward_field
            snr = np.zeros(self.Forward_G.shape[1])
            el_signal = np.zeros((self.Forward_G.shape[0], 2))
            for column in range(self.Forward_G.shape[1]):
                el_signal[:, 0] = self.Forward_G[:, column] * self.Mag[column]
                el_signal[:, 1] = residual * self.QDM_area
                el_sum = np.sum(abs(el_signal[:, 0]))
                el_signal = el_signal[np.argsort(abs(el_signal[:, 0]))]
                res_sum = 0
                forw_sum = 0
                for item in range(1, len(el_signal[:, 0])+1):
                    res_sum += abs(el_signal[-item, 1])
                    forw_sum += abs(el_signal[-item, 0])
                    if forw_sum / el_sum > tol:
                        snr[column] = forw_sum / res_sum
                        break
            np.savetxt(snrfile, snr)