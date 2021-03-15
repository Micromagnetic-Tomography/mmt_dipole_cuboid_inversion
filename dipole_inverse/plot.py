import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.ops import cascaded_union


def plot_contour(DipoleIns, ax, tol=1e-7):
    """ Using an instance of the Dipole class, plots contour of grains at a
    given matplotlib axis object

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis.
    tol
        Tolerance is used to enable merging cuboids of one grain.
    """

    counter = 0
    grainnr = 1
    # plot grains with number
    for i in range(DipoleIns.cuboids.shape[0]):
        xs, ys = DipoleIns.cuboids[i, 0:2]
        dx, dy = DipoleIns.cuboids[i, 3:5]

        # Create overlapping polygon
        polygon = Polygon([(xs - dx - tol, ys - dy - tol),
                           (xs - dx - tol, ys + dy + tol),
                           (xs + dx + tol, ys + dy + tol),
                           (xs + dx + tol, ys - dy - tol)])

        # If continuing with same grain
        if DipoleIns.cuboids[i, 6] == grainnr:
            if counter == 0:  # No grain created before
                united_polygons = polygon
                counter += 1
            else:  # If a polygon has been made before
                # Merge polygons into one
                united_polygons = cascaded_union([polygon,
                                                  united_polygons])
                counter += 1

        # Last particle
        if i == DipoleIns.cuboids.shape[0] - 1:
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
        if DipoleIns.cuboids[i, 6] != grainnr:
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
            grainnr = int(DipoleIns.cuboids[i, 6])  # New grain number
            del united_polygons  # Delete old assembled polygon
            counter = 0  # Reset counter
            united_polygons = polygon  # Start with new polygon
            counter += 1
    return ax


def plot_magnetization(DipoleIns, ax, ax2=None,
                       colormap='coolwarm', tol=1e-7):
    """
    Plots the magnetization of grains with colorscale on a Matplotlib axis

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    ax2
        Optional axis for the colorbar
    colormap
        default colormap is coolwarm
    tol
        Tolerance used to enable merging cuboids of one grain
    """

    counter = 0  # Keeps track of amount of cuboids per grain
    counter2 = 0  # Keeps track of amount of grains
    cvalmax = 0
    grainnr = 1

    # Get minimum/maximum magnetization to make proper colorscale
    cvalmin = np.sqrt(DipoleIns.Mag[0] ** 2 + DipoleIns.Mag[1] ** 2
                      + DipoleIns.Mag[2] ** 2)
    partmin = 1
    # If more than one grain
    if DipoleIns.cuboids.shape[0] > 1:
        for i in range(DipoleIns.Npart):
            cvalnew = np.sqrt(np.power(DipoleIns.Mag[3 * i], 2)
                              + np.power(DipoleIns.Mag[3 * i + 1], 2)
                              + np.power(DipoleIns.Mag[3 * i + 2], 2))
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
    for i in range(DipoleIns.cuboids.shape[0]):
        xs, ys = DipoleIns.cuboids[i, 0:2]
        dx, dy = DipoleIns.cuboids[i, 3:5]

        # Create polygon
        polygon = Polygon([(xs - dx - tol, ys - dy - tol),
                           (xs - dx - tol, ys + dy + tol),
                           (xs + dx + tol, ys + dy + tol),
                           (xs + dx + tol, ys - dy - tol)])

        # If continuing with same grain
        if DipoleIns.cuboids[i, 6] == grainnr:
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
        if DipoleIns.cuboids[i, 6] != grainnr:
            cvalue = np.sqrt(np.power(DipoleIns.Mag[3 * counter2], 2)
                             + np.power(DipoleIns.Mag[
                                 3 * counter2 + 1], 2)
                             + np.power(DipoleIns.Mag[
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
            grainnr = int(DipoleIns.cuboids[i, 6])  # Grain number
            del united_polygons  # Delete old assembled polygon
            counter = 0  # Reset counter
            united_polygons = polygon  # Start with new polygon
            counter += 1
            counter2 += 1

        # If last particle is reached
        if i == DipoleIns.cuboids.shape[0] - 1:
            cvalue = np.sqrt(np.power(DipoleIns.Mag[3 * counter2], 2)
                             + np.power(DipoleIns.Mag[
                                 3 * counter2 + 1], 2)
                             + np.power(DipoleIns.Mag[
                                 3 * counter2 + 2], 2))
            united_polygons = cascaded_union([polygon,
                                              united_polygons])
            mean = united_polygons.representative_point().wkt
            mean = np.matrix(mean[7:-1])

            # If only one grain in total, go to else
            if DipoleIns.cuboids.shape[0] > 1:
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


# TODO: this function requires to be split. Options need to be more general
def plot_inversion_results(DipoleIns, save_path, colormap='coolwarm'):
    """
    Plot the results of an inversion using the Dipole class: forward field,
    residual and magnetization

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    save_path
        pathlib Path with a directory to which saving the plots
    """

    # Original magnetic field with grains
    fig1, ax = plt.subplots(figsize=(25, 15))
    Bzplot = ax.imshow(DipoleIns.QDM_matrix / DipoleIns.QDM_area,
                       cmap=colormap,
                       extent=(DipoleIns.QDM_domain[0, 0],
                               DipoleIns.QDM_domain[1, 0],
                               DipoleIns.QDM_domain[1, 1],
                               DipoleIns.QDM_domain[0, 1]))
    ax = DipoleIns.plot_contour(ax)
    ax.set_title('Measured field with grains')
    ax.set_xlim(DipoleIns.QDM_domain[0, 0], DipoleIns.QDM_domain[1, 0])
    ax.set_ylim(DipoleIns.QDM_domain[0, 1], DipoleIns.QDM_domain[1, 1])
    cbar = plt.colorbar(Bzplot)
    cbar.set_label('B (T)')
    plt.savefig(save_path / "Original_field.png")

    # Forward field with grains
    Forward_field = (np.matmul(
        DipoleIns.Forward_G, DipoleIns.Mag) / DipoleIns.QDM_area)
    Forward_field = Forward_field.reshape(DipoleIns.Ny, DipoleIns.Nx)
    fig2, ax = plt.subplots(figsize=(25, 15))
    Bzforwplot = ax.imshow(Forward_field, cmap=colormap,
                           extent=(DipoleIns.QDM_domain[0, 0],
                                   DipoleIns.QDM_domain[1, 0],
                                   DipoleIns.QDM_domain[1, 1],
                                   DipoleIns.QDM_domain[0, 1]))
    ax = DipoleIns.plot_contour(ax)
    ax.set_title('Forward field with grains')
    ax.set_xlim(DipoleIns.QDM_domain[0, 0], DipoleIns.QDM_domain[1, 0])
    ax.set_ylim(DipoleIns.QDM_domain[0, 1], DipoleIns.QDM_domain[1, 1])
    cbar = plt.colorbar(Bzforwplot)
    cbar.set_label('B (T)')
    plt.savefig(save_path / "Forward_field.png")

    # residual field with grains
    fig3, ax = plt.subplots(figsize=(25, 15))
    diffplot = ax.imshow(
        Forward_field - DipoleIns.QDM_matrix / DipoleIns.QDM_area,
        cmap=colormap, extent=(DipoleIns.QDM_domain[0, 0],
                               DipoleIns.QDM_domain[1, 0],
                               DipoleIns.QDM_domain[1, 1],
                               DipoleIns.QDM_domain[0, 1]))
    ax = DipoleIns.plot_contour(ax)
    ax.set_title('Residual field with grains')
    ax.set_xlim(DipoleIns.QDM_domain[0, 0], DipoleIns.QDM_domain[1, 0])
    ax.set_ylim(DipoleIns.QDM_domain[0, 1], DipoleIns.QDM_domain[1, 1])
    cbar = plt.colorbar(diffplot)
    cbar.set_label('B (T)')
    plt.savefig(save_path / "Residual_field.png")

    # Grain magnetization
    fig4, (ax, ax2) = plt.subplots(2, 1, figsize=(25, 15),
                                   gridspec_kw={'height_ratios': [10, 1]})
    diffplot = ax.imshow(
        Forward_field - DipoleIns.QDM_matrix / DipoleIns.QDM_area,
        cmap='viridis', extent=(DipoleIns.QDM_domain[0, 0],
                                DipoleIns.QDM_domain[1, 0],
                                DipoleIns.QDM_domain[1, 1],
                                DipoleIns.QDM_domain[0, 1]))
    ax, ax2 = DipoleIns.plot_magnetization(ax, ax2)
    ax.set_title('Residual field with magnetization grains')
    ax.set_xlim(DipoleIns.QDM_domain[0, 0], DipoleIns.QDM_domain[1, 0])
    ax.set_ylim(DipoleIns.QDM_domain[0, 1], DipoleIns.QDM_domain[1, 1])
#             cbar = plt.colorbar(diffplot)
#             cbar.set_label('B (T)')
    plt.savefig(save_path / "Magnetization.png")
