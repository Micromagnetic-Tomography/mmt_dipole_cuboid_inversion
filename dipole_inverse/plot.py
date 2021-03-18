import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl
import numpy as np
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.ops import cascaded_union
import grain_geometry_tools as ggt


def set_grain_geometries(DipoleIns, tol=1e-7):
    """
    Generates multiple arrays with grain geometries and apply them to the
    DipoleIns instance
    """

    DipoleIns.cuboid_idxs = DipoleIns.cuboids[:, 6].astype(np.int32)
    (DipoleIns.cuboid_idxs_unique,
     DipoleIns.cuboid_idxs_counts) = np.unique(DipoleIns.cuboid_idxs,
                                               return_counts=True)

    # Generate vertexes to plot the grains
    DipoleIns.grain_vertices = ggt.generate_grain_vertices(DipoleIns.cuboids)
    (DipoleIns.grain_geoms,
     DipoleIns.grain_geoms_coords,
     DipoleIns.grain_centroids) = ggt.generate_grain_geometries(
        DipoleIns.cuboids, DipoleIns.grain_vertices,
        DipoleIns.cuboid_idxs, DipoleIns.cuboid_idxs_unique,
        polygon_buffer=tol,
        generate_centroids=True)


def plot_grain_boundaries(DipoleIns, ax,
                          grain_labels=True,
                          tol=1e-7):
    """
    Plots the grain boundaries viewed from a bird eye perspective at the
    z-axis

    Note
    ----
    Requires set_grain_geometries to be applied to DipoleIns beforehand

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    tol
        Tolerance used to enable merging cuboids of one grain
    """
    grain_geoms_coords = DipoleIns.grain_geoms_coords

    # Draw grain outlines -----------------------------------------------------
    for i, pg_key in enumerate(grain_geoms_coords):
        # We can use i to colour the grain
        col = PolyCollection([grain_geoms_coords[pg_key]],
                             # fc=cmap(i + 1),
                             fc=(0, 0, 0, 0),
                             ec=(0, 0, 0, 1),
                             )
        ax.add_collection(col)

    if grain_labels:
        grain_cs = DipoleIns.grain_centroids
        for grain_idx in grain_cs:
            ax.text(grain_cs[grain_idx][0], grain_cs[grain_idx][1],
                    grain_idx,
                    # fontsize=20,
                    ha="center",
                    va="center")

    ax.autoscale()

    return None


def plot_magnetization_on_grains(DipoleIns, ax,
                                 mag_log_scale=True,
                                 plot_labels=True,
                                 colormap='coolwarm',
                                 tol=1e-7):
    """
    Plots the magnetization of grains with colorscale on a Matplotlib axis

    Note
    ----
    Requires set_grain_geometries to be applied to DipoleIns beforehand

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    colormap
        Colormap of the magnetization
    tol
        Tolerance used to enable merging cuboids of one grain
    """

    mag = np.linalg.norm(DipoleIns.Mag.reshape(-1, 3), axis=1)
    # cuboid_idxs = DipoleIns.cuboid_idxs
    cuboid_idxs_unique, cuboid_idxs_counts = (DipoleIns.cuboid_idxs_unique,
                                              DipoleIns.cuboid_idxs_counts)

    # Generate vertexes to plot the grains
    grain_vertices = DipoleIns.grain_vertices
    grain_geoms_coords, grain_cs = (DipoleIns.grain_geoms_coords,
                                    DipoleIns.grain_centroids)

    # Draw grain shapes and color according to magnetization
    col = PolyCollection(grain_vertices)
    mag_per_cuboid = np.repeat(mag, cuboid_idxs_counts)
    if mag_log_scale:
        mag_per_cuboid = np.log(mag_per_cuboid)
    col.set(array=mag_per_cuboid, cmap='coolwarm',
            # clim=(1e2, 1e8)
            )
    ax.add_collection(col)

    if plot_labels:
        for grain_idx in grain_cs:
            ax.text(grain_cs[grain_idx][0], grain_cs[grain_idx][1],
                    grain_idx,
                    # fontsize=20,
                    ha="center",
                    va="center")

    ax.autoscale()

    return None


def plot_scan_field(DipoleIns,
                    ax,
                    imshow_args=dict(cmap='magma')):
    """
    """

    dx = DipoleIns.QDM_domain_deltax * 0.5
    dy = DipoleIns.QDM_domain_deltay * 0.5

    im = ax.imshow(DipoleIns.QDM_matrix, origin='lower',
                   extent=[DipoleIns.QDM_domain[0, 0] - dx,
                           DipoleIns.QDM_domain[1, 0] + dx,
                           DipoleIns.QDM_domain[0, 1] - dy,
                           DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


def plot_inversion_field(DipoleIns,
                         ax,
                         imshow_args=dict(cmap='magma')):
    """
    """

    dx = DipoleIns.QDM_domain_deltax * 0.5
    dy = DipoleIns.QDM_domain_deltay * 0.5

    inv_field = (DipoleIns.Forward_G @ DipoleIns.Mag).reshape(DipoleIns.Ny, -1)
    im = ax.imshow(inv_field, origin='lower',
                   extent=[DipoleIns.QDM_domain[0, 0] - dx,
                           DipoleIns.QDM_domain[1, 0] + dx,
                           DipoleIns.QDM_domain[0, 1] - dy,
                           DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


def plot_residual(DipoleIns,
                  ax,
                  imshow_args=dict(cmap='magma')):
    """
    """

    dx = DipoleIns.QDM_domain_deltax * 0.5
    dy = DipoleIns.QDM_domain_deltay * 0.5
    Forward_field = (DipoleIns.Forward_G @ DipoleIns.Mag) / DipoleIns.QDM_area
    Forward_field.shape = (DipoleIns.Ny, DipoleIns.Nx)
    res = Forward_field - DipoleIns.QDM_matrix / DipoleIns.QDM_area,
    im = ax.imshow(res, origin='lower',
                   extent=[DipoleIns.QDM_domain[0, 0] - dx,
                           DipoleIns.QDM_domain[1, 0] + dx,
                           DipoleIns.QDM_domain[0, 1] - dy,
                           DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


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
