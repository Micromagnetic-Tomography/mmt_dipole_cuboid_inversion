import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
# import matplotlib as mpl
import numpy as np
from . import grain_geometry_tools as ggt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.colors import LightSource
import matplotlib.colors as mcolors


def set_grain_geometries(DipoleIns,
                         spatial_scaling=None,
                         tol=1e-7):
    """
    Generates multiple arrays with grain geometry properties and generates them
    as variables in the DipoleIns instance. Run this function before performing
    any plotting.

    The variables generated are:

        cuboid_idxs
            Indices of the cuboids in a 1D np.int32 array
        cuboid_idxs_unique
            Unique indices
        cuboid_idxs_counts
            Count of the indices (same length as cuboid_idxs_unique)
        grain_vertices
            Vertices of the grain cuboids as viewed from the top
        grain_geoms
            Shapely polygons obtained by merging grain_vertices of cuboids
            belonging to a single grain
        grain_geoms_coords
            Coordinates of the merged shapely polygons
        grain_centroids
            Dictionary with the index of the grains as keys and a 2-array
            with the coordinates of the centroid of the merged shapely polygon
        spatial_scaling
            Scaling factor for the coordinates. This same scaling is applied
            to the limits of the QDM scan data and inversion results
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

    if spatial_scaling:
        DipoleIns.spatial_scaling = spatial_scaling
    else:
        DipoleIns.spatial_scaling = 1.


def plot_grain_boundaries(DipoleIns, ax,
                          grain_labels=True,
                          boundaries_args=dict(ec=(0, 0, 0, 1)),
                          labels_args=dict(ha='center', va='center'),
                          tol=1e-7):
    """
    Plots the grain boundaries viewed from a bird eye perspective at the
    z-axis

    Notes
    -----
    Requires set_grain_geometries to be applied to DipoleIns beforehand

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    boundaries_args
        Arguments to the PolyCollection handling the boundary drawings
    labels_args
        Arguments to the matplotlib text objects handling the grain labels
    tol
        Tolerance used to enable merging cuboids of one grain

    """
    grain_geoms_coords = DipoleIns.grain_geoms_coords
    scale = DipoleIns.spatial_scaling

    # Draw grain outlines -----------------------------------------------------
    for i, pg_key in enumerate(grain_geoms_coords):
        # We can use i to colour the grain
        col = PolyCollection([grain_geoms_coords[pg_key] * scale],
                             # fc=cmap(i + 1),
                             fc=(0, 0, 0, 0),
                             **boundaries_args
                             )
        ax.add_collection(col)

    if grain_labels:
        grain_cs = DipoleIns.grain_centroids
        for grain_idx in grain_cs:
            ax.text(grain_cs[grain_idx][0] * scale,
                    grain_cs[grain_idx][1] * scale,
                    grain_idx,
                    **labels_args
                    )

    ax.autoscale()


def plot_magnetization_on_grains(DipoleIns,
                                 ax,
                                 mag_log_scale=True,
                                 grain_plot_args=dict(cmap='coolwarm'),
                                 grain_labels=True,
                                 labels_args=dict(ha='center', va='center'),
                                 tol=1e-7):
    """
    Plots the magnetization of grains with colorscale on a Matplotlib axis

    Notes
    -----
    Requires set_grain_geometries to be applied to DipoleIns beforehand

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    mag_log_scale
        Color grains using a log scale of the magnetization
    grain_plot_args
        Arguments to the matplotlib PolyCollection handling the plotting of
        the grains. For example, set the colormap and clim (color limits) of
        the grains
    grain_labels
        Draw index labels at the centroid of the grains
    labels_args
        Arguments to the matplotlib text objects handling the grain labels
    tol
        Tolerance used to enable merging cuboids of one grain

    """

    scale = DipoleIns.spatial_scaling

    mag = np.linalg.norm(DipoleIns.Mag.reshape(-1, 3), axis=1)
    # cuboid_idxs = DipoleIns.cuboid_idxs
    cuboid_idxs_unique, cuboid_idxs_counts = (DipoleIns.cuboid_idxs_unique,
                                              DipoleIns.cuboid_idxs_counts)

    # Generate vertexes to plot the grains
    grain_vertices = DipoleIns.grain_vertices
    grain_geoms_coords, grain_cs = (DipoleIns.grain_geoms_coords,
                                    DipoleIns.grain_centroids)

    # Draw grain shapes and color according to magnetization
    col = PolyCollection(grain_vertices * scale)
    mag_per_cuboid = np.repeat(mag, cuboid_idxs_counts)
    if mag_log_scale:
        mag_per_cuboid = np.log(mag_per_cuboid)
    col.set(array=mag_per_cuboid, **grain_plot_args)
    p = ax.add_collection(col)

    if grain_labels:
        for grain_idx in grain_cs:
            txt = ax.text(grain_cs[grain_idx][0] * scale,
                          grain_cs[grain_idx][1] * scale,
                          grain_idx,
                          **labels_args)

    ax.autoscale()


def plot_scan_field(DipoleIns,
                    ax,
                    imshow_args=dict(cmap='magma')):
    """
    Plots the original scan field data

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    imshow_args
        Extra arguments passed to the imshow plot

    Returns
    -------
    im
        The imshow plot object
    """

    scale = DipoleIns.spatial_scaling

    dx = scale * DipoleIns.QDM_deltax
    dy = scale * DipoleIns.QDM_deltay

    im = ax.imshow(DipoleIns.QDM_matrix, origin='lower',
                   extent=[scale * DipoleIns.QDM_domain[0, 0] - dx,
                           scale * DipoleIns.QDM_domain[1, 0] + dx,
                           scale * DipoleIns.QDM_domain[0, 1] - dy,
                           scale * DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


def plot_inversion_field(DipoleIns,
                         ax,
                         imshow_args=dict(cmap='magma')):
    """
    Plots the inverted field calculated with the Dipole class

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    imshow_args
        Extra arguments passed to the imshow plot

    Returns
    -------
    im
        The imshow plot object
    """
    scale = DipoleIns.spatial_scaling

    dx = scale * DipoleIns.QDM_deltax
    dy = scale * DipoleIns.QDM_deltay

    inv_field = (DipoleIns.Forward_G @ DipoleIns.Mag).reshape(DipoleIns.Ny, -1)
    im = ax.imshow(inv_field, origin='lower',
                   extent=[scale * DipoleIns.QDM_domain[0, 0] - dx,
                           scale * DipoleIns.QDM_domain[1, 0] + dx,
                           scale * DipoleIns.QDM_domain[0, 1] - dy,
                           scale * DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


def plot_residual(DipoleIns,
                  ax,
                  scale_residual=True,
                  imshow_args=dict(cmap='magma')):
    """
    Plots the residual from the inversion of the field

    Parameters
    ----------
    DipoleIns
        An instance of the Dipole class
    ax
        Matplotlib axis
    scale_residual
        If True the residual is scaled by the QDM_area value
    imshow_args
        Extra arguments passed to the imshow plot

    Returns
    -------
    im
        The imshow plot object
    """

    scale = DipoleIns.spatial_scaling

    if scale_residual:
        scale_res = DipoleIns.QDM_area
    else:
        scale_res = 1.

    dx = scale * DipoleIns.QDM_deltax
    dy = scale * DipoleIns.QDM_deltay

    Forward_field = (DipoleIns.Forward_G @ DipoleIns.Mag) / scale_res
    Forward_field.shape = (DipoleIns.Ny, DipoleIns.Nx)
    res = Forward_field - DipoleIns.QDM_matrix / scale_res

    im = ax.imshow(res, origin='lower',
                   extent=[scale * DipoleIns.QDM_domain[0, 0] - dx,
                           scale * DipoleIns.QDM_domain[1, 0] + dx,
                           scale * DipoleIns.QDM_domain[0, 1] - dy,
                           scale * DipoleIns.QDM_domain[1, 1] + dy],
                   **imshow_args)

    return im


# TODO: this function requires to be split. Options need to be more general
def plot_inversion_results(DipoleIns, save_path, colormap='coolwarm'):
    """
    ** DEPRECATED **

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


# -----------------------------------------------------------------------------
# 3D Plotting of cubes
# -----------------------------------------------------------------------------


# Cuboid plotting in Matplotlib
# Code based on: https://stackoverflow.com/a/42611693/498816
def _cuboids(o, size=(1, 1, 1)):
    shift = np.zeros(3)
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],  # bottom
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # left
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],  # front
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],  # back
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],  # right
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]  # top

    X = np.array(X).astype(float)
    for i in range(3):
        shift[i] = size[i] * 0.5
        X[:, :, i] *= size[i]
    X += np.array(o) - shift
    return X


def plotCubeAt(positions,
               sizes=None,
               colors=None,
               lightsource=None,
               **kwargs
               ):
    """Generates a Matplotlib collection of cuboids from their center locations

    Parameters
    ----------
    positions
        Numpy array with shape `(N,3)` with the `x,y,z` centers of the cuboids
    sizes
        Numpy array with shape `(N,3)` with the `lx,ly,lz` side lengths of the
        cuboids
    colors
        Numpy array of size `(N, 3)` with the RGBA colors for every cuboid
    lightsource
        Can be passed as True or as an instance of a Matplotlib's LightSource

    Returns
    -------
    Poly3DCollection of cuboids

    """
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    elif isinstance(colors, (list, np.ndarray)) and len(colors) == 4:
        colors = np.tile(np.array(colors), len(positions)).reshape(-1, 4)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        cube = _cuboids(p, size=s)
        g.append(cube)

    colors = np.repeat(colors, 6, axis=0)

    # Shadow the cubes if asked
    if lightsource:
        # normals
        N = np.array([[0, 0, -1],
                      [0, -1, 0],
                      [1, 0, 0],
                      [-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        N = np.tile(N.reshape(-1), len(positions)).reshape(-1, 3)

        # If ls is passed as True
        if isinstance(lightsource, bool):
            ls = None
        else:
            ls = lightsource

        colors = _shade_colors(colors, N, lightsource=ls)

    col3d = Poly3DCollection(np.concatenate(g),
                             facecolors=colors, **kwargs)
    return col3d


# -----------------------------------------------------------------------------

# Taken from: https://github.com/matplotlib/matplotlib
def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*.
    *color* can also be an array of the same length as *normals*.
    """
    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)

    with np.errstate(invalid="ignore"):
        shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
                 @ lightsource.direction)
    mask = ~np.isnan(shade)

    if mask.any():
        # convert dot product to allowed shading fractions
        in_norm = mcolors.Normalize(-1, 1)
        out_norm = mcolors.Normalize(0.3, 1).inverse

        def norm(x):
            return out_norm(in_norm(x))

        shade[~mask] = 0

        color = mcolors.to_rgba_array(color)
        # shape of color should be (M, 4) (where M is number of faces)
        # shape of shade should be (M,)
        # colors should have final shape of (M, 4)
        alpha = color[:, 3]
        colors = norm(shade)[:, np.newaxis] * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors
