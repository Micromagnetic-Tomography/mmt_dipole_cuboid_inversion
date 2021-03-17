import numpy as np
import shapely.geometry as shg
import shapely.ops as sho
from typing import Union


def generate_grain_vertices(cuboids: np.ndarray):
    """
    Generate grain vertices from a Numpy array containing at leats 6 columns

    Parameters
    ----------
    cuboids
        An N x 6 array with: x y z dx dy dz index
    """

    x, y, dx, dy = (cuboids[:, i] for i in (0, 1, 3, 4))
    grain_vertices = np.column_stack((x - dx, y - dy,
                                      x + dx, y - dy,
                                      x + dx, y + dy,
                                      x - dx, y + dy))
    grain_vertices.shape = (-1, 4, 2)

    return grain_vertices


def generate_grain_geometries(cuboids: np.ndarray,
                              grain_vertices: np.ndarray,
                              cuboid_idxs: Union[None, np.ndarray] = None,
                              cuboid_idxs_unique: Union[None, np.ndarray] = None,
                              polygon_buffer: float = 1e-5
                              ):
    """
    Generates polygons defining grain outlines from groups of cuboids. Polygons
    are given as shapely polygon objects.

    Parameters
    ----------
    cuboids
    grain_vertices
    cuboid_idxs
    cuboid_idxs_unique
    polygon_buffer

    Returns
    -------
        
    """
    if not cuboid_idxs:
        cuboid_idxs = cuboids[:, 6].astype(np.int32)
    if not cuboid_idxs_unique:
        cuboid_idxs_unique = np.unique(cuboid_idxs)
    grain_geoms = []
    for p in cuboid_idxs_unique:
        polygons = map(shg.Polygon,
                       grain_vertices[cuboid_idxs == p])
        # Some of the joined polygons might result in a Multipolygon!
        # Here we join all the polygons. The buffer is necessary to remove
        # odd looking lines appearing within a shape/polygon
        grain_geoms.append(
            sho.unary_union([shg.Polygon(pt.exterior).buffer(polygon_buffer,
                                                             cap_style=3,
                                                             join_style=shg.JOIN_STYLE.mitre)
                             for pt in polygons]))

    # Now get the coordinates of every grain geometry. If we have a
    # Multipolygon, we get the coordinates from separate entities
    # We save every geometry in a dictionary entry
    grain_geoms_coords = {}
    for i, pg in enumerate(grain_geoms):
        idx = cuboid_idxs_unique[i]  # get index of the grain

        if pg.type == 'MultiPolygon':
            grain_geoms_coords[idx] = np.empty((0, 2))
            for pol in pg:
                coords = pol.exterior.coords.xy
                grain_geoms_coords[idx] = np.vstack(
                    (grain_geoms_coords[idx],
                     np.column_stack(coords)))
        else:
            coords = pg.exterior.coords.xy
            grain_geoms_coords[idx] = np.column_stack(coords)

    return grain_geoms, grain_geoms_coords
