import numpy as np
import shapely.geometry as shg
import shapely.ops as sho
from typing import Tuple
from typing import Union


def generate_grain_vertices(cuboids: np.ndarray) -> np.ndarray:
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
                              cuboid_idxs: np.ndarray = np.array([]),
                              cuboid_idxs_unique: np.ndarray = np.array([]),
                              polygon_buffer: float = 1e-5,
                              generate_labels: bool = False,
                              ) -> Union[Tuple[list, dict],
                                         Tuple[list, dict, dict]]:
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
    # TODO: In Ptyhon >3.9 we can be more specific about the types returned:
    # dict[np.ndarray] for example, without importing Dict from typing

    if cuboid_idxs.size == 0:
        cuboid_idxs = cuboids[:, 6].astype(np.int32)
    if cuboid_idxs_unique == 0:
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
    grain_labs = {}
    grain_geoms_coords = {}
    for i, pg in enumerate(grain_geoms):
        idx = cuboid_idxs_unique[i]  # get index of the grain

        if generate_labels:
            labc = pg.representative_point().xy
            grain_labs[idx] = np.array(labc).reshape(-1)

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

    if generate_labels:
        return grain_geoms, grain_geoms_coords, grain_labs
    else:
        return grain_geoms, grain_geoms_coords
