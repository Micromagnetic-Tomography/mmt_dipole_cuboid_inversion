# -*- coding: utf-8 -*-
"""
Cuboid decomposition of microCT particle voxel data
@author: Karl Fabian, NTNU
Licensed under MIT License, 2021
"""

import numpy as np
# import pandas as pd
import csv
import copy
import time


def LEQ(a, b):
    """
    Given two lists a and b, returns True only if all elements of a are less
    or equal than the elements of b. Return False when at least one element
    of a is >= than an element of b
    """
    bl = True
    for x in zip(a, b):
        bl = bl and x[0] <= x[1]
        # TODO: break here the first time x[0] > x[1]
    return bl


def TupleMaxima(lst):
    """
    Given a list of sequences (or 2-sequences) perform:
    - Sort them by their minimum value and invert sorting
    - Choose the one with the largest minimum (first element in sorted list)
      and append it to a list, say MAX
    - Iterate through the other sequences
      - If one of the iterated seq is not LEQ with the first element in MAX
        (largest minimum) then append it to the MAX list
        The next iteration will have to compare if the seq is not LEQ with the
        two (or more) elements in MAX (and so on, MAX can grow during iteration)
    """
    if lst == []:
        return []
    stl = lst

    # Sort only sorts wrt the first element on every nested sequence
    # With key=min, the min is found in every sequence and this is sorted
    stl.sort(reverse=True, key=min)

    # First element is the one with the largest among the minimum of every seq
    maxl = [stl[0]]

    for k in range(1, len(stl)):  # find the largest cuboids
        b = True
        # For every element of the sorted list
        el = stl[k]

        # For every element in maxl
        for mx in maxl:
            # If the sorted list sequence is LEQ, set b to False and continue
            # with next seq in the sorted list seq OR:
            # both are smaller, then the voxel is more of a outlier, do not
            # use for making cuboid!
            if LEQ(el, mx):
                b = False
                break
        # If the sorted list sequence is larger than all the seqs in maxl
        # then append it to maxl
        if b:
            maxl.append(el)

    # This will only sort by the first element in every sequence of maxl
    maxl.sort()
    return maxl  # Returns best location to make square


def tolen(lst):
    """
    Converts a list lst  of 0,1 values to the lengths of cosecutive 1's
    to the right up to an element containing a 0
    tolen( [0,0,1,1,1,0,0,1,1,0,1]) ->  [0,0,3,2,1,0,0,2,1,0,1]
    """
    tsl = lst.tolist()
    tsl.reverse()
    n = 0
    ls = []
    for x in tsl:
        if x != 0:
            n += 1
        else:
            n = 0
        ls.append(n)
    ls.reverse()
    return ls


def get_voxel_file(fn):
    """
    Parameters
    ----------
    fn
        Text file with 4 columns: `index, x, y, z`. First row is skipped. The
        data is in integer format, which represent the index and pixel
        coordinates. Column headers can have any name but should be in order.

    Returns
    -------
    A 4-elements tuple: `(min, max, count, zeros_array)` where::

        min         -> minimum of x y z positions
        max         -> maximum of x y z positions
        count       -> number of voxels
        zeros_array -> a 3D matrix with zeros and ones, with the dimensions
                       according to the grain size, e.g. max_x - min_x
                       Values 1 are placed where x y z positions are located
    """
    # TODO: convert to Numpy

    # Using Pandas here
    # tbl = pd.read_table(fn, delim_whitespace=True)
    # Using Numpy:
    tbl = np.loadtxt(fn, skiprows=1,
                     dtype=dict(names=('index', 'x', 'y', 'z'),
                     formats=('i8', 'i8', 'i8', 'i8')))


    # the arr array: X_pos Y_pos Z_pos
    # TODO: remove the names option, the file should be standard with x y z
    # Maybe set the column  number for x y z
    # arr = [tbl[names[0]].to_list(),
    #        tbl[names[1]].to_list(),
    #        tbl[names[2]].to_list()]
    # arr=[tbl['j'].to_list(),tbl['i'].to_list(),tbl['z'].to_list()]

    arr = [list(tbl['x']), list(tbl['y']), list(tbl['z'])]

    # Minimum of x positions, y and z: (computes the minimum for every list
    # inside the arr list, which are the x y z lists)
    mn = list(map(min, arr))
    pos = [
        [a - mn[0] for a in arr[0]],
        [a - mn[1] for a in arr[1]],
        [a - mn[2] for a in arr[2]],
    ]
    mx = list(map(max, arr))
    # Notice: zarr directions, rows are x, cols are y, 3rd dim is z
    zarr = np.zeros((mx[0] - mn[0] + 1, mx[1] - mn[1] + 1, mx[2] - mn[2] + 1))
    ct = 0
    for p in zip(*pos):
        zarr[p] = 1
        ct += 1
    return (mn, mx, ct, zarr)


def findmax(layer, ldim, a, b):
    """
    For a layer (2D array) with dimensions (ldim[0], ldim[1])
    layer is a 2D array with the number of consecutive 1s to the right
    at positions row=a, col=b
    """
    nmax = layer[a][b]
    cnt = 0
    res = []
    # for k from row a until the last row
    for k in range(a, ldim[0]):
        cnt += 1
        # nmax is the min between the actual value and the one in the next row
        nmax = min(nmax, layer[k][b])
        if nmax == 0:
            break
        res.append([nmax, cnt])
    return TupleMaxima(res)


def findmax3(arr, adim, maxdir, a, b, c):
    p = [x.insert(0, 1) for x in arr[a][b][c]]
    res = arr[a][b][c]
    mv = 0
    if res == []:
        return (maxdir, [])
    for r in res:
        # mv -> max volume
        mv = max(mv, r[1] * r[2])
        if mv > maxdir["vol"]:
            maxdir["pt"] = [a, b, c]
            maxdir["dim"] = [1, r[2], r[1]]
            maxdir["vol"] = mv
    if a == adim[0] - 1:
        return (maxdir, res)
    top = arr[a + 1][b][c]
    if top == []:
        return (maxdir, res)
    lst = []
    for pair in res:
        for tup in top:
            mn = [tup[0] + 1, min(pair[1], tup[1]), min(pair[2], tup[2])]
            v = mn[0] * mn[1] * mn[2]
            if v > mv:
                mv = v
                if v > maxdir["vol"]:
                    maxdir["pt"] = [a, b, c]
                    maxdir["dim"] = [mn[0], mn[2], mn[1]]
                    maxdir["vol"] = v
            lst.append(mn[:])
    return (maxdir, TupleMaxima(lst))


def max_cuboid(zarr):
    narr = []
    for lay in zarr:  # loop through outer array: the x array
        lay2 = []
        # for every element/value in lay -> every line in the yx plane
        for ls in lay:  # now going through y coordinates
            # append the array with number of 1s to the right
            lay2.append(tolen(ls))  # vary z coordinates with set x and y
        # Get 3d array (list of lists of lists) with number of voxels to
        # follow inclusive itself:
        narr.append(lay2)  # --> array with number of 1s
    maxdir = {"pt": [-1, -1, -1], "dim": [0, 0, 0], "vol": 0}
    marr = copy.deepcopy(narr)
    dim = zarr.shape
    for i in range(dim[0]):  # x
        for j in range(dim[1]):  # y
            for k in range(dim[2]):  # z
                # find max at every yz layer: narr[i]
                # dim[1:]  -> (ny, nz)
                #
                # findmax returns tuples; its odd to use marr as a list
                # because we can change its contents by anything
                # Input 3d array, size of y/z dimension, which y/z location:
                marr[i][j][k] = findmax(narr[i], dim[1:], j, k)
                # returns 'connectivity' voxels in yz plane
    qarr = copy.deepcopy(marr)
    for i in reversed(range(dim[0])):
        for j in range(dim[1]):
            for k in range(dim[2]):
                # Fit largest cuboid within a region:
                maxdir, qarr[i][j][k] = findmax3(qarr, dim, maxdir, i, j, k)
    maxdim = maxdir["dim"]
    maxpt = maxdir["pt"]
    for i in range(maxdim[0]):
        for j in range(maxdim[1]):
            for k in range(maxdim[2]):
                # Make zarr zero there where cuboid is extracted
                zarr[maxpt[0] + i][maxpt[1] + j][maxpt[2] + k] = 0
    return maxdir


# input minimum coordinates, and location of voxels in 3d matrix
def cuboid_decomposition_method(mn, zarr):
    cublst = []
    totvol = 0
    t0 = time.time()
    while True:
        maxdir = max_cuboid(zarr)  # calls function to find max cuboid
        if maxdir["vol"] > 0:
            maxpt = maxdir["pt"]
            # volume, [min x, max y, max z + min coord], size :
            cublst.append(
                [maxdir["vol"],
                 [maxpt[i] + mn[i] for i in range(3)],
                 maxdir["dim"]])
            totvol += maxdir["vol"]
        else:
            break
    t0 = time.time() - t0
    return (totvol, t0, cublst)


def CuboidDecomposition(fn, fnout, format_output=False):
    """Perform cuboid aggregation decomposition of pixel data from a text file

    Parameters
    ----------
    fn
        Text file with 4 columns: `index, x, y, z`. First row is skipped. The
        data is in integer format, which represent the index and pixel
        coordinates. Column headers can have any name but should be in order.
    fnout
        File name for output CSV text file with cuboid decomposition.
    format_output
        If False, the output file contains the lower and upper corners
        defining the cuboids in pixel coordinates (pixel centers). If True,
        the output contains indexes, the center coordinates of every cuboid
        and the cuboid lenghts in every spatial dimension.
    """
    mn, mx, count, zarr = get_voxel_file(fn)
    print(f'{count} voxels in: {mn}, {mx}')

    totvol, t0, cublst = cuboid_decomposition_method(mn, zarr)
    print("Time used (ms): ", int(1000 * t0 + 0.5))
    if count != totvol:
        print('Error in decomposition')
        raise Exception(f'#Voxels = {count} != Cuboid volume sum = {totvol}')

    with open(fnout, "w", newline="") as f:
        wrt = csv.writer(f)

        if format_output:
            wrt.writerow(["i", "x", "y", "z", "dx", "dy", "dz"])
            for i, cub in enumerate(cublst):
                row = [i + 1]
                r_min = [c - 0.5 for c in cub[1]]
                r_max = [(cub[1][i] + cub[2][i] - 1 + 0.5) for i in range(3)]
                dr = [(r_max[i] - r_min[i]) for i in range(3)]
                r_center = [(r_min[i] + 0.5 * dr[i]) for i in range(3)]

                row.extend(r_center)
                row.extend(dr)
                wrt.writerow(row)
        else:
            wrt.writerow(["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"])
            for cub in cublst:
                # print(cub)
                row = cub[1]
                row.extend([cub[1][i] + cub[2][i] - 1 for i in range(3)])
                wrt.writerow(row)
