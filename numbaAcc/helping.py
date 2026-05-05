from random import getrandbits
from numba import njit,config
import numpy as np
import time

#config.DISABLE_JIT = True


@njit( cache=True, fastmath=True, nogil=True)
def _goodIndexToInsertTo(y, ys, wheres, ymax):
    """
    This functions tries to find the best index for incoming particle in the existing particle list according to its y coordinate
    :param y: y of target particle
    :param ys: ys of already present particles
    :param wheres: masks
    :param ymax: max value of y
    :return: the next best index to insert particle at
    """

    # first guess for best index according to uniform distribution
    index = int(y / ymax * len(ys))

    # then find index in arrays with closest y ; should be fast if guess is correct
    while index > 0 and ys[index] > y:
        index -= 1
    while index < len(ys) and ys[index] < y:
        index += 1

    # then find the closest empty index
    while index < len(ys) and wheres[index] != 0:
        index += 1

    # if no index found : search backward
    if index >= len(ys):
        # end of the list, search dead particle backward
        index = len(ys) - 1
        while index >= 0 and wheres[index] != 0:
            index -= 1
    return index



@njit( cache=True, fastmath=True, nogil=True)
def movePeriodically(crd, left, right):
    """
    Ensure particules remains in the [left,right] x-space.
    :param: crd: all coordinates of particles in cell
    :param: left: limit of cell
    :param: right: limit of cell
    :return: None
    """
    xs, wheres = crd["xs"], crd["wheres"]
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] > right or xs[i] < left:
            xs[i] =  left +  (xs[i]-left) % (right-left)




@njit( cache=True, fastmath=True, nogil=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


@njit( cache=True, fastmath=True, nogil=True)
def norm(a):
    return _dot(a, a) ** 0.5


@njit( cache=True, fastmath=True, nogil=True)
def twoArraysToOne(x, y, mask, positions):
    """
    update content of positions according to coordinates stores in x1 and y1
    :param x: 1D numpy array (n)
    :param y: 1D numpy array (n)
    :param mask: 1D numpy array (n) containing int
    :param positions: 2D numpy array (n,2)
    :return: None
    """
    for i in range(len(x)):
        if mask[i]:
            positions[i, 0], positions[i, 1] = x[i], y[i]
        else:
            positions[i, 0], positions[i, 1] = -10, -10


@njit( cache=True, fastmath=True, nogil=True)
def twoArraysToTwo(x, y, mask, xForDisplay, yForDisplay):
    """
    remove dead particles for display arrays
    :param x: 1D numpy array (n)
    :param y: 1D numpy array (n)
    :param mask: 1D numpy array (n) containing int
    :param xForDisplay: array in which x coordinates are stored for screen display purpose
    :param yForDisplay: same for y axis

    :return: None
    """
    j = 0
    while xForDisplay[j] > -1000:
        j += 1

    for i in range(len(x)):
        if mask[i] :
            xForDisplay[j] = x[i]
            yForDisplay[j] = y[i]
            j += 1


@njit( cache=True, fastmath=True, nogil=True)
def resetToMinusOne(array):
    for i in range(len(array)):
        if array[i] < 0 :
            break
        array[i] = -1


@njit( cache=True, fastmath=True, nogil=True)
def stream(crd,csts):
    """
    Use velocity Verlet method to update velocities and positions
    :param crd: all coordinates
    :param csts: all constants
    :return: nothing
    """
    forcex, dt, mass = csts["forceX"], csts["dt"], csts["ms"]
    vxs, vys,xs,ys = crd["vxs"],crd["vys"],crd["xs"],crd["ys"]
    dec = forcex * dt / mass #  null or constant
    for i in range(len(xs)):
        vxs[i] += dec
        xs[i] += vxs[i] * dt + dec * dt / 2
        ys[i] += vys[i] * dt + dec * dt / 2

