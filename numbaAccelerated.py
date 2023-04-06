import numba
from numba import jit, prange
import numpy as np


def prettyPrint(values, mask):
    out = ""
    for i in range(len(values)):
        if mask[i] == 0:
            out += "  _  "
        else:
            out += " " + str(values[i])[:4]
    return out


##############################################################
###################### Sorting algorithm #####################
##############################################################

@jit(nopython=True, cache=True)
def roll1(i, ar1, ar2, ar3, ar4, ar5, ar6):
    """
    set values formerly at index i-1 to index i for 5 arrays
    :param i:  final position for values from index i-1
    :param ar1: first array modified
    :param ar2:
    :param ar3:
    :param ar4:
    :param ar5:
    :param ar6:
    :return: None
    """
    ar1[i] = ar1[i - 1]
    ar2[i] = ar2[i - 1]
    ar3[i] = ar3[i - 1]
    ar4[i] = ar4[i - 1]
    ar5[i] = ar5[i - 1]
    ar6[i] = ar6[i - 1]


@jit(nopython=True, cache=True)
def sortCell(xs, ys, vxs, vys, wheres, colors):
    """
    Sort 5 arrays according to ys (second array provided) with insertion algorithm (fastest for quite already sorted arrays)
    :param xs: first array
    :param ys: second array which content is used for sorting
    :param vxs:
    :param vys:
    :param wheres: mask array
    :param colors: color array
    :return: Amount of swaps
    """

    for i in range(1, len(ys)):

        if wheres[i] == 0:
            # we do not sort a dead value
            continue

        value = ys[i]  # assumed to be alive

        swaps = 0
        j = i - 1
        x, y, vx, vy, where, color = xs[i], ys[i], vxs[i], vys[i], wheres[i], colors[i]

        while (value < ys[j] or wheres[j] == 0) and j >= 0:
            roll1(j + 1, xs, ys, vxs, vys, wheres, colors)
            swaps += 1
            j = j - 1

        if swaps > 0:
            xs[j + 1], ys[j + 1], vxs[j + 1], vys[j + 1], wheres[j + 1], colors[j + 1] = x, y, vx, vy, where, color

    return swaps


##############################################################
################## Static wall interraction ##################
##############################################################


@jit(nopython=True, cache=True, fastmath=True)
def staticWallInterraction(ys, vys, wheres, width, dt, m):
    """
    update coordinates and velocities of particles interacting with solid walls (which are horizontal)
    Compute the forces applied to both walls

    :param ys: array containing y coordinates of particles
    :param vys: array containing y velocities of particles
    :param wheres: mask array
    :param width: width if the cell
    :param dt: time step for simulation (all collision between particles and walls occurred within dt)
    :param m: mass of particles (all supposed equals)
    :return: fup,fdown, the two forces (computed positively)
    """
    fup, fdown = 0., 0.

    for i in range(len(ys)):
        if wheres[i] == 0:
            continue

        if ys[i] < 0:
            ys[i] = -ys[i]
            vys[i] = - vys[i]
            fdown += abs(vys[i])  # f = 2 * vy * m * dn / dt

        if ys[i] > width:
            ys[i] = 2 * width - ys[i]
            vys[i] = - vys[i]
            fup += abs(vys[i])  # f = -2 * vy * ms * dn / dt

    return fdown * 2 * m / dt, fup * 2 * m / dt


##############################################################
###################  Average temperature  ####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True)
def computeAverageTemperature(vxs, vys, wheres, m, kb):
    """
    compute average temperature in cell
    :param vxs: x velocities
    :param vys: y velocities
    :param wheres: mask array

    :param m: mass
    :param kb: boltzmann constant
    :return: averaged temperature
    """
    vc = 0
    n = 0
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        n += 1
        vc += vxs[i] ** 2 + vys[i] ** 2
    return m * vc / (2 * kb * n)


##############################################################
###################  Collision detection  ####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True)
def detectAllCollisions(xs, ys, vxs, vys, wheres, colors, dt, d, nbSearch):
    nbCollide = 0.
    colors[:] = 0.
    ra = np.array([0., 0.])
    rb = np.array([0., 0.])
    va = np.array([0., 0.])
    vb = np.array([0., 0.])
    for i in prange(len(xs)):
        if wheres[i] == 0:
            continue

        for j in range(i, min(len(xs), i + nbSearch)):
            if wheres[j] == 0 or j == i:
                continue
            coll, t = isCollidingFast(xs[i], ys[i], xs[j], ys[j], vxs[i], vys[i], vxs[j], vys[j], dt, d)
            if coll:
                # update of colors (for display purpose)
                colors[i] = 0.5
                colors[j] = 0.5

                # restore locations before collision
                xs[i] -= t * vxs[i]
                ys[i] -= t * vys[i]
                xs[j] -= t * vxs[j]
                ys[j] -= t * vys[j]

                # update velocities
                ra[0], ra[1] = xs[i], ys[i]
                rb[0], rb[1] = xs[j], ys[j]
                va[0], va[1] = vxs[i], vys[i]
                vb[0], vb[1] = vxs[j], vys[j]
                vfa, vfb = bounce(ra, rb, va, vb)
                vxs[i], vys[i] = vfa[0], vfa[1]
                vxs[j], vys[j] = vfb[0], vfb[1]

                # remove particles
                xs[i] += t * vxs[i]
                ys[i] += t * vys[i]
                xs[j] += t * vxs[j]
                ys[j] += t * vys[j]
                nbCollide += 1.

    return nbCollide


@jit(nopython=True, cache=True, fastmath=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


@jit(nopython=True, cache=True, fastmath=True)
def bounce(ra, rb, va, vb):
    """
    compute accurate bounce between particles
    :param ra: location (2D) of first part at the very moment of collision
    :param rb: location (2D) of second part at the very moment of collision
    :param va: velocity of first particle
    :param vb: velocity of second particle
    :return: vaf,vbf, velocities after impact
    """
    ab = rb - ra
    dotn = dot((vb - va), ab) * ab / dot(ab, ab)

    vaf = (va + dotn)
    vbf = (vb - dotn)

    return vaf, vbf


@jit(nopython=True, cache=True, fastmath=True)
def isCollidingFast(x1, y1, x2, y2, vx1, vy1, vx2, vy2, dt, d):
    """
    Detect whether two particles have collided between t-dt and t
    :param x1: x coord of first particle at time t
    :param y1: y coord of first particle
    :param x2: x coord of second particle
    :param y2: y coord of second particle
    :param vx1: x velocity of first particle
    :param vy1:
    :param vx2:
    :param vy2:
    :param dt: time step
    :param d: circle's diameters
    :return: True if collision occurred, (and its time to remove), else False (and 0 time)
    """
    #if abs(x2-x1) - d > abs((vx2 - vx1))*dt:
    #    return False,0

    dvc = (vx2 - vx1) ** 2 + (vy2 - vy1) ** 2
    drc = (x2 - x1) ** 2 + (y2 - y1) ** 2

    scal = (x2 - x1) * (vx2 - vx1) + (y2 - y1) * (vy2 - vy1)
    a = dvc
    b = - 2 * scal
    c = drc - d ** 2
    delta = b ** 2 - 4 * a * c
    if delta < 0:

        return False, 0.
    if delta >= 0:
        t = (-b + delta ** 0.5) / (2 * a)
        if t < 0 or t > dt:

            return False, 0.
        else:

            return True, t


##############################################################
#################  helper for openglView  ####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True)
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
