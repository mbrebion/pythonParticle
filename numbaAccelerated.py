from random import getrandbits
from numba import jit
import numpy as np


##############################################################
############### Left Right periodic boundaries ###############
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def movePeriodically(xs, rightLimit):
    """
    Ensure particules remains in the [0,rightlimit] x-space.
    :param xs: array containing w coordinates
    :param rightLimit: right limit.
    :return: None
    """
    for i in range(len(xs)):
        if xs[i] > rightLimit or xs[i]<0:
            xs[i] = xs[i] % rightLimit


##############################################################
###################### Particle Swapping #####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def fromAtoB(ia, ib, xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb):
    # swap

    xsb[ib] = xsa[ia]
    ysb[ib] = ysa[ia]
    vxsb[ib] = vxsa[ia]
    vysb[ib] = vysa[ia]
    wheresb[ib] = wheresa[ia]
    colorsb[ib] = colorsa[ia]

    # reset
    vxsa[ia] = 0.
    vysa[ia] = 0.
    wheresa[ia] = 0


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def moveToSwap(xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb, xLim, kindOfLim):
    """
    Move particles from a coordinates to b coordinates (swap if needed) and return the amount of particle moved
    :param xLim: x left or right limit
    :param kindOfLim: if True, limit is at right side, if False, at Left side
    :return: the amount of particle moved
    """
    count = 0
    if kindOfLim:
        for i in range(len(xsa)):
            if xsa[i] > xLim and wheresa[i] != 0:
                fromAtoB(i, count, xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb)
                count += 1
    else:
        for i in range(len(xsa)):
            if xsa[i] < xLim and wheresa[i] != 0:
                fromAtoB(i, count, xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb)
                count += 1

    return count


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def moveSwapToNeighbor(xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb, amount, ymax):
    """
    Move particles from swap a to other cell b
    :param amount: number of particle to be moved
    :param ymax: max y coordinate of cells
    :return: None
    """
    for i in range(amount):
        bestIndex = goodIndexToInsertTo(ysa[i], ysb, wheresb, ymax)
        fromAtoB(i, bestIndex, xsa, ysa, vxsa, vysa, wheresa, colorsa, xsb, ysb, vxsb, vysb, wheresb, colorsb)


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def indicesToSwapRight(xs, xmax, wheres, indices):
    """
    retrieve the indices of particle which must be swapped to right and store therm to indices
    :param xs: x coordinates
    :param xmax: end of cell coordinate
    :param wheres: mask
    :param indices: output array
    :return: Number of particle to swap
    """

    for i in range(len(indices)):
        indices[i] = 0
    newIndex = 0

    for i in range(len(xs)):
        if xs[i] > xmax and wheres[i] != 0:
            indices[newIndex] = i
            newIndex += 1
    return newIndex


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def indicesToSwapLeft(xs, xmin, wheres, indices):
    """
    retrieve the indices of particle which must be swapped to right and store therm to indices
    :param xs: x coordinates
    :param xmin: begin of cell coordinate
    :param wheres: mask
    :param indices: output array
    :return: Number of particle to swap
    """

    for i in range(len(indices)):
        indices[i] = 0
    newIndex = 0

    for i in range(len(xs)):
        if xs[i] < xmin and wheres[i] != 0:
            indices[newIndex] = i
            newIndex += 1
    return newIndex


##############################################################
###################### Sorting algorithm #####################
##############################################################


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def goodIndexToInsertTo(y, ys, wheres, ymax):
    """

    :param y: y of target particle
    :param ys: ys of already present particles
    :param wheres: masks
    :param ymax: max value of y
    :return: the next best index to insert particle at
    """

    index = int(y / ymax * len(ys))  # guess for best index

    while index < len(ys) and ys[index] < y:
        index += 1

    while index < len(ys) and wheres[index] != 0:
        index += 1

    if index >= len(ys):
        # end of the list, search dead particle backward
        index = len(ys) - 1
        while index >= 0 and wheres[index] != 0:
            index -= 1

    return index


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
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


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
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

        value = ys[i]  # assumed to be alive

        swaps = 0
        j = i - 1
        x, y, vx, vy, where, color = xs[i], ys[i], vxs[i], vys[i], wheres[i], colors[i]

        while value < ys[j] and j >= 0:
            roll1(j + 1, xs, ys, vxs, vys, wheres, colors)
            swaps += 1
            j = j - 1

        if swaps > 0:
            xs[j + 1], ys[j + 1], vxs[j + 1], vys[j + 1], wheres[j + 1], colors[j + 1] = x, y, vx, vy, where, color

    return swaps


##############################################################
########### Static and moving wall interraction ##############
##############################################################


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def movingWallInteraction(xs, vxs,vys, wheres, x, v, dt, mp, m):
    """
        update coordinates and velocities of particles interacting left wall (is exists)

        :param xs: array containing x coordinates of particles
        :param vxs: array containing x velocities of particles
        :param vys: array containing x velocities of particles
        :param wheres: mask array
        :param x : x coordinate of wall
        :param v : velocity coordinate of wall
        :param dt : time step
        :param mp : mass of particle
        :param m : mass of wall

        :return: Force applied to wall at both sides
        """
    fpl = 0.
    fpr = 0.
    for i in range(len(xs)):

        # bounce on moving wall must be taken into account more precisely

        if wheres[i] * (xs[i] - x) < 0:
            # particle crossed the wall, in either direction
            delta = -(xs[i] - x) / (v - vxs[i])

            if delta < 0 : # already treated collision
                continue

            # move backward the particle
            xs[i] -= vxs[i] * delta

            # bounce and compute force applied to wall
            # newV = (2 * m * v + (mp - m) * vxs[i]) / (m + mp)

            newV = -  vxs[i] + 2 * v  # infinitely massive wall

            if wheres[i] < 0:
                fpl += - (newV - vxs[i]) * mp / dt
            else:
                fpr += - (newV - vxs[i]) * mp / dt

            vxs[i] = newV
            vys[i] *= (-1)**getrandbits(1)

            # move forward the particle
            xs[i] += vxs[i] * delta

    return fpl, fpr


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInteractionLeft(xs, vxs,vys, wheres, left,dt, m):
    """
        update coordinates and velocities of particles interacting left wall (is exists)

        :param xs: array containing x coordinates of particles
        :param vxs: array containing x velocities of particles
        :param vys: array containing x velocities of particles
        :param wheres: mask array
        :param left: left coordinate of cell
        :param dt: time step for simulation (all collision between particles and walls occurred within dt)
        :param m: mass of particles (all supposed equals)
        return: fleft (computed positively)
        """

    fleft = 0.
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] < left:
            xs[i] = 2 * left - xs[i]
            vxs[i] *= -1
            vys[i] *= (-1) ** getrandbits(1)

            fleft += abs(vxs[i])
    return fleft * 2 * m / dt

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInteractionRight(xs, vxs,vys, wheres, right,dt, m):
    """
        update coordinates and velocities of particles interacting left wall (is exists)

        :param xs: array containing x coordinates of particles
        :param vxs: array containing x velocities of particles
        :param vys: array containing y velocities of particles
        :param wheres: mask array
        :param right: right coordinate of cell
        :param dt: time step for simulation (all collision between particles and walls occurred within dt)
        :param m: mass of particles (all supposed equals)
        return: fright (computed positively)
        """

    fright = 0.
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] > right:
            xs[i] = 2 * right - xs[i]
            vxs[i] *= -1
            vys[i] *= (-1)** getrandbits(1)
            fright += abs(vxs[i])
    return fright * 2 * m / dt


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInterractionUpAndDown(ys,vxs, vys, wheres, width, dt, m,vStar):
    """
    update coordinates and velocities of particles interacting with solid walls (which are horizontal)
    Compute the forces applied to both walls

    :param ys: array containing y coordinates of particles
    :param vxs: array containing y velocities of particles
    :param vys: array containing y velocities of particles
    :param wheres: mask array
    :param width: width of the cell
    :param dt: time step for simulation (all collision between particles and walls occurred within dt)
    :param m: mass of particles (all supposed equals)
    :param vStar: quadratic mean velocity
    :return: fup,fdown, the two forces (computed positively)
    """
    fup, fdown = 0., 0.
    sqtwo = (4/3)**0.5

    for i in range(len(ys)):
        if wheres[i] == 0:
            continue

        bounce = False
        if ys[i] < 0:
            ys[i] *= -1
            bounce = True

        elif ys[i] > width:
            ys[i] = 2 * width - ys[i]
            bounce = True

        if bounce:
            vOld = vys[i]
            if vStar > 0:
                vxs[i] = np.random.normal(0, vStar/sqtwo, 1)[0]
                vys[i] = - abs(np.random.normal(0, vStar/sqtwo, 1)[0]) * vOld/abs(vOld)
            else:
                vxs[i] *= (-1)**getrandbits(1)
                vys[i] *= -1

            fup += abs(vys[i]-vOld)/2  # f = -2 * vy * ms * dn / dt

    return fdown * 2 * m / dt, fup * 2 * m / dt


##############################################################
##########  Average temperature and kinetic energy ###########
##############################################################


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBinsOld(ys,ymax,vxs,wheres,bins):
    """
    :param ys:  y positions
    :param ymax:  max y value
    :param vxs: x velocities
    :param wheres: mask array
    :param bins: list dedicated to store the means
    :return: modified bins
    """
    ns = [0.]*len(bins)
    delta = ymax/(len(bins)-1)
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        k = int(ys[i]/delta)
        alphakp = (ys[i] - k * delta) / delta
        alphak = 1 - alphakp

        bins[k] += vxs[i] * alphak
        ns[k] += alphak
        bins[k+1] += vxs[i] * alphakp
        ns[k+1] += alphakp

    for i in range(len(bins)):
        bins[i] /= max(ns[i],1)
    return bins

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBeforeWall(xs,x,measureSpan,vxs,wheres,bins):
    """

    :param xs: x location of particles
    :param x: x location of wall
    :param measureSpan: x span of measures
    :param vxs: vx velocities
    :param wheres: mask array
    :param bins:
    :return:
    """
    ns = [0]*len(bins)
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        index = int((x-xs[i]) / measureSpan * len(bins))
        if len(bins) > index >= 0:
            bins[index] += vxs[i]
            ns[index] += 1

    for i in range(len(bins)):
        bins[i] /= max(ns[i],1)
    return bins

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBins(ys,ymax,vxs,wheres,bins):
    """
    :param ys:  y positions
    :param ymax:  max y value
    :param vxs: x velocities
    :param wheres: mask array
    :param bins: list dedicated to store the means
    :return: modified bins
    """
    ns = [0]*len(bins)
    for i in range(len(vxs)):
        if wheres[i] >= 0:
            continue
        index = int(ys[i]/ymax * len(bins))
        bins[index]+=vxs[i]
        ns[index] +=1

    for i in range(len(bins)):
        bins[i] /= max(ns[i],1)
    return bins

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocity(vxs,wheres):
    """

    :param vxs: x velocities
    :param wheres: mask array
    :return: X averaged velocity of the cell
    """
    vx = 0.
    n = 0
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        vx += vxs[i]
        n += 1
    return vx/n


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def computeEcs(vxs, vys, wheres, m):
    """
        compute kinetic energy left and right to wall
        :param vxs: x velocities
        :param vys: y velocities
        :param wheres: mask array
        :param m: mass
        :return: left and right kinetic energy
        """
    ecl = 0.
    ecr = 0.
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue

        vc = vxs[i] ** 2 + vys[i] ** 2
        if wheres[i] < 0:
            ecl += m * vc / 2
        else:
            ecr += m * vc / 2
    return ecl, ecr


##############################################################
###################  Collision detection  ####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def detectAllCollisions(xs, ys, vxs, vys, wheres, colors, dt, d, histo, coloringPolicy,xMax):
    """
    This method update particle positions and velocities taking into account elastic collisions
    It first tries to detect efficiently if collisions occurred,
    assuming that particles are already sorted by y coordinate
    And then apply collision rules precisely (real instant of collision is computed and particle are shifted correctly)

    :param xs:
    :param ys:
    :param vxs:
    :param vys:
    :param wheres:
    :param colors:
    :param dt: time step
    :param d: particle diameter
    :param histo: histogram of collisions detection according to j-i for storing purpose
    :param coloringPolicy: (str), colors are updated according to policy
    :param xMax: if >0, collisions are not taken into account when xs[i]>= xMax
    :return: number of collisions
    """
    nbCollide = 0.
    ra = np.array([0., 0.])
    rb = np.array([0., 0.])
    va = np.array([0., 0.])
    vb = np.array([0., 0.])

    ln = len(xs)
    nbSearch = len(histo)
    vxmax = 0
    if xMax < 0:
        xMax = 1e9

    for i in range(ln - 1):  # last particle can't collide to anyone
        if wheres[i] == 0 or xs[i]>= xMax:
            continue
        vxmax = max(vxmax,abs(vxs[i]))

        for j in range(i + 1, min(len(xs), i + nbSearch)):
            if wheres[i] * wheres[j] <= 0 or xs[i] > xMax:  # different side (<0) or one dead particle (==0)
                continue

            coll, t = isCollidingFast(xs[i], ys[i], xs[j], ys[j], vxs[i], vys[i], vxs[j], vys[j], dt, d)

            if coll:
                # record strategy
                histo[j - i] += 1

                if coloringPolicy == "coll":
                    colors[i] = 1
                    colors[j] = 1

                nbCollide += 1.

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

                # re-shift particles
                xs[i] += t * vxs[i]
                ys[i] += t * vys[i]
                xs[j] += t * vxs[j]
                ys[j] += t * vys[j]

    if coloringPolicy == "vx":
        vxmax = max(vxmax,abs(vxs[-1]))
        for i in range(ln ):  # last particle can't collide to anyone
            if wheres[i] == 0 or xs[i] >= xMax:
                continue
            colors[i] = vxs[i]/vxmax

    return nbCollide


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
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


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
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
    if abs(x2 - x1) - d > abs((vx2 - vx1)) * dt:
        return False, 0

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
        if t < 0:
            return False, 0.
        elif t > dt:
            if (x1-x2)**2 + (y1-y2)**2 < d**2 and t < 10*dt:
                # in this case, collision has really occurred, but before this time step.
                # it must be treated.
                return True, t#min(t,4*dt)
            return False, 0.
        else:
            return True, t


##############################################################
#####################  Helper functions  #####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def retieveIndex(id, wheres):
    """
    find present index of particle id in particle arrays
    :param id:
    :param wheres : mask array containing list of indices
    :return: index of particle id
    """
    for i in range(len(wheres)):
        if wheres[i] == id:
            return i
    return -1


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def countAlive(wheres):
    count = 0
    for i in range(len(wheres)):
        if wheres[i] != 0:
            count += 1
    return count


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def countAliveLeft(xs, wheres, x):
    countWhere, countLoc = 0, 0
    for i in range(len(wheres)):
        if wheres[i] != 0 and xs[i] < x:
            countLoc += 1
        if wheres[i] < 0:
            countWhere += 1
    if countLoc != countWhere:
        print("problem ", countLoc, countWhere)
    return countWhere


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def countAliveRight(xs, wheres, x):
    countWhere, countLoc = 0, 0
    for i in range(len(wheres)):
        if wheres[i] != 0 and xs[i] > x:
            countLoc += 1
        if wheres[i] > 0:
            countWhere += 1
    if countLoc != countWhere:
        print("problem ", countLoc, countWhere)
    return countWhere


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def checkCorrectSide(wheres, xs, x):
    for i in range(len(xs)):
        if wheres[i] * (xs[i] - x) < 0:
            print("      ", wheres[i], xs[i], x)


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def norm(a):
    return dot(a, a) ** 0.5


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
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


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def advect(xs, ys, vxs, vys, dt,forcex,mass):
    """
    Use Verlet method to update velocities and positions
    :param xs: x position
    :param ys:
    :param vxs: x velocity
    :param vys:
    :param dt: time step
    :param forcex: horizontal force to be applied to all particles
    :param mass: mass of particles
    :return:
    """
    dec = forcex*dt/mass
    for i in range(len(xs)):
        vxs[i] += dec
        xs[i] += vxs[i] * dt
        ys[i] += vys[i] * dt

