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
        if xs[i] > rightLimit or xs[i] < 0:
            xs[i] = xs[i] % rightLimit


##############################################################
###################### Particle Swapping #####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)

def fromAtoB(ia, ib, xsa, ysa, vxsa, vysa, wheresa, lastColla, xsb, ysb, vxsb, vysb, wheresb, lastCollb):
    # swap

    xsb[ib] = xsa[ia]
    ysb[ib] = ysa[ia]
    vxsb[ib] = vxsa[ia]
    vysb[ib] = vysa[ia]
    wheresb[ib] = wheresa[ia]
    lastCollb[ib] = lastColla[ia]

    # reset
    vxsa[ia] = 0.
    vysa[ia] = 0.
    #ysa is not reset as still used for sorting purposes.
    # velocities are set to 0 so that y and x stay constants
    wheresa[ia] = 0


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def moveToSwap(xsa, ysa, vxsa, vysa, wheresa, lastColla, xsb, ysb, vxsb, vysb, wheresb, lastCollb, xLim, kindOfLim):
    """
    Move particles from a coordinates to b coordinates (swap if needed) and return the amount of particle moved
    :param xLim: x left or right limit
    :param kindOfLim: if True, limit is at right side, if False, at Left side
    :return: the amount of particle moved
    """
    count = 0
    for i in range(len(xsa)):
        if wheresa[i] != 0 and (xsa[i] > xLim) == kindOfLim:
            fromAtoB(i, count, xsa, ysa, vxsa, vysa, wheresa, lastColla, xsb, ysb, vxsb, vysb, wheresb, lastCollb)
            count += 1

    return count


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def moveSwapToNeighbor(xsa, ysa, vxsa, vysa, wheresa, lastColla, xsb, ysb, vxsb, vysb, wheresb, lastCollb, amount, ymax):
    """
    Move particles from swap a to other cell b
    :param amount: number of particle to be moved
    :param ymax: max y coordinate of cells
    :return: None
    """
    for i in range(amount):
        bestIndex = goodIndexToInsertTo(ysa[i], ysb, wheresb, ymax)
        fromAtoB(i, bestIndex, xsa, ysa, vxsa, vysa, wheresa, lastColla, xsb, ysb, vxsb, vysb, wheresb, lastCollb)


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def indicesToSwapRight(xs, xmax, wheres, indices):
    """
    retrieve the indices of particle which must be swapped to right and store them to indices
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
    while index >0 and ys[index] > y:
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


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def roll1(i, ar1, ar2, ar3, ar4, ar5, ar6):
    """
    set values formerly at index i-1 to index i for 6 arrays
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
def sortCell(xs, ys, vxs, vys, wheres, lastColls):
    """
    Sort 6 arrays according to ys (second array provided) with insertion algorithm (fastest for quite already sorted arrays)
    :param xs: first array
    :param ys: second array which content is used for sorting
    :param vxs:
    :param vys:
    :param wheres: mask array
    :param colors: color array
    :return: Amount of swaps
    """

    for i in range(1, len(ys)):

        value = ys[i]  # sort both alive and dead particles to keep dead indices equally spaced ?

        swaps = 0
        j = i - 1
        x, y, vx, vy, where, lastColl = xs[i], ys[i], vxs[i], vys[i], wheres[i], lastColls[i]

        while value < ys[j] and j >= 0:
            roll1(j + 1, xs, ys, vxs, vys, wheres, lastColls)
            swaps += 1
            j = j - 1

        if swaps > 0:
            xs[j + 1], ys[j + 1], vxs[j + 1], vys[j + 1], wheres[j + 1], lastColls[j + 1] = x, y, vx, vy, where, lastColl

    return swaps


##############################################################
########### Static and moving wall interraction ##############
##############################################################


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def movingWallInteraction(xs, vxs, vys, wheres,lastColls, x, v, dt, mp, m,time):
    """
    update coordinates and velocities of particles interacting left wall (if exists)
    The exact time (if wall velocity is constant) is first computed ; then the particle is correctly bounced back

    :param xs: array containing x coordinates of particles
    :param vxs: array containing x velocities of particles
    :param vys: array containing x velocities of particles
    :param wheres: mask array
    :param lastColls: last collision array
    :param x : x coordinate of wall
    :param v : velocity coordinate of wall
    :param dt : time step
    :param mp : mass of particle
    :param m : mass of wall
    :param time: current time
    :return: Force applied to wall at both sides
    """

    fpl = 0.
    fpr = 0.
    for i in range(len(xs)):

        if wheres[i] * (xs[i] - x) < 0:
            # particle crossed the wall, in either direction
            delta = -(xs[i] - x) / (v - vxs[i])

            if delta < 0:  # already treated collision
                continue

            # move backward the particle

            xs[i] -= vxs[i] * delta
            lastColls[i] = time - delta

            # bounce and compute force applied to wall
            # newV = (2 * m * v + (mp - m) * vxs[i]) / (m + mp)

            newV = - vxs[i] + 2 * v  # infinitely massive wall

            if wheres[i] < 0:
                fpl += - (newV - vxs[i]) * mp / dt
            else:
                fpr += - (newV - vxs[i]) * mp / dt

            vxs[i] = newV
            #vys[i] *= (-1) ** getrandbits(1)

            # move forward the particle
            xs[i] += vxs[i] * delta

    return fpl, fpr


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInteractionLeft(xs, vxs, vys, wheres,lastColls, left, dt, m,time):
    """
        update coordinates and velocities of particles interacting with left static wall (if exists)

        :param xs: array containing x coordinates of particles
        :param vxs: array containing x velocities of particles
        :param vys: array containing x velocities of particles
        :param wheres: mask array
        :param lastColls: last collision array
        :param left: left coordinate of cell
        :param dt: time step for simulation (all collision between particles and walls occurred within dt)
        :param m: mass of particles (all supposed equals)
        :param time: current time
        return: fleft (computed positively)
        """

    fleft = 0.
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] < left:
            odt = abs( (xs[i] - left) / vxs[i])
            lastColls[i] = time - odt
            xs[i] = 2 * left - xs[i]
            vxs[i] *= -1
            #vys[i] *= (-1) ** getrandbits(1)

            fleft += abs(vxs[i])
    return fleft * 2 * m / dt


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInteractionRight(xs, vxs, vys, wheres, lastColls, right, dt, m,time):
    """
        update coordinates and velocities of particles interacting left wall (if exists)

        :param xs: array containing x coordinates of particles
        :param vxs: array containing x velocities of particles
        :param vys: array containing y velocities of particles
        :param wheres: mask array
        :param lastColls: last collision array
        :param right: right coordinate of cell
        :param dt: time step for simulation (all collision between particles and walls occurred within dt)
        :param m: mass of particles (all supposed equals)
        :param time: current time
        return: fright (computed positively)
        """

    fright = 0.

    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] > right:
            odt = abs( (xs[i] - right) / vxs[i])
            lastColls[i] = time - odt
            xs[i] = 2 * right - xs[i]
            vxs[i] *= -1
            #vys[i] *= (-1) ** getrandbits(1)
            fright += abs(vxs[i])
    return fright * 2 * m / dt


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def staticWallInterractionUpAndDown(ys, vxs, vys, wheres,lastColls, width, dt, m, vStar,time):
    """
    update coordinates and velocities of particles interacting with solid walls (which are horizontal)
    Compute the forces applied to both walls

    :param ys: array containing y coordinates of particles
    :param vxs: array containing y velocities of particles
    :param vys: array containing y velocities of particles
    :param wheres: mask array
    :param lastColls: last collision array
    :param width: width of the cell
    :param dt: time step for simulation (all collision between particles and walls occurred within dt)
    :param m: mass of particles (all supposed equals)
    :param vStar: quadratic mean velocity
    :param time: current time
    :return: fup,fdown, the two forces (computed positively)
    """
    fup, fdown = 0., 0.
    sqtwo = (4 / 3) ** 0.5
    odt = 0.

    for i in range(len(ys)):
        if wheres[i] == 0:
            continue

        bounce = False
        if ys[i] < 0:

            ys[i] *= -1
            odt = abs(ys[i] / vys[i])
            bounce = True

        elif ys[i] > width:
            odt = abs((ys[i]-width) / vys[i])
            ys[i] = 2 * width - ys[i]
            bounce = True

        if bounce:
            vOld = vys[i]
            lastColls[i] = time - odt # time of particle/wall collision must be computed
            if vStar > 0:
                vxs[i] = np.random.normal(0, vStar / sqtwo, 1)[0]
                vys[i] = - abs(np.random.normal(0, vStar / sqtwo, 1)[0]) * vOld / abs(vOld)
            else:
                #vxs[i] *= (-1) ** getrandbits(1)  # used for PRA (velocity reversal cause rugous wall behavior)
                vys[i] *= -1

            fup += abs(vys[i] - vOld) / 2  # f = -2 * vy * ms * dn / dt

    return fdown * 2 * m / dt, fup * 2 * m / dt


##############################################################
##########  Average temperature and kinetic energy ###########
##############################################################





@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBeforeWall(xs, x, measureSpan, vxs, wheres, bins, counts):
    """

    :param xs: x location of particles
    :param x: x location of wall
    :param measureSpan: x span of measures
    :param vxs: vx velocities
    :param wheres: mask array
    :param bins:
    :return:
    """
    ns = [0] * len(bins)
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        index = int((x - xs[i]) / measureSpan * len(bins))
        if len(bins) > index >= 0:
            bins[index] += vxs[i]
            counts[index] += 1

    return bins, counts


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBins(ys, ymax, vxs, wheres, bins):
    """
    :param ys:  y positions
    :param ymax:  max y value
    :param vxs: x velocities
    :param wheres: mask array
    :param bins: list dedicated to store the means
    :return: modified bins
    """
    ns = [0] * len(bins)
    for i in range(len(vxs)):
        if wheres[i] >= 0:
            continue
        index = int(ys[i] / ymax * len(bins))
        bins[index] += vxs[i]
        ns[index] += 1

    for i in range(len(bins)):
        bins[i] /= max(ns[i], 1)
    return bins


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def computeSumVelocityLeftOfWall(xs, vxs, wheres, wallLocation):
    """
    :param xs : x locations
    :param vxs: x velocities
    :param wheres: mask array
    :param wallLocation: x wall location
    :return: sum of velocities left of wall
    """
    sm = 0.
    for i in range(len(vxs)):
        if wheres[i] == 0 or xs[i] > wallLocation:
            continue
        sm += vxs[i]
    return sm


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocity(vxs, wheres):
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
    return vx / n


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
def detectCollisionsAtInterface(indices, nindices, xs, ys, vxs, vys, wheres, lastColls, nxs, nys, nvxs, nvys, nwheres,
                                nlastColls, dt, d, time):
    """
        This method update particle positions and velocities taking into account elastic collisions
        It is focused on collisions occurring between particles from different cells only.
        Collisions between particles from the same cell are handled before those, in another function
        It is here assumed that particles' lists are sorted according to y value,
        even if some collisions between particles within the same cell might change a little the order

        coords variable with n refers to the other cell
    """

    nbCollide = 0
    i = 0
    otherFirst = 0
    maxDeltaY = 0.

    while indices[i] >= 0:
        # index i refers to particles index in indices (particles closed to boundaries)

        ii = indices[i]  # ii refers to real particle index
        nextY = ys[indices[i + 1]]

        for j in range(otherFirst, len(nindices)):

            if nindices[j] < 0:
                break  # no more particles in other side
            jj = nindices[j]
            if wheres[ii] * nwheres[jj] <= 0:  # different side (<0) or one dead particle (==0)
                continue

            maxDeltaY = max(maxDeltaY, (abs(vys[ii]) + abs(nvys[jj])) * dt + d  )
            deltay = nys[jj] - ys[ii]  # y spacing between ii and jj particle

            if nys[jj] < nextY - maxDeltaY:
                otherFirst = j

            if abs(deltay) < maxDeltaY:
                coll, odt = isCollidingFast(xs[ii], ys[ii], nxs[jj], nys[jj], vxs[ii], vys[ii], nvxs[jj], nvys[jj], dt, d, lastColls[ii], nlastColls[jj], time)
                if coll:
                    nbCollide += 1.
                    lastColls[ii] = time - odt
                    nlastColls[jj] = time - odt
                    dealWithCollision(xs, ys, vxs, vys, nxs, nys, nvxs, nvys, ii, jj, odt)
            else:
                if deltay > maxDeltaY:
                    break

        i = i + 1

    indices *= 0
    nindices *= 0
    indices += -1
    nindices += -1

    return nbCollide


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def dealWithCollision(xs, ys, vxs, vys, nxs, nys, nvxs, nvys,i,ni,t):
    """
    variables starting by n concern second particle, which may be located on different arrays (collisions at interface)
    """
    # restore locations before collision
    xs[i] -= t * vxs[i]
    ys[i] -= t * vys[i]
    nxs[ni] -= t * nvxs[ni]
    nys[ni] -= t * nvys[ni]

    ra = np.array([0., 0.])
    rb = np.array([0., 0.])
    va = np.array([0., 0.])
    vb = np.array([0., 0.])

    # update velocities
    ra[0], ra[1] = xs[i], ys[i]
    rb[0], rb[1] = nxs[ni], nys[ni]
    va[0], va[1] = vxs[i], vys[i]
    vb[0], vb[1] = nvxs[ni], nvys[ni]

    vfa, vfb = bounce(ra, rb, va, vb)

    vxs[i], vys[i] = vfa[0], vfa[1]
    nvxs[ni], nvys[ni] = vfb[0], vfb[1]

    # re-shift particles
    xs[i] += t * vxs[i]
    ys[i] += t * vys[i]
    nxs[ni] += t * nvxs[ni]
    nys[ni] += t * nvys[ni]



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def detectAllCollisions(xs, ys, vxs, vys, wheres, lastColls, indicesLeft, indicesRight, dt, d, histo, coloringPolicy,
                        left, right, time):
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
    :param lastColls: used to store the last time of collision
    :param indicesLeft: arrays where indices of particles which are at left side of cell are stored
    :param indicesRight: arrays where indices of particles which are at right side of cell are stored
    :param dt: time step
    :param d: particle diameter
    :param histo: not used in this algorithm ; stays here to mimic behavior of old function
    :param coloringPolicy: (str), colors updated according to policy
    :param left : left coord of cell
    :param right : right coord of cell
    :param time : current time

    :return: number of collisions
    """
    nbCollide = 0
    secureSearchCoeff = 1.1

    ln = len(xs)
    leftIndex = 0
    rightIndex = 0

    vymax = 0
    for i in range(ln):
        if wheres[i] == 0:  # ghost particle
            continue

        # searching vymax :
        vymax = max(vymax,abs(vys[i]))

        # detect indices of particles closes to sides of cell
        both = 0
        if xs[i] < left + (abs(vxs[i] * dt) + d) * secureSearchCoeff:  # better use a less restrictive criteria
            indicesLeft[leftIndex] = i
            leftIndex += 1
            both += 1

        if xs[i] > right - (abs(vxs[i] * dt) + d) * secureSearchCoeff:  # same comment
            indicesRight[rightIndex] = i
            rightIndex += 1
            both += 1
        if both == 2:
            print("particle in both neighborhoods : cells too thins")

    # searching all possible collisions :
    for i in range(ln - 1):  # last particle can't collide to anyone else

        if wheres[i] == 0:  # ghost particle
            continue

        x, y, vx, vy, w, lc = xs[i], ys[i], vxs[i], vys[i],wheres[i], lastColls[i] # read once !

        for j in range(i + 1, ln):
            if w * wheres[j] <= 0:  # different side (<0) or one dead particle (==0)
                # or particle which has already collided
                continue

            # The following is very important, this criterion is used to stop the searches for part i and step to part
            # i+1.  If too restrictive, collisions might be missed. If too broad, computation time may increase
            if abs(ys[j]-y) > ((-vy + vymax) * dt + d)*secureSearchCoeff:
                break  # if here, other particles are too high to be able to interact with particle i

            coll, odt = isCollidingFast(x, y, xs[j], ys[j], vx, vy, vxs[j], vys[j], dt, d, lc, lastColls[j],time)

            if coll:
                nbCollide += 1.
                dealWithCollision(xs, ys, vxs, vys, xs, ys, vxs, vys, i, j, odt)
                lastColls[i] = time - odt
                lastColls[j] = time - odt
                continue



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
def isCollidingFast(x1, y1, x2, y2, vx1, vy1, vx2, vy2, dt, d, lc1, lc2,time):
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
    :param lc1: last collision of particle 1
    :param lc2: last collision of particle 2
    :param time: current time
    :return: True if collision occurred, (and its time to remove), else False (and 0 time)
    """
    if abs(x2 - x1) - d > abs(vx2 - vx1) * dt or abs(y2 - y1) - d > abs(vy2 - vy1) * dt:
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
            # collision in the future
            return False, 0.

        elif t > dt or time-t < lc1 or time-t < lc2:
            # collision occurred more than a time step ago, or collision occurred before last collision for i or j
            # last collision might either be with particle or wall
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < d ** 2:
                # in this case, collision has really occurred as one particle is inside anotherone, but before this time step.
                # it must be treated.
                return True, t
            return False, 0.

        else:
            return True, t


##############################################################
#####################  Helper functions  #####################
##############################################################

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def retrieveIndexs(mid, wheres,start):
    """
    find present index of particle id in particle arrays
    :param mid:
    :param wheres : mask array containing list of indices
    :return: index of particle id
    """

    ln = len(wheres)
    rs = start-int(ln/500)
    for i in range(ln):
        index = (rs+i) % ln
        if wheres[index] == mid:
            return index
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
    if countLoc != countWhere and False:
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
    if countLoc != countWhere and False:
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
def twoArraysToTwo(x, y, mask, colors, sign, xForDisplay, yForDisplay):
    """
    remove dead particles for display arrays
    :param x: 1D numpy array (n)
    :param y: 1D numpy array (n)
    :param mask: 1D numpy array (n) containing int
    :param colors: 1D numpy array (n) colors (float < or >0)
    :param sign: only particle with colors of sign sign(float) are outputed
    :param xForDisplay: array in which x coordinates are stored for screen display purpose
    :param yForDisplay: same for y axis

    :return: None
    """
    j = 0
    while xForDisplay[j] > -1000:
        j += 1

    for i in range(len(x)):
        if mask[i] and colors[i] * sign > 0:
            xForDisplay[j] = x[i]
            yForDisplay[j] = y[i]
            j += 1


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def advect(xs, ys, vxs, vys, dt, forcex, mass):
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
    dec = forcex * dt / mass
    for i in range(len(xs)):
        vxs[i] += dec
        xs[i] += vxs[i] * dt
        ys[i] += vys[i] * dt
