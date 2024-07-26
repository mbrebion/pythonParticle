from numba import jit
from numbaAccelerated import isCollidingFast, dealWithCollision



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def ComputeXVelocityBinsOld(ys, ymax, vxs, wheres, bins):
    """
    :param ys:  y positions
    :param ymax:  max y value
    :param vxs: x velocities
    :param wheres: mask array
    :param bins: list dedicated to store the means
    :return: modified bins
    """
    ns = [0.] * len(bins)
    delta = ymax / (len(bins) - 1)
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue
        k = int(ys[i] / delta)
        alphakp = (ys[i] - k * delta) / delta
        alphak = 1 - alphakp

        bins[k] += vxs[i] * alphak
        ns[k] += alphak
        bins[k + 1] += vxs[i] * alphakp
        ns[k + 1] += alphakp

    for i in range(len(bins)):
        bins[i] /= max(ns[i], 1)
    return bins







@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def detectAllCollisionsOLD(xs, ys, vxs, vys, wheres, colors, indicesLeft, indicesRight, dt, d, histo, coloringPolicy,
                        left, right):
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
    :param indicesLeft: arrays where indices of particles which are at left side of cell are stored
    :param indicesRight: arrays where indices of particles which are at right side of cell are stored
    :param dt: time step
    :param d: particle diameter
    :param histo: histogram of collisions detection according to j-i for storing purpose
    :param coloringPolicy: (str), colors updated according to policy
    :param left : left coord of cell
    :param right : right coord of cell

    :return: number of collisions
    """
    nbCollide = 0.

    ln = len(xs)
    nbSearch = len(histo)
    vxmax = 0
    leftIndex = 0
    rightIndex = 0

    for i in range(ln - 1):  # last particle can't collide to anyone
        if wheres[i] == 0:
            continue
        vxmax = max(vxmax, abs(vxs[i]))

        # detect indices of particles closes to sides of cell
        both = 0
        if xs[i] < left + abs(vxs[i] * dt)*1.5 + d*1.2:   # better use a less restrictive criteria
            indicesLeft[leftIndex] = i
            leftIndex += 1
            both +=1

        if xs[i] > right - abs(vxs[i] * dt)*1.5 - d*1.2:   # same comment
            indicesRight[rightIndex] = i
            rightIndex += 1
            both += 1
        if both == 2:
            print("particle in both neighborhoods")

        for j in range(i + 1, min(len(xs), i + nbSearch)):
            if wheres[i] * wheres[j] <= 0:  # different side (<0) or one dead particle (==0)
                continue

            coll, t = isCollidingFast(xs[i], ys[i], xs[j], ys[j], vxs[i], vys[i], vxs[j], vys[j], dt, d)

            if coll:
                # record strategy
                histo[j - i] += 1

                if coloringPolicy == "coll":
                    colors[i] = 1
                    colors[j] = 1

                nbCollide += 1.
                dealWithCollision(xs, ys, vxs, vys, xs, ys, vxs, vys, i, j, t)

    if coloringPolicy == "vx":
        vxmax = max(vxmax, abs(vxs[-1]))
        for i in range(ln):
            if wheres[i] == 0:
                continue
            colors[i] = vxs[i] / vxmax

    return nbCollide



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def detectAllCollisionsUnsafe(xs, ys, vxs, vys, wheres, colors, indicesLeft, indicesRight, dt, d, histo, coloringPolicy,
                        left, right):
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
    :param indicesLeft: arrays where indices of particles which are at left side of cell are stored
    :param indicesRight: arrays where indices of particles which are at right side of cell are stored
    :param dt: time step
    :param d: particle diameter
    :param histo: not used in this algorithm ; stays here to mimic behavior of old function
    :param coloringPolicy: (str), colors updated according to policy
    :param left : left coord of cell
    :param right : right coord of cell

    :return: number of collisions
    """
    nbCollide = 0
    secureSearchCoeff = 1.1

    ln = len(xs)
    leftIndex = 0
    rightIndex = 0


    vymax = 0
    for i in range(ln):
        if wheres[i] == 0: # ghost particle
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
        x,y,vx,vy,w = xs[i], ys[i], vxs[i], vys[i],wheres[i] # read once !

        for j in range(i + 1, ln):
            if w * wheres[j] <= 0:  # different side (<0) or one dead particle (==0)
                continue

            # The following is very important, this criterion is used to stop the searches for part i and step to part
            # i+1.  If too restrictive, collisions might be missed. If too broad, computation time may increase
            if abs(ys[j]-y)  > ( (-vy + vymax) * dt + d)*secureSearchCoeff:
                break  # if here, other particles are too high to be able to interact with particle i


            coll, t = isCollidingFast(x, y, xs[j], ys[j], vx, vy, vxs[j], vys[j], dt, d)

            if coll:
                nbCollide += 1.
                dealWithCollision(xs, ys, vxs, vys, xs, ys, vxs, vys, i, j, t)

    return nbCollide



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def isCollidingFastOLD(x1, y1, x2, y2, vx1, vy1, vx2, vy2, dt, d):
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
            return False, 0.

        elif t > dt :
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < d ** 2 :                      # TODO : is this correct ?
                # in this case, collision has really occurred as one particle is inside anotherone, but before this time step.
                # it must be treated.
                return True, t
            return False, 0.

        else:
            return True, t
