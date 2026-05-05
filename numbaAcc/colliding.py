from random import getrandbits
from numba import njit
import numpy as np
from numbaAcc.helping import _dot

#config.DISABLE_JIT = True

##############################################################
########### Static and moving wall interraction ##############
##############################################################




# don't set nogil = True so that only one thread at a time modifies wall location
@njit( cache=True, fastmath=True)#, nogil=True)
def movingWallInteraction(crd,csts, x, v, m):
    """
    update coordinates and velocities of particles interacting left wall (if exists)
    The exact time (if wall velocity is constant) is first computed ; then the particle is correctly bounced back

    :param crd: multi-array containing all coords of particles in cell
    :param csts: all physical constants
    :param x : x coordinate of wall
    :param v : velocity coordinate of wall
    :param m : mass of wall
    :return: new location and velocity of wall
    """
    wheres,xs,vxs,lastColls,vys = crd["wheres"],crd["xs"],crd["vxs"],crd["lastColls"],crd["vys"]
    time,mp = csts["time"], csts["ms"]

    for i in range(len(xs)):

        # should be revised to take particle radius into consideration

        if wheres[i] * (xs[i] - x) < 0:
            # particle crossed the wall, in either direction
            delta = -(xs[i] - x) / (v - vxs[i])

            if delta < 0:  # already treated collision
                continue

            # move backward the particle and the wall
            xs[i] -= vxs[i] * delta
            x -= v * delta
            #lastColls[i] = time - delta

            # bounce and compute force applied to wall
            newVx = (2 * m * v + (mp - m) * vxs[i]) / (m + mp)
            v = ( (m - mp) * v + 2 * mp * vxs[i] ) / (m + mp)   # bug here !
            vxs[i] = newVx

            vys[i] *= (-1) ** getrandbits(1)

            # move forward the particle and wall
            xs[i] += vxs[i] * delta
            x += v * delta

    return x, v  # should return now x and v of wall


@njit(cache=True, fastmath=True, nogil=True)
def staticWallInteractionLeft(crd,csts, left):
    """
    update coordinates and velocities of particles interacting with left static wall (if exists)
    :param crd: multi-array containing all coords of particles in cell
    :param csts: all physical constants
    :param left: left coordinate of cell
    return: fleft (computed positively)
    """

    wheres, xs, vxs, lastColls, vys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"]
    time, m ,dt, dec= csts["time"], csts["ms"], csts["dt"], csts["rpc"]*csts["ds"]/2.

    realLeft = left + dec
    fleft = 0.
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] < left:
            odt = abs( (xs[i] - (realLeft)) / vxs[i])
            #lastColls[i] = time - odt
            xs[i] = 2 * realLeft - xs[i]
            vxs[i] *= -1
            vys[i] *= (-1) ** getrandbits(1)

            fleft += abs(vxs[i])
    return fleft * 2 * m / dt


@njit( cache=True, fastmath=True, nogil=True)
def staticWallInteractionRight(crd,csts, right):
    """
    update coordinates and velocities of particles interacting with right static wall (if exists)
    :param crd: multi-array containing all coords of particles in cell
    :param csts: all physical constants
    :param right: right coordinate of cell
    return: fright (computed positively)
    """

    wheres, xs, vxs, lastColls, vys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"]
    time, m ,dt, dec = csts["time"], csts["ms"], csts["dt"],  csts["rpc"]*csts["ds"]/2.

    realRight = right - dec
    fright = 0.
    for i in range(len(xs)):
        if wheres[i] == 0:
            continue

        if xs[i] > right:
            odt = abs( (xs[i] - realRight) / vxs[i])
            #lastColls[i] = time - odt
            xs[i] = 2 * realRight - xs[i]
            vxs[i] *= -1
            vys[i] *= (-1) ** getrandbits(1)
            fright += abs(vxs[i])
    return fright * 2 * m / dt


@njit(cache=True, fastmath=True, nogil=True)
def staticWallInterractionDown(crd,csts,vStar):
    """
    update coordinates and velocities of particles interacting with solid walls (which are horizontal)
    Compute the force applied to the wall

    :param crd: multi-array containing all coords of particles in cell
    :param csts: all physical constants
    :param vStar: quadratic mean velocity for imposing temperature
    :return: fdown, the force (computed positively)
    """

    wheres, xs, vxs, lastColls, vys, ys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"], crd["ys"]
    time, m, dt, dec = csts["time"], csts["ms"], csts["dt"],csts["rpc"]*csts["ds"]/2.

    realDown = 0. + dec
    fdown = 0.
    sqtwo = (4 / 3) ** 0.5

    for i in range(len(ys)//4): # we assume that particles which may cross the lower boundary are sorted at the beginning
        if wheres[i] == 0:
            continue

        if ys[i] < realDown:

            ys[i] = 2*realDown - ys[i]
            odt = abs( (ys[i]-realDown) / vys[i])
            vOld = vys[i]
            #lastColls[i] = time - odt # time of particle/wall collision must be computed

            if vStar > 0: # imposing temperature at boundary
                vxs[i] = np.random.normal(0, vStar / sqtwo, 1)[0]
                vys[i] = - abs(np.random.normal(0, vStar / sqtwo, 1)[0]) * vOld / abs(vOld)
            else:
                # comment next line for PLA boundary
                vxs[i] *= (-1) ** getrandbits(1)  # used for PRA (velocity reversal cause ruguous wall behavior)
                vys[i] *= -1

            fdown += abs(vys[i] - vOld) / 2  # f = -2 * vy * ms * dn / dt

    return  fdown * 2 * m / dt

@njit( cache=True, fastmath=True, nogil=True)
def staticWallInterractionUp(crd,csts,vStar):
    """
    update coordinates and velocities of particles interacting with solid walls (which are horizontal)
    Compute the force applied to the wall

    :param crd: multi-array containing all coords of particles in cell
    :param csts: all physical constants
    :param vStar: quadratic mean velocity for imposing temperature
    :return: fup, the force (computed positively)
    """

    wheres, xs, vxs, lastColls, vys, ys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"], crd["ys"]
    time, m, dt, width, dec = csts["time"], csts["ms"], csts["dt"], csts["width"], csts["rpc"]*csts["ds"]/2.
    fup = 0.
    sqtwo = (4 / 3) ** 0.5
    realWidth = width - dec

    for i in range(3*len(ys)//4, len(ys)):
        if wheres[i] == 0:
            continue


        if ys[i] > realWidth:
            odt = abs((ys[i]-realWidth) / vys[i])
            ys[i] = 2 * realWidth - ys[i]
            vOld = vys[i]
            #lastColls[i] = time - odt # time of particle/wall collision must be computed
            if vStar > 0 :  # imposing temperature at boundary
                vxs[i] = np.random.normal(0, vStar / sqtwo, 1)[0]
                vys[i] = - abs(np.random.normal(0, vStar / sqtwo, 1)[0]) * vOld / abs(vOld)
            else:
                # comment next line for PLA boundary
                #vxs[i] *= (-1) ** getrandbits(1)  # used for PRA (velocity reversal cause ruguous wall behavior)
                vys[i] *= -1

            fup += abs(vys[i] - vOld) / 2  # f = -2 * vy * ms * dn / dt

    return  fup * 2 * m / dt



##############################################################
###################  Collision detection  ####################
##############################################################


@njit( cache=True, fastmath=True, nogil=True)
def _isCollidingFast(x1, y1, x2, y2, vx1, vy1, vx2, vy2, dt, d, lc1, lc2,time):
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
        return 0, 0

    dvc = (vx2 - vx1) ** 2 + (vy2 - vy1) ** 2
    drc = (x2 - x1) ** 2 + (y2 - y1) ** 2

    scal = (x2 - x1) * (vx2 - vx1) + (y2 - y1) * (vy2 - vy1)
    a = dvc
    b = - 2 * scal
    c = drc - d ** 2
    delta = b ** 2 - 4 * a * c

    if delta < 0:
        return 0, 0.

    if delta >= 0:
        t = (-b + delta ** 0.5) / (2 * a)
        if t < 0:
            # collision in the future
            return 0, 0.

        elif  (t > dt or time-t < lc1 or time-t < lc2):
            # collision occurred more than a time step ago, or collision occurred before last collision for i or j
            # last collision might either be with particle or wall
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < d ** 2:
                # in this case, collision has really occurred as one particle is inside anotherone, but before this time step.
                # it must be treated.
                return 2, t
            return 0, 0.

        else:
            return 1, t




@njit(cache=True, fastmath=True, nogil=True)
def _dealWithCollision(xs, ys, vxs, vys, nxs, nys, nvxs, nvys,i,ni,t):
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

    vfa, vfb = _bounce(ra, rb, va, vb)

    vxs[i], vys[i] = vfa[0], vfa[1]
    nvxs[ni], nvys[ni] = vfb[0], vfb[1]

    # re-shift particles
    xs[i] += t * vxs[i]
    ys[i] += t * vys[i]
    nxs[ni] += t * nvxs[ni]
    nys[ni] += t * nvys[ni]


@njit( cache=True, fastmath=True, nogil=True)
def _bounce(ra, rb, va, vb):
    """
    compute accurate bounce between particles
    :param ra: location (2D) of first part at the very moment of collision
    :param rb: location (2D) of second part at the very moment of collision
    :param va: velocity of first particle
    :param vb: velocity of second particle
    :return: vaf,vbf, velocities after impact
    """
    ab = rb - ra
    dotn = _dot((vb - va), ab) * ab / _dot(ab, ab)

    vaf = (va + dotn)
    vbf = (vb - dotn)

    return vaf, vbf



@njit( cache=True, fastmath=True, nogil=True)
def detectCollisionsAtInterface(crd,ncrd,csts):
    """
        This method update particle positions and velocities taking into account elastic collisions
        It is focused on collisions occurring between particles from different cells only.
        Collisions between particles from the same cell are handled before those, in another function
        It is here assumed that particles' lists are sorted according to y value,
        even if some collisions between particles within the same cell might change a little the order

        :param crd: coordinates of first cell
        :param ncrd: coordinates of other cell
        :param csts: constants
        :return : number of collision which were captured

    """
    wheres, xs, vxs, lastColls, vys, ys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"], crd["ys"]
    nwheres, nxs, nvxs, nlastColls, nvys, nys = ncrd["wheres"], ncrd["xs"], ncrd["vxs"], ncrd["lastColls"], ncrd["vys"], ncrd["ys"]
    indices = crd["indLeft"]
    nindices = ncrd["indRight"]
    time, dt, d = csts["time"], csts["dt"], csts["ds"]



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
            if 1. * wheres[ii] * nwheres[jj] <= 0:  # different side (<0) or one dead particle (==0)
                continue

            maxDeltaY = max(maxDeltaY, (abs(vys[ii]) + abs(nvys[jj])) * dt + d)
            deltay = nys[jj] - ys[ii]  # y spacing between ii and jj particle

            if nys[jj] < nextY - maxDeltaY:
                otherFirst = j

            if abs(deltay) < maxDeltaY :
                coll, odt = _isCollidingFast(xs[ii], ys[ii], nxs[jj], nys[jj], vxs[ii], vys[ii], nvxs[jj], nvys[jj], dt, d, lastColls[ii], nlastColls[jj], time)
                if coll>0:

                    nbCollide += 1
                    _dealWithCollision(xs, ys, vxs, vys, nxs, nys, nvxs, nvys, ii, jj, odt)
                    lastColls[ii] = time - odt
                    nlastColls[jj] = time - odt
            else:
                if deltay > maxDeltaY :
                    break

        i = i + 1


    return nbCollide


@njit( cache=True, fastmath=True, nogil=True)
def detectAllCollisions(crd,csts,left, right):
    """
    This method update particle positions and velocities taking into account elastic collisions
    It first tries to detect efficiently if collisions occurred,
    assuming that particles are already sorted by y coordinate
    And then apply collision rules precisely (real instant of collision is computed and particle are shifted correctly)

    :param crd: coordinates of first cell
    :param ncrd: coordinates of other cell
    :param csts: constants
    :param left : left coord of cell
    :param right : right coord of cell
    :return: number of collisions within the cell
    """

    wheres, xs, vxs, lastColls, vys, ys = crd["wheres"], crd["xs"], crd["vxs"], crd["lastColls"], crd["vys"], crd["ys"]
    indicesLeft = crd["indLeft"]
    indicesRight = crd["indRight"]
    time, dt, d = csts["time"], csts["dt"], csts["ds"]

    nbCollide = 0
    secureSearchCoeff = 1.1
    ln = len(xs)
    leftIndex = 0
    rightIndex = 0

    vymax = 0.
    vxmax = 0.
    for i in range(ln):
        # searching vymax and reset indices of particles close to left/right sides
        indicesLeft[i] = -1
        indicesRight[i] = -1
        if wheres[i] != 0:  # ghost particle
            vymax = max(vymax,abs(vys[i]))
            vxmax = max(vxmax,abs(vxs[i]))

    ########################################################
    ##              Main collision search loop            ##
    ########################################################

    for i in range(ln - 1):  # last particle can't collide to anyone else

        if wheres[i] == 0:  # ghost particle
            continue

        x, y, vx, vy, w, lc = xs[i], ys[i], vxs[i], vys[i],wheres[i], lastColls[i] # read once !


        for j in range(i + 1, ln):
            if 1. * w * wheres[j] <= 0:  # different side of wall (<0) or one dead particle (==0)
                # or particle which has already collided
                # 1. used to avoid integer overflow
                continue


            # The following is very important, this criterion is used to stop the searches for part i and step to part
            # i+1.
            # If too restrictive, collisions might be missed. If too broad, computation time may increase
            if abs(ys[j]-y) > ((-vy + vymax) * dt + d)*secureSearchCoeff :
                break  # if here, other particles are too high to be able to interact with particle i
            coll, odt = _isCollidingFast(x, y, xs[j], ys[j], vx, vy, vxs[j], vys[j], dt, d, lc, lastColls[j],time)

            if coll>0:
                nbCollide += 1.
                _dealWithCollision(xs, ys, vxs, vys, xs, ys, vxs, vys, i, j, odt)
                lastColls[i] = time - odt
                lastColls[j] = time - odt
                continue

    ########################################################
    ## detection of particles closed to left/right sides  ##
    ########################################################

    for i in range(ln):
        if wheres[i] == 0:  # ghost particle
            continue
        both = 0
        # worst case : particle of other cell has vxmax velocity and leaves the cell right
        # at the begening of the timestep
        if xs[i] < left + ((vxmax + max(vxs[i],0))*dt + d )*secureSearchCoeff :
            indicesLeft[leftIndex] = i
            leftIndex += 1
            both += 1

        if xs[i] > right - ((vxmax + max(-vxs[i],0))*dt + d )*secureSearchCoeff :
            indicesRight[rightIndex] = i
            rightIndex += 1
            both += 1
        if both == 2:
            nbCollide = -1 # should stop the program ; this cell is too thin

    return nbCollide

