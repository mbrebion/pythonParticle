
from numba import njit,config
#config.DISABLE_JIT = True



##############################################################
##########  Average temperature and kinetic energy ###########
##############################################################

@njit( cache=True, fastmath=True, nogil=True)
def computeTemperature(crd,csts):
    ECTa,ECTb = (computeEcs(crd,csts))
    ECMa,ECMb = computeMacroEcs(crd,csts)
    ECm = ECTa+ECTb - ECMa - ECMb
    return ECm / countAlive(crd,csts) / csts["kbs"]

@njit( cache=True, fastmath=True, nogil=True)
def computeTemperatureLeft(crd,csts):
    ECTa,ECTb = (computeEcs(crd,csts))
    ECMa,ECMb = computeMacroEcs(crd,csts)
    ECm = ECTa - ECMa
    return ECm / countAliveLeft(crd["wheres"]) /csts["kbs"]

@njit( cache=True, fastmath=True, nogil=True)
def computeTemperatureRight(crd,csts):
    ECTa,ECTb = (computeEcs(crd,csts))
    ECMa,ECMb = computeMacroEcs(crd,csts)
    ECm = ECTb - ECMb
    return ECm / countAliveRight(crd["wheres"]) /csts["kbs"]


@njit( cache=True, fastmath=True, nogil=True)
def computeEcs(crd, csts):
    """
        compute kinetic energy left and right to wall
        :param vxs: x velocities
        :param vys: y velocities
        :param wheres: mask array
        :param m: mass
        :return: left and right kinetic energy
        """
    vxs, vys, wheres, m = crd["vxs"], crd["vys"], crd["wheres"], csts["ms"]
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

@njit( cache=True, fastmath=True, nogil=True)
def computeMacroEcs(crd,csts):
    """
        compute kinetic energy left and right to wall
        :param vxs: x velocities
        :param vys: y velocities
        :param wheres: mask array
        :param m: mass
        :return: left and right kinetic energy
        """
    vxs, vys, wheres, m = crd["vxs"], crd["vys"], crd["wheres"], csts["ms"]
    vxL,vyL = 0., 0.
    vxR, vyR = 0., 0.
    countL,countR = 0,0

    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue

        if wheres[i] < 0:
            countL += 1
            vxL += vxs[i]
            vyL += vys[i]
        else:
            countR += 1
            vxR += vxs[i]
            vyR += vys[i]

    countL = max(countL,1)
    countR = max(countR, 1)
    ecl = 0.5 * m * countL * ( (vxL/countL)**2 + (vyL/countL)**2 )
    ecr = 0.5 * m * countR * ( (vxR / countR) ** 2 + (vyR / countR) ** 2)
    return ecl, ecr



@njit( cache=True, fastmath=True, nogil=True)
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


@njit(cache=True, fastmath=True, nogil=True)
def computeXVelocityBins(crd,csts, bins):
    """
    :param bins: array dedicated to store the means
    :return: modified bins
    """
    ys, ymax, vxs, wheres = crd["ys"], csts["width"], crd["vxs"], crd["wheres"]
    ns = [0] * len(bins)
    bins *= 0
    for i in range(len(vxs)):
        if wheres[i] == 0:
            continue

        index = min(int(ys[i] / ymax * len(bins)),len(bins)-1)

        bins[index] += vxs[i]
        ns[index] += 1

    for i in range(len(bins)):
        bins[i] /= max(ns[i], 1)
    return bins


@njit(cache=True, fastmath=True, nogil=True)
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


@njit( cache=True, fastmath=True, nogil=True)
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



@njit( cache=True, fastmath=True, nogil=True)

def computeColorRatio(crd,csts):
    wheres,colors= crd["wheres"], crd["colors"]
    count = 0
    total = 0
    for i in range(len(wheres)):
        if wheres[i] != 0:
            total += 1
            if colors[i] <= 0.5:
                count += 1
    return count/max(1,total)

@njit( cache=True, fastmath=True, nogil=True)
def countAlive(crd,csts):
    wheres = crd["wheres"]
    count = 0
    for i in range(len(wheres)):
        if wheres[i] != 0:
            count += 1
    return count


@njit( cache=True, fastmath=True, nogil=True)
def countAliveLeft(crd, csts):
    xs ,wheres, x = crd["xs"], crd["wheres"], csts["x"]
    countWhere, countLoc = 0, 0
    for i in range(len(wheres)):
        if wheres[i] != 0 and xs[i] < x:
            countLoc += 1
        if wheres[i] < 0:
            countWhere += 1
    if countLoc != countWhere and False:
        print("problem ", countLoc, countWhere)
    return countWhere


@njit( cache=True, fastmath=True, nogil=True)
def countAliveRight(crd, csts):
    xs, wheres, x = crd["xs"], crd["wheres"], csts["x"]
    countWhere, countLoc = 0, 0
    for i in range(len(wheres)):
        if wheres[i] != 0 and xs[i] > x:
            countLoc += 1
        if wheres[i] > 0:
            countWhere += 1
    if countLoc != countWhere and False:
        print("problem ", countLoc, countWhere)
    return countWhere

