
from numba import njit,config
from numbaAcc.helping import _goodIndexToInsertTo
#config.DISABLE_JIT = True



##############################################################
###################### Particle Swapping #####################
##############################################################

@njit( cache=True, fastmath=True, nogil=True)

def _fromAtoB(crdA, crdB, ia,ib):
    # swap
    crdB[ib] = crdA[ia]
    # reset
    # ysa is not reset as still used for sorting purposes.
    # velocities are set to 0 so that y and x stay constants
    crdA["vxs"][ia] = 0.
    crdA["vys"][ia] = 0.
    crdA["wheres"][ia] = 0



@njit(cache=True, fastmath=True, nogil=True)
def moveToSwap(crd,swapCrd, xLim, kindOfLim):
    """
    Move particles from cell to swap and return the amount of particle moved
    :param crd: coordinates of particles in cell
    :param swapCrd: coordinates of particles in swap
    :param kindOfLim: if True, limit is at right side, if False, at Left side
    :return: the amount of particle moved
    """
    indices = crd["indRight"] if kindOfLim else crd["indLeft"]
    wheres,xs = crd["wheres"], crd["xs"]
    count = 0
    for i in indices:
        if i < 0:
            break
        if wheres[i] != 0 and (xs[i] > xLim) == kindOfLim:
            _fromAtoB( crd, swapCrd,i, count)
            count += 1

    return count



@njit( cache=True, fastmath=True, nogil=True)
def moveSwapToNeighbor(swapCrd,crd,amount,ymax):
    """
    Move particles from swap a to other cell b
    :param swapCrd: coordinates stored in swap arrays
    :param crd: coordinates of particles in neighboring cell
    :param amount: number of particle to be moved
    :param ymax: width of cells

    :return: None
    """
    ys,swapYs,wheres = crd["ys"], swapCrd["ys"], crd["wheres"]
    for i in range(amount):
        bestIndex = _goodIndexToInsertTo(swapYs[i], ys, wheres, ymax)
        _fromAtoB(swapCrd,crd,i,bestIndex)




