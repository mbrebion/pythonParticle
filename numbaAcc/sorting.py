from numba import njit,config
import numpy as np

#config.DISABLE_JIT = True

##############################################################
###################### Sorting algorithm #####################
##############################################################



@njit( cache=True, fastmath=True, nogil=True)
def sortCell(crd,temp):
    """
    Sort crd arrays according to ys with insertion algorithm (fastest for quite already sorted arrays)
    :param crd: numpy structured array
    :return: Amount of swaps
    """
    ys = crd["ys"]
    for i in range(1, len(ys)):

        value = ys[i]  # sort both alive and dead particles to keep dead indices equally spaced ?

        swaps = 0
        j = i - 1

        temp[0] = crd[i] # is it safe without copy ?

        while value < ys[j] and j >= 0:
            #roll1(j + 1, xs, ys, vxs, vys, wheres, lastColls,colors)
            crd[j+1] = crd[j]
            swaps += 1
            j = j - 1

        if swaps > 0:
            #xs[j + 1], ys[j + 1], vxs[j + 1], vys[j + 1], wheres[j + 1], lastColls[j + 1], colors[j+1] = x, y, vx, vy, where, lastColl,col
            crd[j+1] = temp[0]

    return swaps
