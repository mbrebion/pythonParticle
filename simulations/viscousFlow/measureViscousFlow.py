import time

from domain import Domain
from constants import ComputedConstants
import numpy as np


X = 0.1
Y = 0.1
nPart = 32000
T = 318
P = 1e5
ls = Y/50  # mean free path

ComputedConstants.thermodynamicSetup(T,X,Y,P,nPart,ls)
ComputedConstants.forceX = 9.81 * ComputedConstants.ms * 300
ComputedConstants.boundaryTemperature = 300 #K
ComputedConstants.dt *= 1

def vxyVelocityProfile(y):
    vmax = 29
    YY = 2*Y
    return 4*vmax/YY**2 * y * (YY-y)


domain = Domain(12, periodic=True,v_xYVelocityProfile=vxyVelocityProfile)
domain.setMaxWorkers(2)


print("start recording")

# averaging properties
nbBin = 12
maxVelocityAccuracy = 0.3
onevery = 4


def computeBlock(sizeBlock=10000):
    """
    compute size iteration to estimate velocites and standard deviations. Then deduce the averaged standard deviation of the x velocities
    and provide the estimated size block, i.e the number of iterations required to obtain a given accuracy on velocities and the averaged velocities
    :param sizeBlock: nb of iterations
    :param size: requiresAccuracy for velocities
    :return: astd,averages
    """

    bins = []
    itFirst = ComputedConstants.it
    # compute standard deviation
    while ComputedConstants.it - itFirst < sizeBlock:
        domain.update()

        if ComputedConstants.it % onevery == 0:  # wait onevery it to ensure independence of snapshots
            bins.append(np.array(domain.computeXVelocityBins(nbBin)))
        if ComputedConstants.it % 2000 == 0:
            time.sleep(4.5)

    stds = np.array([0.]*nbBin)
    averages = np.array([0.]*nbBin)
    for i in range(len(bins)):
        averages += bins[i]
        stds += bins[i]**2

    averages /= len(bins)
    stds = np.sqrt(stds/len(bins) - averages**2)
    averageSTD = np.average(stds)
    return averageSTD,averages


def estimateBestSizeBlock(averageSTD,requiredAccuracy):
    return int(onevery * (averageSTD / requiredAccuracy) ** 2)*10


def isConverged(vs1,vs2,accuracy):
    diff = (sum([(vs1[i]-vs2[i])**2 for i in range(len(vs1))])**0.5)/len(vs1)
    return diff<accuracy


sizeBlock = 5e4
std, vs = computeBlock(sizeBlock) # warm up


def lts(l):
    out = "["
    for e in l:
        out += str(e)+","
    out = out[:-1]
    out += "]"
    return out


count = 1
vs = np.array([0.]*len(vs))
while ComputedConstants.it < 30e5:
    std, newvs = computeBlock(sizeBlock)
    count += 1
    ratio = 2 / (count + 1)
    print(std,ComputedConstants.it,domain.computeTemperature(),lts(newvs))
    vs = ratio * newvs + (1-ratio) * vs
    print(" averaging", lts(vs))
    print()


