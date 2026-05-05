import time
from domain import Domain, MASS, Kb
import numpy as np
import thermo
from numbaAcc import measuring
import sys
np.set_printoptions(threshold=sys.maxsize)

X = 0.1
Y = 0.1
nPart = 256000
T = 312
P = 1e5
ls = 2e-3  # mean free path
nbBin = 12
nc = 64
gee = 50

kbs = thermo.getKbSimu(1e5, Y * X, T, nPart)
ms = thermo.getMSimu(MASS, Kb, kbs)

def vxyVelocityProfile(y):
    YY = 2 * Y
    return 4 * vmax / YY ** 2 * y * (YY - y)

def eta0(l, T, Ns):
    """
    l: mean free path
    T: temperature
    Ns: Nb of particles
    :return:viscosity provided by boltzmann theory
    """
    sigma = Y ** 2 / (2 * 2 ** 0.5 * l * Ns)
    # nu = Ns * np.pi * (sigma/2)**2 / Y**2

    kbs = thermo.getKbSimu(1e5, Y ** 2, T, Ns)
    ms = thermo.getMSimu(MASS, Kb, kbs)

    etaB = 1 / (2 * sigma) * (ms * kbs * T / np.pi) ** 0.5
    return etaB


def getVMax(eta, T):
    rhog = MASS * P / (Kb * T) * 9.81 * gee
    return rhog * (2 * Y) ** 2 / (8 * eta)


etaTH = eta0(ls, T, nPart)
vmax = getVMax(etaTH, T)
print("*****")
print("etaTH = ", round(etaTH, 4), "USI")
print("vMax = ", round(vmax, 1), "m/s")
print("Re   = ", round(ms * nPart / X / Y * vmax * Y / etaTH, 2))
print("rho   = ", round(ms * nPart / X / Y, 2))
print("*****")


domain = Domain( nc, T, X, Y, P, nPart, ls, drOverLs=0.02,maxWorkers=2, periodic=True, v_xYVelocityProfile=vxyVelocityProfile)
domain.csts["forceX"] = 9.81 * ms * gee #300
for c in domain.cells:
    c.boundaryTempDown = 300. #K


print("start simulating")

def computeBlock(sizeBlock=1000):
    """
    compute size iteration to estimate velocites and standard deviations. Then deduce the averaged standard deviation of the x velocities
    and provide the estimated size block, i.e the number of iterations required to obtain a given accuracy on velocities and the averaged velocities
    :param sizeBlock: nb of iterations
    :return: astd,averages
    """
    bins = []
    bin = np.array([0. for _ in range(nbBin)])
    itFirst = domain.csts["it"]*1.
    # compute standard deviation
    while domain.csts["it"] - itFirst < sizeBlock:
        domain.update()

        if domain.csts["it"] % 10 == 0:  # wait onevery it to ensure independence of snapshots
            domain.computeParam(measuring.computeXVelocityBins, extensive=False, additionalParam=bin)
            bins.append(bin.copy())



    stds = np.array([0.] * nbBin)
    averages = np.array([0.] * nbBin)
    for i in range(len(bins)):
        averages += bins[i]
        stds += bins[i] ** 2

    averages /= len(bins)
    stds = np.sqrt(stds / len(bins) - averages ** 2)
    averageSTD = np.average(stds)
    return averageSTD, averages


sizeBlock = 100
std, vs = computeBlock(sizeBlock * 2)  # warm up

print("start recording")


def lts(l):
    return repr(l)


count = 0
vs = np.array([0.] * len(vs))

while domain.csts["it"] < 100000:
    std, newvs = computeBlock(sizeBlock)
    count += 1
    print(round(std,2),
          domain.csts["it"],
          round(domain.computeParam(measuring.computeTemperature,extensive=False),2),
          lts(newvs) )
    vs += newvs
    print(" averaging", lts(vs/count))
    print()




