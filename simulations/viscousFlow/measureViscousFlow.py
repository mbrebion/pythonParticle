import time

from domain import Domain
from constants import ComputedConstants, MASS, Kb
import numpy as np
import thermo

X = 0.1
Y = 0.1
nPart = 1024000*4
T = 312
P = 1e5
ls = 2e-3  # mean free path
nbBin = 12

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
gee = 50
ComputedConstants.forceX = 9.81 * ComputedConstants.ms * gee  # 300
ComputedConstants.boundaryTemperatureDown = 300  # K
ComputedConstants.dt *= 0.5


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
print("Re   = ", round(ComputedConstants.ms * nPart / X / Y * vmax * Y / etaTH, 2))
print("rho   = ", round(ComputedConstants.ms * nPart / X / Y, 2))
print("*****")


def vxyVelocityProfile(y):
    YY = 2 * Y
    return 4 * vmax / YY ** 2 * y * (YY - y)


domain = Domain(512, periodic=True, v_xYVelocityProfile=vxyVelocityProfile)
domain.setMaxWorkers(2)


print("start simulating")


def computeBlock(sizeBlock=10000):
    """
    compute size iteration to estimate velocites and standard deviations. Then deduce the averaged standard deviation of the x velocities
    and provide the estimated size block, i.e the number of iterations required to obtain a given accuracy on velocities and the averaged velocities
    :param sizeBlock: nb of iterations
    :return: astd,averages
    """

    bins = []
    itFirst = ComputedConstants.it
    # compute standard deviation
    while ComputedConstants.it - itFirst < sizeBlock:
        domain.update()

        if ComputedConstants.it % 4 == 0:  # wait onevery it to ensure independence of snapshots
            bins.append(np.array(domain.computeXVelocityBins(nbBin)))

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
    out = "["
    for e in l:
        out += str(round(e, 2)) + ","
    out = out[:-1]
    out += "]"
    return out


count = 0
vs = np.array([0.] * len(vs))

while ComputedConstants.it < 1000000:
    std, newvs = computeBlock(sizeBlock)
    count += 1
    print(round(std,2),ComputedConstants.it,round(domain.computeTemperature(),2),round(domain.computeTemperatureUncorrected(),2),lts(newvs))
    vs += newvs
    print(" averaging", lts(vs/count))
    print()

    time.sleep(20)


