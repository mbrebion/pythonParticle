import time

import numpy as np

from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.4
Y = 0.4
ls = X / 25
nPart = 256000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(32)
domain.setMaxWorkers(4)
nWarmUp = 20


def velocity(t):
    if t <= ComputedConstants.dt * nWarmUp:
        return 0.
    return -ComputedConstants.vStar / 5


xInit = 8 * X / 10
domain.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)

ComputedConstants.dt *= 1


# Cell.collision = False

def cste(Ec, x):
    Ns = 4/5*nPart * np.pi * (ComputedConstants.ds / 2) ** 2   # TODO : 4/5 or 1 ???
    S = x * Y
    return Ec * S * (1 - 2 * Ns / S)


# warm up
for x in range(nWarmUp - 2):
    domain.update()

ecl = domain.computeKineticEnergyLeftSide()
cinit = cste(ecl, domain.wall.location())
first = True

tStart = time.time()
while domain.wall.location() > 4 * X / 10:
    domain.update()

    if first or ComputedConstants.it % 100 == 0:
        first = False
        ecl = domain.computeKineticEnergyLeftSide()
        ecMacro = domain.computeAverageVelocityLeftOfWall() ** 2 / 2 * ComputedConstants.ms * domain.countLeft()
        ecr = domain.computeKineticEnergyRightSide()
        Cste = cste(ecl, domain.wall.location())
        print(domain.wall.location() / X, ecl, Cste / cinit, ecMacro, time.time() - tStart)
        time.sleep(0.2)

ecl = domain.computeKineticEnergyLeftSide()
ecMacro = domain.computeAverageVelocityLeftOfWall() ** 2 / 2 * ComputedConstants.ms * domain.countLeft()
Cste = cste(ecl, domain.wall.location())
print(domain.wall.location() / X, ecl, Cste / cinit, ecMacro, time.time() - tStart)





# a : 32_0025_128k
# b : 16_005_128k
# c : 16_005_128k
# d : 16_005_128k
# d': 16_005_256k
# e : 16_005_256k
# f : 16_005_256k - sans gestion de l'interface
# g : 1_005_256k
# h : 1_005_256k



