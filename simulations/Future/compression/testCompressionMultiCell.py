import time
import numpy as np
from constants import ComputedConstants
from domain import Domain


X = 0.4
Y = 0.4
ls = X/25
nPart = 64000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(32)
domain.setMaxWorkers(4)
nWarmUp = 10

def velocity(t):
    if t <= ComputedConstants.dt * nWarmUp:
        return 0.
    return -ComputedConstants.vStar/5


xInit = X/2 + X/10
domain.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)

def cste(Ec,x):
    Ns = nPart * np.pi * (ComputedConstants.ds/2)**2
    S = x*Y
    return Ec * S * (1 - 2 * Ns / S)


# warm up
for x in range(nWarmUp-2):
    domain.update()


ecl = domain.computeKineticEnergyLeftSide()
cinit = cste(ecl,domain.wall.location())
first = True
c = 0

tStart = time.time()
while domain.wall.location() > X/2-X/10:
    domain.update()

    if first or (ComputedConstants.it) % 400 == 0:
        first = False
        ecl = domain.computeKineticEnergyLeftSide()
        ecMacro = domain.computeAverageVelocityLeftOfWall()**2 / 2 * ComputedConstants.ms*domain.countLeft()
        Cste = cste(ecl,domain.wall.location())
        print(domain.wall.location()/X, ecl, ecMacro,Cste/cinit)

ecl = domain.computeKineticEnergyLeftSide()
ecMacro = domain.computeAverageVelocityLeftOfWall() ** 2 / 2 * ComputedConstants.ms * domain.countLeft()
Cste = cste(ecl,domain.wall.location())
print(domain.wall.location()/X, ecl, ecMacro,Cste/cinit)

