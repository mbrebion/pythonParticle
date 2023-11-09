import math
import time
import numpy as np
from constants import ComputedConstants
from domain import Domain
from cell import Cell


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.4
Y = 0.4
ls = X/25
nPart = 32000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(1)
domain.setMaxWorkers(1)
nWarmUp = 10

def velocity(t):
    if t <= ComputedConstants.dt * nWarmUp:
        return 0.
    return -25

xInit = 8 * X / 10
domain.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)

ComputedConstants.dt *= 0.2
#Cell.collision = False
def cste(Ec,x):
    Ns = nPart * np.pi * (ComputedConstants.ds/2)**2
    S = x*Y
    return Ec * S * (1 - 2 *Ns/S )

def cstep(Ec,x):
    Ns = nPart * np.pi * (ComputedConstants.ds/2)**2
    S = x*Y
    return Ec * S * (1 - 2 * Ns/S + 0.5 * (Ns/S)**2)


# warm up
for x in range(nWarmUp-2):
    domain.update()


ecl = domain.computeKineticEnergyLeftSide()
cinit = cste(ecl,domain.wall.location())
cinitp = cstep(ecl,domain.wall.location())
first = True
bins = domain.computeXVelocityBins(16)
c=0
while domain.wall.location() > 4 * X / 10:
    domain.update()

    if ComputedConstants.it % 20 == 0:
        nbins = domain.computeXVelocityBins(16)
        c+=1
        r = 2/(c)
        bins = r * nbins + (1-r) * bins


    if first or (ComputedConstants.it) % 200 == 0:
        first = False
        ecl = domain.computeKineticEnergyLeftSide()
        ecr = domain.computeKineticEnergyRightSide()
        Cste = cste(ecl,domain.wall.location())
        Cstep = cstep(ecl, domain.wall.location())
        print(domain.wall.location() / X, " <-> ", bins)
        #print(domain.wall.location()/X, ecl, Cste/cinit,Cstep/cinitp)
        time.sleep(0.5)

ecl = domain.computeKineticEnergyLeftSide()
Cste = cste(ecl,domain.wall.location())
Cstep = cstep(ecl,domain.wall.location())
print( domain.wall.location()/X, ecl, Cste/cinit,Cstep/cinitp)

