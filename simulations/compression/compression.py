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


X = 0.2
Y = 0.8
ls = X/100
nPart = 16000*8
T = 300
P = 1e5


def velocity(t):
    return -5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(32)
domain.setMaxWorkers(2)
#Cell.collision = False

xInit = 8 * X / 10
domain.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)


def cste(Ec,x):
    Ns = nPart * np.pi * (ComputedConstants.ds/2)**2
    S = x*Y
    return Ec * S * (1 - 2 *Ns/S )

def cstep(Ec,x):
    Ns = nPart  * np.pi * (ComputedConstants.ds/2)**2
    S = x*Y+2*(x+Y)*ComputedConstants.ds/2 - 4 * (ComputedConstants.ds/2)**2
    return Ec * S * (1 - 2 * Ns/S +  0.5 * (Ns/S)**2)

ComputedConstants.dt *= 1
domain.update()


ecl = domain.computeKineticEnergyLeftSide()
cinit = cste(ecl,domain.wall.location())
cinitp = cstep(ecl,domain.wall.location())

while domain.wall.location() > 4 * X / 10:
    domain.update()
    if (ComputedConstants.it+9995) % 500 == 0:
        ecl = domain.computeKineticEnergyLeftSide()
        ecr = domain.computeKineticEnergyRightSide()
        Cste = cste(ecl,domain.wall.location())
        Cstep = cstep(ecl, domain.wall.location())

        print(velocity(ComputedConstants.time),domain.wall.location()/X, ecl, Cste/cinit,Cstep/cinitp)
        time.sleep(1)

ecl = domain.computeKineticEnergyLeftSide()
Cste = cste(ecl,domain.wall.location())
Cstep = cstep(ecl,domain.wall.location())
print( domain.wall.location()/X, ecl, Cste/cinit,Cstep/cinitp)