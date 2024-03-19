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
ls = X/25
nPart = 128000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(64)

#nPart = 32000
#domain = Domain(1) # 516 secondes
#domain = Domain(4) # 291 secondes
#domain = Domain(8) # 198 secondes
#domain = Domain(16) # 149 secondes

#nPart = 128000
#domain = Domain(1) # 7648 secondes
#domain = Domain(4) # 3212 secondes
#domain = Domain(8) # 1872 secondes
#domain = Domain(16) # 1212 secondes
#domain = Domain(32) # 856 secondes
#domain = Domain(64) # 696 secondes
#domain = Domain(128) # 668 secondes

#nPart = 256000
#domain = Domain(1) # 30352 secondes
#domain = Domain(64) # 2028 secondes
#domain = Domain(128) # 1728 secondes
#domain = Domain(256) # 1496 secondes

domain.setMaxWorkers(1)
nWarmUp = 10

def velocity(t):
    if t <= ComputedConstants.dt * nWarmUp:
        return 0.
    return -10


xInit = 8 * X / 10  
domain.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)

ComputedConstants.dt *= 1
#Cell.collision = False

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
bins = domain.computeXVelocityBins(16)
c = 0

tStart = time.time()
while domain.wall.location() > 4 * X / 10:
    domain.update()

    if ComputedConstants.it % 20 == 0 and False:
        nbins = domain.computeXVelocityBins(16)
        c+=1
        r = 2/(c)
        bins = r * nbins + (1-r) * bins


    if first or (ComputedConstants.it) % 400 == 0:
        first = False
        ecl = domain.computeKineticEnergyLeftSide()
        ecMacro = domain.computeAverageVelocityLeftOfWall()**2 / 2 * ComputedConstants.ms*domain.countLeft()
        ecr = domain.computeKineticEnergyRightSide()
        Cste = cste(ecl,domain.wall.location())
        print(domain.wall.location()/X, ecl, Cste/cinit,ecMacro, time.time()-tStart)
        time.sleep(0.5)

ecl = domain.computeKineticEnergyLeftSide()
ecMacro = domain.computeAverageVelocityLeftOfWall() ** 2 / 2 * ComputedConstants.ms * domain.countLeft()
Cste = cste(ecl,domain.wall.location())
print( domain.wall.location()/X, ecl, Cste/cinit,ecMacro,time.time()-tStart)

