import time

from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.02
Y = 0.01
ls = 1e-3
nPart = 16000*4
T = 300
P = 1e5


def velocity(t):
    return -2


ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(4*2)
domain.setMaxWorkers(1)
domain.addMovingWall(1000, 4 * X / 5, 40, imposedVelocity=velocity)

print("number of cells : ", len(domain.cells))
it = 0

init = domain.computeKineticEnergyLeftSide() * domain.wall.location()
while domain.wall.location() > 2 * X / 5:
    it += 1
    domain.update()
    if it % 400 == 0:
        ecl = domain.computeKineticEnergyLeftSide()
        ecr = domain.computeKineticEnergyRightSide()
        print(ComputedConstants.time, domain.wall.location(), ecl, ecr, ecl * domain.wall.location() / init)
        time.sleep(1)
