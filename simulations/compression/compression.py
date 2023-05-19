from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.2
Y = 0.1
ls = 20e-3
nPart = 2000
T = 300
P = 1e5


def velocity(t):
    return -0.


ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(4)
domain.setMaxWorkers(1)
domain.addMovingWall(1000, X/2, 40, imposedVelocity=velocity)

print("number of cells : ", len(domain.cells))
it = 0

while domain.wall.location() > X/10:
    it += 1
    domain.update()
    if it % 50 == 0:
        ecl = domain.computeKineticEnergyLeftSide()
        ecr = domain.computeKineticEnergyRightSide()
        print(ComputedConstants.time, domain.wall.location(), ecl,ecr, domain.countLeft())
