from constants import ComputedConstants
from domain import Domain
from cell import Cell


X = 0.2
Y = 0.4
ls = Y*2
nPart = 1
T = 300
P = 1e5


def velocity(t):
    return -5


Cell.collision = False
ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(1)
domain.addMovingWall(1000, X, 40, imposedVelocity=velocity)

c = domain.cells[0].coords

dec = 0.01
c.xs[0] = X - dec*1.2
c.ys[0] = Y - dec
c.vxs[0] = 10
c.vys[0] = 16
ComputedConstants.dt = dec / 10 * 2

print(c.xs[0],c.ys[0],c.vxs[0],c.vys[0])
domain.update()
print(c.xs[0],c.ys[0],c.vxs[0],c.vys[0])
