from constants import ComputedConstants
from domain import Domain

X = 0.4
Y = 0.1
ls = X / 25
nPart = 128000
T = 300
P = 1e5
v = 0
m = 1

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(32)
domain.setMaxWorkers(4)
outputName = "m1v00N128k.txt"
outputColor = "black"

domain.addMovingWall(m, X / 2, v)

print(domain.computeKineticEnergyLeftSide(),domain.computeKineticEnergyRightSide(),domain.countLeft(),domain.countRight())