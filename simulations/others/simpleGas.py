import math

from domain import Domain
from constants import ComputedConstants

X = 0.1
Y = 0.1
nPart = 64000
T = 300
P = 1e5
eta = 0.05
ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2
ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)
domain = Domain(32)
domain.setMaxWorkers(2)

it = 0
while it < 1000:
    it += 1
    domain.update()

print("finish")