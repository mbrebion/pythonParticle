import math

from domain import Domain
from constants import ComputedConstants
import numpy as np
from cell import Cell
import thermo

X = 0.1
Y = 0.1
nPart = 12000
T = 300
P = 1e5
eta = 0.7
# eta = N pi (ds/2)^2 / (XY)
ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2


# P (S - N b) = N k T <- low density approx

# ( P S ) / (N k T) = 1 / (1-eta)^2  <- better model

ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)
domain = Domain(1)
domain.setMaxWorkers(1)

ps = []
ts = []
it = 0
print("warm up")
while it < 2500:
    it += 1
    domain.update()

print("start recording")
while it < 4e5:
    it += 1
    domain.update()
    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % 10000 == 0:

        p = np.average(ps)
        up = np.std(ps) / len(ps) # / np.sqrt(len(ps)) # correlated measures
        t = np.average(ts)
        z = p * X*Y / (nPart * ComputedConstants.kbs*t)
        print(it, ComputedConstants.time, p,up,t,z)


"""
pour 4000 particules, 300 K, P=1e5 de cible, X = Y = 0.1
eta     1e-4,       3e-4,      1e-3,      3e-3,      1e-2,      3e-2,      1e-1,     2e-1,     3e-1,      4e-1,     5e-1,     6e-1,     7e-1
Z     1.000204,   1.00062,   1.00208,   1.00597,   1.01999,    1.0626,    1.2344,   1.5522,   2.0367,    2.6980,   3.997,     5.991,    9.37
"""