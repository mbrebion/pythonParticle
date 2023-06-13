import time
from domain import Domain
from constants import ComputedConstants
import numpy as np
from cell import Cell

X = 0.1
Y = 0.1
nPart = 64000
T = 300
P = 1e5
ds = 5e-5

# P (V - N b) = N k T

ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)
domain = Domain(4)
domain.setMaxWorkers(1)

ps = []
ts = []
it = 0
print("warm up")
while it < 300:
    it += 1
    domain.update()
print("start recording")
while it < 10000:
    it += 1
    domain.update()
    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % 50 == 0:
        p = np.average(ps)
        up = np.std(ps) / np.sqrt(len(ps))
        print(it, ComputedConstants.time, p,up,domain.computeTemperature())

