import math
import time

from domain import Domain
from constants import ComputedConstants
import numpy as np

X = 0.1
Y = 0.1
nPart = 128000
T = 300
P = 1e5
eta = 0.5
# eta = N pi (ds/2)^2 / (XY)
ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2

# P (S - N b) = N k T <- low density approx
# ( P S ) / (N k T) = 1 / (1-eta)^2  <- better model

ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)
domain = Domain(16)
domain.setMaxWorkers(1)
ComputedConstants.dt *= 0.01

ps = []
ts = []
it = 0
print("warm up")
while it < 10000:
    it += 1
    domain.update()

domain.resetCount()
print("start recording")
while it < 10e5:
    it += 1
    domain.update()
    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % 10000 == 0:

        p = np.average(ps)
        up = np.std(ps) / len(ps)**(3/4) # / np.sqrt(len(ps)) # correlated measures
        t = np.average(ts)
        z = p * X*Y / (nPart * ComputedConstants.kbs*t)
        print(it, ComputedConstants.time, p,up,t,z)


#eta = 0.1
#nPArt  4000    16000    64000 (X16)    |    64000(X4)  64000 (X1)
#  Z    1.2348  1.2355     1.2364       |     1.2365      1.2367