import math
import time

from domain import Domain
from constants import ComputedConstants
import numpy as np

X = 0.1
Y = 0.1
nPart = 8000
T = 300
P = 1e5
eta = 0.05
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
while it < 5000:
    it += 1
    domain.update()

domain.resetCount()
print("start recording")
while it < 24e5:
    it += 1
    domain.update()
    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % 1000 == 0:
        print((domain.insideAverage), (domain.interfaceAverage))
        time.sleep(0.2)
    if it % 10000 == 0:

        p = np.average(ps)
        up = np.std(ps) / len(ps)**(3/4) # / np.sqrt(len(ps)) # correlated measures
        t = np.average(ts)
        z = p * X*Y / (nPart * ComputedConstants.kbs*t)
        print(it, ComputedConstants.time, p,up,t,z)


"""
pour 4000 particules, 300 K, P=1e5 de cible, X = Y = 0.1
eta     1e-4,       3e-4,      1e-3,      3e-3,      1e-2,      3e-2,      1e-1,     2e-1,     3e-1,      4e-1,     5e-1,     6e-1,     7e-1
Z     1.000204,   1.00062,   1.00208,   1.00597,   1.01999,    1.0626,    1.2344,   1.5522,   2.0367,    2.6980,   3.997,     5.991,    9.37
"""



# eta = 0.3   : theoretical target : 204082
# 1 domain
# P = 205811
# nbColl = 499.4

# eta = 0.05   : theoretical target : 110803
# 1 domain
# P = 205811
# nbColl = 499.4


# 16 domains ; # without interface collisions
# P = 211519
# nbColl : 1261


# 16 domains ; # with interface collisions
# P =  209332
# nbColl = 1209.6 + 42.4 : 1252

# 32 domains ; with interfaces collisions
# P = 209410
# nbColl = 1165.5 + 87.5   : 1253


#dt = l/v* / 100
# 32 domains ; with no interfaces collisions
# P = 212556
# nbColl = 254.37


# 32 domains ; with  interfaces collisions
# P = 207432
# nbColl = 231  + 17.2 -> 248.2


# 16 domains ; # with interface collisions
# P =  205812
# nbColl = 238.5 + 8.3 : 246.8


# 1 domain
# P = 205488
# nbColl = 249.3