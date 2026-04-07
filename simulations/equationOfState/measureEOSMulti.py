import math
import time

from domain import Domain
from constants import ComputedConstants
import numpy as np

X = 0.1
Y = 0.1
nPart = 32e3
T = 300
P = 1e5
eta = 0.05
# eta = N pi (ds/2)^2 / (XY)
ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2

# P (S - N b) = N k T <- low density approx
# ( P S ) / (N k T) = 1 / (1-eta)^2  <- better model

ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)
domain = Domain(8)
domain.setMaxWorkers(2)
ComputedConstants.dt *= 1

ps = []
ts = []
it = 0
print("warm up")
print("z_th = ", 1/(1-eta)**2)
alpha = 0.1280

print("z_th,2 = ", (1+alpha*eta**2)/(1-eta)**2 )

while it < 5000:
    it += 1
    domain.update()

domain.resetCount()
print("start recording")
while it < 500e3     :
    it += 1
    domain.update()
    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % (10000) == 0:

        p = np.average(ps)
        up = np.std(ps) / len(ps)**(3/4) # / np.sqrt(len(ps)) # correlated measures
        t = np.average(ts)
        z = p * X*Y / (nPart * ComputedConstants.kbs*t)
        print(it, ComputedConstants.time, p,up,t,z)


#eta = 0.001 ; Z_th = 1.00200
    # dt x 1
    #nPart  1024000(X128)  64000 (X16)  16000 (X4)
    #Z       1.0015(1)      1.0019(2)   1.00201(2)

    # dt x 0.25
    #nPart  1024000(X256)  64000 (X16)  16000 (X4)
    #Z         1.0020     1.0021(2)    1.00222(4)

    # dt x 4
    #nPart  1024000(X128)  64000 (X16)  16000 (X4)
    #Z       1.00052      1.00180(4)    1.00222(4)


#eta = 0.1 ; Z_th = 1.020304 ; 1.020317
    # dt x 1
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)
    # Z       1.02020(5)    1.02029(2)   1.02034(1)

    # dt x 0.25
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)
    # Z       1.020494(20)  1.02057(3)   1.02061(4)

    # dt x 4
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)
    # Z                                  1.01865(5)


#eta = 0.1 ; Z_th = 1.2345 ; 1.2361
    # dt x 1
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)
    # Z                      1.23592(3)      1.2354 (1)


    # dt x 0.25
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)    10000 (X1)
    # Z                      1.23600     1.2353 (1)     1.2341

    # dt x 4
    # nPart  1024000(X128)  64000 (X16)  16000 (X4)    1000 (X1)
    # Z                     1.235901      1.2354       1.23280

#eta = 0.3
#n = 64000 ; Dt x1 : 2.0580   ; Dt x0.1 : 2.0579



#eta = 0.05 n=32000 ; Dt = 1 : 1.10873