import time
import numpy as np
from random import getrandbits
from numba import jit

vStar = 426.28 #m/s
m=1e-5

@jit(nopython=True, cache=True, fastmath=True)
def getStateBeforeX0(X,Xf,x0,vx0,vy,V):
    """
    All velocities are assumed to be positive
    :param X: initial wall location (other wall at 0)
    :param Xf: final wall location (must be <X and >0)
    :param x0: initial particle location
    :param vx0: initial horizontal particle velocity
    :param vy: (initial) vertical particle velocity
    :param V: wall velocity
    :return: i,vxi,Eci : the number of wall particle collision, the horizontal velocity after the last collision and the kinetic energy of the particle, still after the last collision
    """

    side = getrandbits(1)
    # location of first collision
    i=0
    Xip = X-V * (X + (-1)**side * x0) / (V+vx0)
    vi=vx0
    while Xip > Xf:
        i+=1
        vi = vx0 + 2 * i * V
        Xip = Xip * (vi-V)/(vi+V)
    return i,vi,0.5*m*(vy**2 *0+ vi**2), Xip/X



@jit(nopython=True, cache=True, fastmath=True)
def run(V,X,Xf,nPart):

    EcInit = 0
    EcFinal = 0
    varEcFinal = 0
    for n in range(nPart):
        x0 = np.random.uniform(0,X,1)[0]
        vx0 = abs(np.random.normal(0, vStar/2**0.5, 1)[0])
        vy = np.random.normal(0, vStar/2**0.5, 1)[0]
        EcInit += 0.5 * m * (vx0**2 )#+ vy**2)
        i,vi,Ec,xx = getStateBeforeX0(X,Xf,x0,vx0,vy,V)
        Ec -= 0.5 * m * V ** 2  # macroscopic kinetic energy
        EcFinal += Ec
        varEcFinal += Ec**2

    EcInit /= nPart
    EcFinal = EcFinal/nPart
    varEcFinal = (varEcFinal / nPart - EcFinal ** 2) ** 0.5
    return EcFinal * Xf**2 / (EcInit * X**2), varEcFinal * Xf**2 / (EcInit * X**2) / nPart ** 0.5


ti = time.perf_counter()
X = (4/5) * 0.1
Xf = X/2
nPart = 10000
V = 42
r,ur = run(V,X,Xf,nPart)
print(time.perf_counter() - ti)
print(str(r)[:8], "+-", str(ur)[:8])
