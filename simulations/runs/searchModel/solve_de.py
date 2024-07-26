import numpy as np
from scipy.integrate import solve_ivp
import scipy.special as ss

b = 1/3.6

X0 = 0.2  # m
Y = 0.1  # m
xstar = 2 * 512000 * np.pi * (1.72633e-06 / 2) ** 2 / (X0 * Y)

lsCoeff = 1/2**0.5

def getModel(gamma,lsOX):
    v0star = 414.04 # m/s
    V0 = gamma * v0star
    k = 1 - lsOX *lsCoeff

    def delta(x):
        t = (1-x) * X0 / V0
        tab = X0/ (v0star * (2/3)**0.5 + V0) * b
        return -gamma * (k-1 - k * np.exp( -(t/tab) ) )

    def f(x,u):
        return -u/(x-xstar) - 2 * (1-xstar)**0.5 / (x-xstar)**(3/2) * delta(x)

    u = solve_ivp(f,[1,0.5],[0.],max_step=0.001)  # résolution numérique
    return u.t, u.y[0]


def getModelApprox(gamma,lsOX):
    xs = np.arange(0.5,1,0.001)
    k = 1 - lsOX * lsCoeff
    alpha = 1 / b * (1 + (2/3)**0.5 / gamma)
    sq = ((xs-xstar)/(1-xstar))**0.5
    epsilon = 4*gamma*(1-k) *(1-sq) + 2*k*gamma / alpha / (1-xstar) * (1 - np.exp( alpha *(xs-1)) *sq )
    return xs,epsilon /sq**2


