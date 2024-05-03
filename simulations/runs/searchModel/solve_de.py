import numpy as np
from scipy.integrate import solve_ivp
import scipy.special as ss

b = 1/4

def getModel(gamma,lsOX):
    v0star = 414 # m/s
    V0 = gamma * v0star
    X0 = 0.32 # m
    Y = 0.4  # m
    ls = lsOX * X0
    k = 1 - lsOX * 1
    alpha = (gamma + np.sqrt(2/3)) / (b * gamma)

    xstar = 2 * 512000 * np.pi *(1.38107e-05/2)**2 / (Y * Y) # correctif pour passer du GP au gaz réel : il s'agit du double de la compacité


    def delta(x):
        t = (1-x) * X0 / V0
        teta = 2 * np.pi**(0.5-2) * x**(5/2) / (ls * v0star) * X0**2 # viscous damping time
        omega0 = np.pi * np.sqrt(2/3) * v0star / X0 * x**(-3/2) # acoustic pulsation
        ta = X0 / (v0star) * b
        return -gamma * (k-1 - k * np.exp(-t/ta) )
        #return gamma * np.exp(-t/ta)

    def f(x,u):
        return -u/(x-xstar) - 2 / (x-xstar)**(3/2) * delta(x)

    u = solve_ivp(f,[1,0.5],[0.],max_step=0.001)  # résolution numérique
    return u.t, u.y[0]


def getModelApprox(gamma,lsOX):
    xs = np.arange(0.5,1,0.001)
    alpha = 1 / (b * gamma)
    ubis = 2 * b * gamma ** 2 / xs * (1 - np.exp(alpha * (xs - 1)) / np.sqrt(xs))  # modèle approché
    return xs,ubis

def getModelExact(gamma,lsOX):
    xs = np.arange(0.5,1,0.001)
    alpha = 1 / (b * gamma)
    uter = [-4 * gamma / np.sqrt(alpha) / x * (
            ss.dawsn(np.sqrt(alpha * x)) * np.exp(alpha * (x - 1)) - ss.dawsn(np.sqrt(alpha))) for x in xs] # modèle exact
    return xs,uter



