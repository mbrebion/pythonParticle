# set of thermo functions used to compute simulation values from real values
import numpy as np

def getNCollisionEstimate(Nc,ls,Hc,drOverLs):
    """

    :param Nc: number of particle in the cell
    :param ls: mean free path
    :param Hc: height of the cell
    :return: the estimated (overestimated) number of particles which may collide to one during a single time step
    """
    return max(int(drOverLs*2 * Nc * ls / Hc *1.2),50)

def getKbSimu(pressure, volume, temperature, NSimu):
    """
    compute the 2D equivalent boltzmann constant for a 2D GP of Ns instead of N particles. Thermodynamic values are the one of the real 3D gaz
    :param pressure: real pressure (in Pa)
    :param volume:  real volume (in m^3)
    :param temperature: real temperature (in K)
    :param NSimu: simulated number of particle
    :return: boltzmann constant used in 2D simulation (in USI)
    """
    return pressure * volume / (NSimu * temperature)


def getMSimu(mass, Kb, Kbs):
    """
    compute mass of simulated particles
    :param Kb: real Boltzmann constant
    :param Kbs: used Boltzmann constant
    :param mass: target mass of real particles

    :return: effective mass used in 2D simulations (in kg)
    """
    return mass * Kbs / Kb


def getMeanFreePathSimulated(S, ds, Ns):
    """
    compute the average mean free path
    :param S: Surface of cell
    :param ds: diameter (simulated) of particle
    :param Ns: number of simulated particle
    :return: average mean free path
    """
    # 2**0.5 is here to take into account particle's movement

    return S / (2 * 2 ** 0.5 * ds * Ns)


def getDiameter(S, Ns, ls):
    """
    compute the effective diameter of 2D particles to ensure a mean free path ls
    :param S: Surface of cell
    :param Ns: number of simulated particles
    :param ls: mean free path asked

    :return: the diameter of simulated particle (in m)
    """

    return S / (2 * 2 ** 0.5 * Ns * ls)



def getDiameterHenderson(S,Ns, ls, tol=1e-11, max_iter=1000):
    """
    compute the effective diameter, by taking into account the Henderson model
    NewtonRaphson method is used to extract d
    """
    n = Ns/S
    # first guess
    d = 1.0 / (2 * 2**0.5 * n * ls)

    for i in range(max_iter):

        phi = min( (n * np.pi * d ** 2) / 4.0 , 0.8)

        Z = (1.0 + (phi ** 2) / 8.0) / (1-phi) ** 2
        delta = (np.pi * d) / (4*np.sqrt(2) * (Z - 1.0)) - ls

        if abs(delta) < tol:
            return d

        h = d * 1e-3
        phi_h = (n * np.pi * (d + h) ** 2) / 4.0
        Z_h = (1.0 + (phi_h ** 2) / 8.0) / (1.0 - phi_h) ** 2
        delta_dh = (np.pi * (d + h)) / (4*np.sqrt(2) * (Z_h - 1.0)) - ls
        df = (delta_dh - delta) / h

        if df == 0 : break
        d = d - delta / df


    return d


# Exemple d'appel
# d_final = get_diameter_numba(0.42, 0.5)


def getMeanSquareVelocity(KbS, ms, temperature):
    """
    compute the mean square velocity (in 2D)
    :param KbS: boltzmann constant used in simulation
    :param ms: mass used in simulation
    :param temperature: real temperature
    :return: mean square velocity vStar in m/s
    """
    return (2 * KbS / ms * temperature) ** 0.5


def getDtCollision(meanSquareVelocity, ls, drOverLs):
    """
    compute optimal time step in order to capture particle collisions
    :param meanSquareVelocity: mean quare velocity v*
    :param ls: mean free path
    :param drOverLs: ratio between expected displacement during one time step and mean free path
    :return: optimal time step in order not to miss particle/particle collision
    """
    return ls / meanSquareVelocity * drOverLs


def getDtNoCollision(meanSquareVelocity, L):
    """
       compute optimal time step in order to capture particle collisions
       :param meanSquareVelocity: mean quare velocity v*
       :param L: Length of cell
       :return: optimal time step in order not to miss particle/cell boundaries collisions
       """
    return L / meanSquareVelocity * 0.1
