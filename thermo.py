# set of thermo functions used to compute simulation values from real values
from math import pi

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


def getMSimu(mass,Kb,Kbs):
    """
    compute mass of simulated particles
    :param Kb: real Boltzmann constant
    :param Kbs: used Boltzmann constant
    :param mass: target mass of real particles

    :return: effective mass used in 2D simulations (in kg)
    """
    return mass * Kbs / Kb

def getMeanFreePathSimulated(S,ds,Ns):
    """
    compute the average mean free path
    :param S: Surface of cell
    :param ds: diameter (simulated) of particle
    :param Ns: number of simulated particle
    :return: average mean free path
    """
    return S/(2 * 2**0.5 * ds * Ns)

def getDiameter(d,S,H,Ns,P,kb,T):
    """
    compute the effective diameter of 2D particles to preserve ratio X = l / Delta x
    where l in the mean free path and Delta x, the average distance between particles
    :param d: real diameter for 3D gaz
    :param S: Surface of cell
    :param H: length so that SH = V
    :param Ns: number of simulated particles
    :param P: target pressure (in Pa)
    :param kb: real Boltzmann constant
    :param T: target temperature (in K)

    :return: the effective diameter (in m)
    """

    return pi/2 * d**2 * (S/Ns)**(1/2) * (P/(kb*T))**(2/3) * 10

def getMeanSquareVelocity(KbS ,ms,temperature):
    """
    compute the mean square velocity
    :param KbS: boltzmann constant used in simulation
    :param ms: mass used in simulation
    :param temperature: real temperature
    :return: mean square velocity vStar in m/s
    """
    return (2 * KbS/ ms * temperature)**0.5


def getDtCollision(meanSquareVelocity,l):
    """
    compute optimal time step in order to capture particle collisions
    :param meanSquareVelocity: mean quare velocity v*
    :param d: mean free path
    :return: optimal time step in order not to miss particle/particle collision
    """
    return l / meanSquareVelocity * 0.02


def getDtNoCollision(meanSquareVelocity,L):
    """
       compute optimal time step in order to capture particle collisions
       :param meanSquareVelocity: mean quare velocity v*
       :param L: Length of cell
       :return: optimal time step in order not to miss particle/cell boundaries collisions
       """
    return L / meanSquareVelocity * 0.1