# set of thermo functions used to compute simulation values from real values


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


def getMeanSquareVelocity(KbS, ms, temperature):
    """
    compute the mean square velocity (in 2D)
    :param KbS: boltzmann constant used in simulation
    :param ms: mass used in simulation
    :param temperature: real temperature
    :return: mean square velocity vStar in m/s
    """
    return (2 * KbS / ms * temperature) ** 0.5


def getDtCollision(meanSquareVelocity, l):
    """
    compute optimal time step in order to capture particle collisions
    :param meanSquareVelocity: mean quare velocity v*
    :param d: mean free path
    :return: optimal time step in order not to miss particle/particle collision
    """
    return l / meanSquareVelocity * 0.05


def getDtNoCollision(meanSquareVelocity, L):
    """
       compute optimal time step in order to capture particle collisions
       :param meanSquareVelocity: mean quare velocity v*
       :param L: Length of cell
       :return: optimal time step in order not to miss particle/cell boundaries collisions
       """
    return L / meanSquareVelocity * 0.1
