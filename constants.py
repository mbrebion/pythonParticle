import thermo

MASS = 4.83e-26  # kg ; mean mass of air particle
Kb = 1.38e-23  # USI ; Boltzmann constant
DIAMETER = 0.37e-9  # m ; effective diameter of average air particle
H = 1  # m ; S*H = V

DEAD = 0
LEFT = 1
RIGHT = 2


class ComputedConstants:
    meanFreePath = None
    vStar = None
    nbPartTarget = None
    nbPartCreated = None
    dt = None
    ds = None
    ms = None
    initTemp = None
    volume = None
    surface = None
    initPressure = None
    width = None
    length = None
    kbs = None
    time = None
    it = None
    nbCells = None
    decoloringRatio = 0.85

    @classmethod
    def thermodynamicSetup(cls, initTemp, length, width, initPressure, nbPartTarget, ls):
        """
        Compute thermodynamic values common to all cells
        :param initTemp: mean temperature used in simulation (in K)
        :param length: length of cells (in m)
        :param width: width of cells (between walls)
        :param initPressure: mean pressure used in simulation (in Pa)
        (pressure to be obtained with nbPartTarget particles)
        :param nbPartTarget: target number of particle in cell
        (used to compute simulation values for mass, diameter and boltzmann constant)
        The actual number of particles may then differ in cells, resulting to mean pressure
         being different from the one provided.
        :param ls: mean free path required (in m)
        :return: None
        """
        cls.ls = ls
        cls.width = width
        cls.length = length
        cls.initTemp = initTemp
        cls.initPressure = initPressure
        cls.nbPartTarget = nbPartTarget
        cls.surface = cls.length * cls.width
        cls.volume = cls.length * cls.width * H
        cls.kbs = thermo.getKbSimu(cls.initPressure, cls.volume, cls.initTemp, nbPartTarget)
        cls.ms = thermo.getMSimu(MASS, Kb, cls.kbs)
        cls.ds = thermo.getDiameter(cls.surface, cls.nbPartTarget, cls.ls)

        cls.vStar = thermo.getMeanSquareVelocity(cls.kbs, cls.ms, cls.initTemp)

        cls.dt = thermo.getDtCollision(cls.vStar, cls.ls)

        cls.time = 0.
        cls.it = 0

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Kbs = ", "{:.2e}".format(cls.kbs), "J/K")
        print("ms = ", "{:.2e}".format(cls.ms), "kg")
        print("d = ", "{:.3e}".format(cls.ds), "m")
        print("v* = ", "{:.2e}".format(cls.vStar), "m/s")
        print("dOM/L = v*dt/L = ", "{:.2e}".format(cls.vStar * cls.dt / cls.length))
        print("dOM/d = v*dt/d = ", "{:.2e}".format(cls.vStar * cls.dt / cls.ds))
        print("l : ", "{:.2e}".format(cls.ls), " m")
        print("tau : ", "{:.2e}".format(cls.ls / cls.vStar), " s")
        print("dt : ", "{:.2e}".format(cls.dt), " s")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()
