import thermo

MASS = 4.83e-26  # kg ; mean mass of air particle
Kb = 1.38e-23  # USI ; Boltzmann constant
DIAMETER = 0.37e-9  # m ; effective diameter of average air particle
Z = 1  # m ; S*Z = V

DEAD = 0
LEFT = 1
RIGHT = 2

INITSIZEEXTRARATIO = 1.5


class ComputedConstants:
    meanFreePath = None
    vStar = None
    nbPartTarget = None
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
    alphaAveraging = 0.025  # temperature and pressure averaging

    # for wall


    # for opengl
    resX = None
    resY = None

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
        cls.volume = cls.length * cls.width * Z
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
        print("diameter = ", "{:.5e}".format(cls.ds), "m")
        print("v* = ", "{:.5e}".format(cls.vStar), "m/s")
        print("dOM/L = v*dt/L = ", "{:.2e}".format(cls.vStar * cls.dt / cls.length))
        print("dOM/d = v*dt/d = ", "{:.2e}".format(cls.vStar * cls.dt / cls.ds))
        print("l : ", "{:.5e}".format(cls.ls), " m")
        print("tau : ", "{:.2e}".format(cls.ls / cls.vStar), " s")
        print("d/l : ", "{:.2e}".format(cls.ds / cls.ls), " ")
        print("dt : ", "{:.5e}".format(cls.dt), " s")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()

    @classmethod
    def thermodynamicSetupFixedDiameter(cls, initTemp, length, width, initPressure, nbPartTarget, ds):
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
        :param ds: diameter of simulated particles
        :return: None
        """
        cls.ds = ds
        cls.width = width
        cls.length = length
        cls.initTemp = initTemp
        cls.initPressure = initPressure
        cls.nbPartTarget = nbPartTarget

        cls.surface = cls.length * cls.width
        cls.ls = thermo.getMeanFreePathSimulated(cls.surface, cls.ds, cls.nbPartTarget)
        cls.volume = cls.length * cls.width * Z
        cls.kbs = thermo.getKbSimu(cls.initPressure, cls.volume, cls.initTemp, nbPartTarget)
        cls.ms = thermo.getMSimu(MASS, Kb, cls.kbs)

        cls.vStar = thermo.getMeanSquareVelocity(cls.kbs, cls.ms, cls.initTemp)
        cls.dt = thermo.getDtCollision(cls.vStar, cls.ls)

        cls.time = 0.
        cls.it = 0

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Kbs = ", "{:.2e}".format(cls.kbs), "J/K")
        print("ms = ", "{:.2e}".format(cls.ms), "kg")
        print("diameter = ", "{:.5e}".format(cls.ds), "m")
        print("v* = ", "{:.5e}".format(cls.vStar), "m/s")
        print("dOM/L = v*dt/L = ", "{:.2e}".format(cls.vStar * cls.dt / cls.length))
        print("dOM/d = v*dt/d = ", "{:.2e}".format(cls.vStar * cls.dt / cls.ds))
        print("l : ", "{:.5e}".format(cls.ls), " m")
        print("tau : ", "{:.2e}".format(cls.ls / cls.vStar), " s")
        print("d/l : ", "{:.2e}".format(cls.ds / cls.ls), " ")
        print("dt : ", "{:.5e}".format(cls.dt), " s")
        print("fill ratio s_parts/S : ", "{:.5e}".format( (cls.ds/2)**2 * 3.14159 * cls.nbPartTarget / cls.surface ))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print()
