import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numbaAccelerated
import thermo
import time

INITSIZEEXTRARATIO = 1.2

MASS = 4.83e-26  # kg ; mean mass of air particle
Kb = 1.38e-23  # USI ; Boltzmann constant
DIAMETER = 0.37e-9  # m ; effective diameter of average air particle
H = 1  # m ; S*H = V

DEAD = 0
LEFT = 1
RIGHT = 2


class Cell:
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

    @classmethod
    def thermodynamicSetup(cls, initTemp, length, width, initPressure, nbPartTarget):
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
        :return: None
        """
        cls.width = width
        cls.length = length
        cls.initTemp = initTemp
        cls.initPressure = initPressure
        cls.nbPartTarget = nbPartTarget
        cls.surface = cls.length * cls.width
        cls.volume = cls.length * cls.width * H
        cls.kbs = thermo.getKbSimu(cls.initPressure, cls.volume, cls.initTemp, nbPartTarget)
        cls.ms = thermo.getMSimu(MASS, Kb, cls.kbs)
        cls.ds = thermo.getDiameter(DIAMETER, cls.surface, H, cls.nbPartTarget, cls.initPressure, Kb, cls.initTemp)

        cls.vStar = thermo.getMeanSquareVelocity(cls.kbs, cls.ms, cls.initTemp)
        cls.meanFreePath = thermo.getMeanFreePathSimulated(cls.surface, cls.ds, cls.nbPartTarget)
        cls.dt = thermo.getDtCollision(cls.vStar, cls.meanFreePath)  # time step is computed by global mean square velocity,
        # which may be different from cell to cell

        print("%%%%%%%%%%%%%%%%%%")

        print("Kbs = ", "{:.2e}".format(cls.kbs), "J/K")
        print("ms = ", "{:.2e}".format(cls.ms), "kg")
        print("d = ", "{:.2e}".format(cls.ds), "m")
        print("v* = ", "{:.2e}".format(cls.vStar), "m/s")
        print("dOM/L = v*dt/L = ", "{:.2e}".format(cls.vStar * cls.dt / cls.length))
        print("dOM/d = v*dt/d = ", "{:.2e}".format(cls.vStar * cls.dt / cls.ds))
        print("l : ", thermo.getMeanFreePathSimulated(cls.surface, cls.ds, cls.nbPartTarget), " m")
        print("tau : ", thermo.getMeanFreePathSimulated(cls.surface, cls.ds, cls.nbPartTarget) / cls.vStar, " s")
        print("dt : ", cls.dt, " s")
        print("%%%%%%%%%%%%%%%%%%")
        print()

    def __init__(self, partRatio, effectiveTemp):
        """
        Create a cell
        :param partRatio: ratio of particle (nbPart = nbPartTarget * ratio)
        :param effectiveTemp: temperature required for this cell (May be different than Cell.initTemp)
        """
        self.nbPart = partRatio * Cell.nbPartTarget
        self.arraySize = int(self.nbPart * INITSIZEEXTRARATIO)

        self.nbSearch = 5  # max index spacing between particles which may collide
        # this value should be computed according to amount of parts and their sizes

        # creation of arrays
        self.xs = np.empty(self.arraySize, dtype=float)
        self.ys = np.empty(self.arraySize, dtype=float)
        self.vxs = np.empty(self.arraySize, dtype=float)
        self.vys = np.empty(self.arraySize, dtype=float)
        self.wheres = np.ones(self.arraySize, dtype=np.int8) * DEAD
        self.colors = np.zeros(self.arraySize, dtype=float)

        self.positions = np.zeros((self.arraySize, 2), dtype=np.float)

        self.instantPressure = Cell.initPressure * partRatio
        self.averagedPressure = Cell.initPressure * partRatio

        self.temperature = effectiveTemp

        self.nbCollision = 0

        # living particles
        indices = np.linspace(0, self.arraySize - 1, self.nbPart, dtype=np.int32)
        if self.nbPart != len(np.unique(indices)):
            print("bad particle number ", self.nbPart, self.arraySize)
            exit(1)

        self.wheres[indices] = LEFT

        # init of locations and velocities
        self.randomInit(effectiveTemp)

    def randomInit(self, effectiveTemp):
        vStar = thermo.getMeanSquareVelocity(Cell.kbs, Cell.ms, effectiveTemp)

        self.vxs = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)
        self.vys = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)

        # enforce true temperature
        indices = np.nonzero(self.wheres != DEAD)
        vStarComputed = (np.average(self.vxs[indices] ** 2 + self.vys[indices] ** 2)) ** 0.5
        ratio = (vStarComputed / vStar)
        self.vxs /= ratio
        self.vys /= ratio

        # locations and states
        self.xs = (1 + np.random.random(self.arraySize)) * Cell.length / 2
        for i in range(self.arraySize):
            self.ys[i] = (i + 0.5) * Cell.width / self.arraySize

        self.sort()

    def advect(self):
        self.xs += self.vxs * Cell.dt
        self.ys += self.vys * Cell.dt
        # self.vys -= 2000000 * Cell.dt

    def computeTemperature(self):
        self.temperature = numbaAccelerated.computeAverageTemperature(self.vxs, self.vys, self.wheres, Cell.ms,
                                                                      Cell.kbs)

    def computePressure(self, fup, fdown):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :return: None
        """

        alpha = Cell.vStar * Cell.dt / Cell.length /10
        self.instantPressure = (fup + fdown) / (2 * Cell.length)
        self.averagedPressure = alpha * self.instantPressure + (1 - alpha) * self.averagedPressure

    def wallBounce(self):

        fup, fdown = numbaAccelerated.staticWallInterraction(self.ys, self.vys, self.wheres, Cell.width, Cell.dt,
                                                             Cell.ms)
        self.computePressure(fup, fdown)

        # left wall
        self.vxs = np.where(self.xs > 0, self.vxs, -self.vxs)
        self.xs = np.where(self.xs > 0, self.xs, - self.xs)

        # right wall
        self.vxs = np.where(self.xs < Cell.length, self.vxs, -self.vxs)
        self.xs = np.where(self.xs < Cell.length, self.xs, 2 * Cell.length - self.xs)

    #######################################

    def collide(self):
        """
        Compute collisions between particles
        nbNeighbour is the number of neighbours par particle i which are checked
        :return:
        """
        nbNeighbour = 2*int(self.nbPart**0.5)
        self.nbCollision += numbaAccelerated.detectAllCollisions(self.xs, self.ys, self.vxs, self.vys, self.wheres,
                                                                 self.colors, Cell.dt, Cell.ds, nbNeighbour)

    #######################################

    def sort(self):
        return numbaAccelerated.sortCell(self.xs, self.ys, self.vxs, self.vys, self.wheres, self.colors)

    def update(self):

        self.advect()

        self.sort()

        self.collide()

        self.computeTemperature()

        self.wallBounce()



    def ouputBuffer(self):
        numbaAccelerated.twoArraysToOne(self.xs, self.ys, self.wheres, self.positions)
        return self.positions

    def plot(self):
        # plot using matplotlib

        indices = np.nonzero(self.wheres != DEAD)

        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        plt.subplot(221)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.axis("equal")
        fig = plt.gcf()
        ax = fig.gca()
        plt.xlim(0, Cell.length)
        plt.ylim(0, Cell.width)
        circles = []
        for i in indices[0]:
            circles.append(plt.Circle((self.xs[i], self.ys[i]), Cell.ds))

        p = PatchCollection(circles)
        p.set_color("r")
        ax.add_collection(p)
        plt.grid()

        plt.subplot(222)
        plt.xlabel("vx")
        plt.ylabel("count")

        plt.hist(self.vxs[indices], bins='auto')

        plt.subplot(223)
        plt.xlabel("vy")
        plt.ylabel("count")
        plt.hist(self.vys[indices], bins='auto')

        plt.show()
