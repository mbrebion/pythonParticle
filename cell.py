import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numbaAccelerated
import thermo, tracker

INITSIZEEXTRARATIO = 1.2

MASS = 4.83e-26  # kg ; mean mass of air particle
Kb = 1.38e-23  # USI ; Boltzmann constant
DIAMETER = 0.37e-9  # m ; effective diameter of average air particle
H = 1  # m ; S*H = V

DEAD = 0
LEFT = 1
RIGHT = 2


class Cell:
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

    def __init__(self, partRatio, effectiveTemp):
        """
        Create a cell
        :param partRatio: ratio of particle (nbPart = nbPartTarget * ratio)
        :param effectiveTemp: temperature required for this cell (It may be different from Cell.initTemp)
        """
        self.nbPart = partRatio * Cell.nbPartTarget
        self.arraySize = int(self.nbPart * INITSIZEEXTRARATIO)
        nb = max(5, int(0.4 * self.nbPart ** 0.5) + 1)
        self.histo = np.zeros(nb, dtype=int)
        # the amount of neighbors checked for collisions is adapted dynamically to ensure fast computations
        # and miss less than 0.1 % of collisions

        # time and iterations count
        self.it = 0
        self.time = 0

        # creation of arrays
        self.xs = np.empty(self.arraySize, dtype=float)
        self.ys = np.empty(self.arraySize, dtype=float)
        self.vxs = np.empty(self.arraySize, dtype=float)
        self.vys = np.empty(self.arraySize, dtype=float)
        self.wheres = np.ones(self.arraySize, dtype=np.int32) * DEAD
        self.colors = np.zeros(self.arraySize, dtype=float)

        # thermodynamic instant and averaged variables
        self.instantPressure = Cell.initPressure * partRatio
        self.averagedPressure = Cell.initPressure * partRatio
        self.temperature = effectiveTemp

        # living particles
        indices = np.linspace(0, self.arraySize - 1, self.nbPart, dtype=np.int32)
        if self.nbPart != len(np.unique(indices)):
            print("bad particle number ", self.nbPart, self.arraySize)
            exit(1)
        self.wheres[indices] = range(1, self.nbPart + 1)

        # output buffer
        self.upToDate = False
        self.positions = np.zeros((self.arraySize, 2), dtype=np.float)  # not used for computations but for opengl draws

        # init of locations and velocities
        self.randomInit(effectiveTemp)

        # trackers
        self.trackers = []

    def addTracker(self, id):
        self.trackers.append(tracker.Tracker(self, id))

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
        self.xs = (np.random.random(self.arraySize)) * Cell.length
        for i in range(self.arraySize):
            self.ys[i] = (i + 0.5) * Cell.width / self.arraySize

        self.sort()

    def advect(self):
        self.xs += self.vxs * Cell.dt
        self.ys += self.vys * Cell.dt

    def computeTemperature(self):
        self.temperature = numbaAccelerated.computeAverageTemperature(self.vxs, self.vys, self.wheres, Cell.ms, Cell.kbs)

    def computePressure(self, fup, fdown):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :return: None
        """

        alpha = Cell.vStar * Cell.dt / Cell.length / 10
        self.instantPressure = (fup + fdown) / (2 * Cell.length)
        self.averagedPressure = alpha * self.instantPressure + (1 - alpha) * self.averagedPressure

    def wallBounce(self):

        # up and down
        fup, fdown = numbaAccelerated.staticWallInterractionUpAndDown(self.ys, self.vys, self.wheres, Cell.width, Cell.dt, Cell.ms)
        self.computePressure(fup, fdown)

        # left wall
        numbaAccelerated.staticWallInterractionLeft(self.xs, self.vxs, self.wheres, Cell.length)
        # right wall
        numbaAccelerated.staticWallInterractionRight(self.xs, self.vxs, self.wheres, Cell.length)

    #######################################

    def collide(self):
        """
        Compute collisions between particles
        nbNeighbour is the number of neighbours par particle i which are checked
        :return:
        """

        self.colors *= 0.92
        numbaAccelerated.detectAllCollisions(self.xs, self.ys, self.vxs, self.vys, self.wheres, self.colors, Cell.dt, Cell.ds, self.histo)

    #######################################

    def sort(self):
        return numbaAccelerated.sortCell(self.xs, self.ys, self.vxs, self.vys, self.wheres, self.colors)

    def updateConstants(self):
        self.upToDate = False  # invalidate position buffer

        Cell.it += 1  # to be moved upper once cells are gathered in broader class
        Cell.time += Cell.dt
        self.computeTemperature()

    def update(self):

        self.advect()

        self.sort()

        self.collide()

        self.wallBounce()

        self.updateConstants()

        if Cell.it % 100 == 0:
            self.improveSpeed()

        for t in self.trackers:
            t.doMeasures()

    def improveSpeed(self):
        ln = len(self.histo)
        nbZeros = ln - np.count_nonzero(self.histo) - 1
        if nbZeros > 2:
            # Too many neighbors are searched for nothing
            self.histo = np.zeros(ln - nbZeros + 2, dtype=int)
            return
        sm = np.sum(self.histo)
        expectedTail = int(sm * 1e-3)
        if self.histo[-1] > expectedTail:
            # here, we might miss somme collisions.
            # statistic studies (not proved) tend to show that sum_(k+1,+oo)(histo) \approx histo(k)
            # when k is big enough. Thus, we check here is un-detection rate remains below 0.1 %

            more = int((np.log(self.histo[-1] + 1) - np.log(expectedTail + 1)) / np.log(2))
            # estimation of additional neighbors to be checked
            self.histo = np.zeros(ln + more, dtype=int)
            return

        if np.sum(self.histo) > 20 * self.nbPart:
            self.histo *= 0

    def getPositionsBuffer(self):
        if not self.upToDate:
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
