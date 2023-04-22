import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numbaAccelerated
from constants import *
from coords import Coords

INITSIZEEXTRARATIO = 1.2


class Cell:

    def __init__(self, nbPart, effectiveTemp, left, right, startIndex):
        """
        Create a cell
        :param nbPart: effective number of part in cell
        :param effectiveTemp: temperature required for this cell (It may be different from Cell.initTemp)
        :param left: left coordinate of cell
        :param right: right coordinate of cell
        :param startIndex: first id of particle lying in this cell
        """

        self.left = left
        self.right = right
        self.length = self.right - self.left
        self.startIndex = startIndex

        self.nbPart = nbPart
        self.arraySize = int(self.nbPart * INITSIZEEXTRARATIO)
        nb = max(self.nbPart // 100,20)
        #nb =self.nbPart*2
        self.histo = np.zeros(nb, dtype=int)
        # the amount of neighbors checked for collisions is adapted dynamically to ensure fast computations
        # and miss less than 0.1 % of collisions

        # time and iterations count
        self.it = 0
        self.time = 0

        # creation of arrays
        self.coords = Coords(self.arraySize)

        # creation of swap arrays
        swapSize = int(0.05 * self.arraySize)
        self.leftSwap = Coords(swapSize)
        self.rightSwap = Coords(swapSize)

        # thermodynamic instant and averaged variables
        self.instantPressure = ComputedConstants.initPressure * nbPart / ComputedConstants.initPressure * ComputedConstants.nbCells
        self.averagedPressure = self.instantPressure
        self.temperature = effectiveTemp

        # living particles

        indices = np.linspace(0, self.arraySize - 1, self.nbPart, dtype=np.int32)
        if self.nbPart != len(np.unique(indices)):
            print("bad particle number ", self.nbPart, self.arraySize)
            exit(1)
        self.coords.wheres[indices] = range(1, self.nbPart + 1)
        self.coords.wheres[indices] += self.startIndex

        # output buffer
        self.upToDate = False
        self.positions = np.zeros((self.arraySize, 2), dtype=np.float)  # not used for computations but for opengl draws

        # init of locations and velocities
        self.randomInit(effectiveTemp)

        # neighboring cells
        self.leftCell = None
        self.rightCell = None

    def randomInit(self, effectiveTemp):
        vStar = thermo.getMeanSquareVelocity(ComputedConstants.kbs, ComputedConstants.ms, effectiveTemp)

        self.coords.vxs = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)
        self.coords.vys = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)

        # enforce true temperature
        indices = np.nonzero(self.coords.wheres != DEAD)
        vStarComputed = (np.average(self.coords.vxs[indices] ** 2 + self.coords.vys[indices] ** 2)) ** 0.5
        ratio = (vStarComputed / vStar)
        self.coords.vxs /= ratio
        self.coords.vys /= ratio

        # locations and states
        self.coords.xs = self.left + (np.random.random(self.arraySize)) * self.length
        for i in range(self.arraySize):
            self.coords.ys[i] = (i + 0.5) * ComputedConstants.width / self.arraySize

        self.sort()

        self.coords.updateTuple()

    def advect(self):
        self.coords.xs += self.coords.vxs * ComputedConstants.dt
        self.coords.ys += self.coords.vys * ComputedConstants.dt

    def computeTemperature(self):
        self.temperature = numbaAccelerated.computeAverageTemperature(self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.ms,
                                                                      ComputedConstants.kbs)

    def computePressure(self, fup, fdown):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :return: None
        """

        alpha = ComputedConstants.vStar * ComputedConstants.dt / ComputedConstants.length / 10
        self.instantPressure = (fup + fdown) / (2 * ComputedConstants.length)
        self.averagedPressure = alpha * self.instantPressure + (1 - alpha) * self.averagedPressure

    def wallBounce(self):
        # up and down
        fup, fdown = numbaAccelerated.staticWallInterractionUpAndDown(self.coords.ys, self.coords.vys, self.coords.wheres, ComputedConstants.width,
                                                                      ComputedConstants.dt,
                                                                      ComputedConstants.ms)
        self.computePressure(fup, fdown)

        # left wall
        if self.leftCell is None:
            numbaAccelerated.staticWallInterractionLeft(self.coords.xs, self.coords.vxs, self.coords.wheres, self.left)

        # right wall
        if self.rightCell is None:
            numbaAccelerated.staticWallInterractionRight(self.coords.xs, self.coords.vxs, self.coords.wheres, self.right)

    ##############################################################
    ####################          Swapping        ################
    ##############################################################

    def applySwap(self):
        """
        swap particles between cells when necessary
        :return: None
        """
        if self.leftCell is not None:
            numbaAccelerated.moveSwapToNeighbor(*self.leftSwap.tpl, *self.leftCell.coords.tpl, self.leftSwap.alive, ComputedConstants.width)

        if self.rightCell is not None:
            numbaAccelerated.moveSwapToNeighbor(*self.rightSwap.tpl, *self.rightCell.coords.tpl, self.rightSwap.alive, ComputedConstants.width)

    def prepareSwap(self):
        """
        identify particles to be swapped and move them to swap arrays
        :return: None
        """
        if self.leftCell is not None:
            self.leftSwap.alive = numbaAccelerated.moveToSwap(*self.coords.tpl, *self.leftSwap.tpl, self.left, False)

        if self.rightCell is not None:
            self.rightSwap.alive = numbaAccelerated.moveToSwap(*self.coords.tpl, *self.rightSwap.tpl, self.right, True)

    ##############################################################

    def collide(self):
        """
        Compute collisions between particles
        nbNeighbour is the number of neighbours par particle i which are checked
        :return: None
        """

        self.coords.colors *= ComputedConstants.decoloringRatio
        numbaAccelerated.detectAllCollisions(*self.coords.tpl,
                                             ComputedConstants.dt,
                                             ComputedConstants.ds,
                                             self.histo)

    #######################################

    def sort(self):
        return self.coords.sort()

    def updateConstants(self):
        self.upToDate = False  # invalidate position buffer

        self.computeTemperature()

    def update(self):
        self.advect()

        self.sort()

        self.collide()

        self.wallBounce()

        self.updateConstants()

        if ComputedConstants.it % 1500 == 0:
            self.improveSpeed()

    def improveSpeed(self):
        ln = len(self.histo)
        h = np.array(self.histo,dtype=float)
        h /= np.sum(h)
        h *= 100
        k = 1
        for i in range(1, ln):
            if h[i] > h[1] / 4:
                k = i

        k = min(int(ln * 0.8), k)  # safeguard
        Sk = 0
        for i in range(k, ln):
            Sk += h[i]
        xk = 1 - h[k] / Sk

        DeltaMax = int(0.5 + k + np.log(0.1 / Sk) / np.log(xk))

        self.histo = np.zeros(DeltaMax, dtype=int)
        print(DeltaMax)


    def improveSpeedOld(self):
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

    def count(self):
        return numbaAccelerated.countAlive(self.coords.wheres)

    def getPositionsBuffer(self):
        if not self.upToDate:
            numbaAccelerated.twoArraysToOne(self.coords.xs, self.coords.ys, self.coords.wheres, self.positions)

        return self.positions

    def printHisto(self):
        out = "["
        for v in self.histo:
            out += str(v) + ","
        out = out[0:-1] + "]"
        return out

    def plot(self):
        # plot using matplotlib

        indices = np.nonzero(self.coords.wheres != DEAD)

        plt.rcParams["figure.figsize"] = [7.50, 7.50]
        plt.rcParams["figure.autolayout"] = True

        plt.subplot(221)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.axis("equal")
        fig = plt.gcf()
        ax = fig.gca()
        plt.xlim(0, ComputedConstants.length)
        plt.ylim(0, ComputedConstants.width)
        circles = []
        for i in indices[0]:
            circles.append(plt.Circle((self.coords.xs[i], self.coords.ys[i]), ComputedConstants.ds))

        p = PatchCollection(circles)
        p.set_color("r")
        ax.add_collection(p)
        plt.grid()

        plt.subplot(222)
        plt.xlabel("vx")
        plt.ylabel("count")

        plt.hist(self.coords.vxs[indices], bins='auto')

        plt.subplot(223)
        plt.xlabel("vy")
        plt.ylabel("count")
        plt.hist(self.coords.vys[indices], bins='auto')

        plt.show()
