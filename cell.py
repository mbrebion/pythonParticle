import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numbaAccelerated
from constants import *
from coords import Coords


class Cell:
    colorCollisions = True

    def __init__(self, nbPart, effectiveTemp, left, right, startIndex, nbPartTarget=None):
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
        if nbPartTarget is None:
            nbPartTarget = self.nbPart
        self.arraySize = int(nbPartTarget * INITSIZEEXTRARATIO)
        nb = max(nbPartTarget // 100, 30)

        self.histo = np.zeros(nb, dtype=int)
        # the amount of neighbors checked for collisions is adapted dynamically to ensure fast computations
        # and miss less than 0.1 % of collisions

        # time and iterations count
        self.it = 0
        self.time = 0

        # creation of arrays
        self.coords = Coords(self.arraySize)

        # creation of swap arrays
        swapSize = int(0.1 * self.arraySize)
        self.leftSwap = Coords(swapSize)
        self.rightSwap = Coords(swapSize)

        # thermodynamic instant and averaged variables
        self.instantPressure = ComputedConstants.initPressure
        self.averagedPressure = self.instantPressure
        self.temperature = effectiveTemp
        self.averagedTemperature = effectiveTemp
        self.ecl = 0.  # left of wall kinetic energy
        self.ecr = 0.

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

        # wall
        self.wall = None

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

    ##############################################################
    ################## Compute thermodynamic      ################
    ##############################################################

    def computeKineticEnergy(self):
        """
        :param x: wall location
        :return: kinetic energies (left and right of wall)
        """
        self.ecl, self.ecr = numbaAccelerated.computeEcs(self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.ms)
        return self.ecl, self.ecr

    def computeTemperature(self):
        self.temperature = numbaAccelerated.computeAverageTemperature(self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.ms,
                                                                      ComputedConstants.kbs)

        alpha = ComputedConstants.alphaAveraging
        self.averagedTemperature = alpha * self.temperature + (1 - alpha) * self.averagedTemperature

        return self.temperature

    def computePressure(self, fup, fdown):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :return: None
        """

        alpha = ComputedConstants.vStar * ComputedConstants.dt / ComputedConstants.length
        self.instantPressure = (fup + fdown) / (2 * ComputedConstants.length)
        self.averagedPressure = alpha * self.instantPressure + (1 - alpha) * self.averagedPressure

    def count(self):
        return numbaAccelerated.countAlive(self.coords.wheres)

    def countLeft(self, x):
        out = numbaAccelerated.countAliveLeft(self.coords.xs, self.coords.wheres, x)
        return out

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
    #################   Wall and Collision       #################
    ##############################################################

    def wallBounce(self):
        # up and down
        fup, fdown = numbaAccelerated.staticWallInterractionUpAndDown(self.coords.ys, self.coords.vys, self.coords.wheres, ComputedConstants.width,
                                                                      ComputedConstants.dt,
                                                                      ComputedConstants.ms)
        self.computePressure(fup, fdown)

        # moving Wall
        if self.wall is not None:
            fpl, fpr = numbaAccelerated.movingWallInteraction(self.coords.xs, self.coords.vxs, self.coords.wheres, self.wall.location(), self.wall.velocity(),
                                                              ComputedConstants.dt, ComputedConstants.ms, self.wall.mass())
            self.wall.addToForce(fpl, fpr)

        # left wall
        if self.leftCell is None:
            numbaAccelerated.staticWallInteractionLeft(self.coords.xs, self.coords.vxs, self.coords.wheres, self.left)

        # right wall
        if self.rightCell is None:
            numbaAccelerated.staticWallInteractionRight(self.coords.xs, self.coords.vxs, self.coords.wheres, self.right)

    def collide(self):
        """
        Compute collisions between particles
        nbNeighbour is the number of neighbours par particle i which are checked
        :return: None
        """
        if Cell.colorCollisions:
            self.coords.colors *= ComputedConstants.decoloringRatio

        x = 1e9
        if self.wall is not None:
            x = self.wall.location()

        numbaAccelerated.detectAllCollisions(*self.coords.tpl,
                                             ComputedConstants.dt,
                                             ComputedConstants.ds,
                                             self.histo, Cell.colorCollisions)

    def improveCollisionDetectionSpeed(self):
        """
        Improve collision detectionSpeed by estimated the smallest number of neighbor to check for collision
        :return: None
        """
        ln = len(self.histo)
        h = np.array(self.histo, dtype=float)
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

    def sort(self):
        return self.coords.sort()

    ##############################################################
    #################      Helper functions     ##################
    ##############################################################

    def updateConstants(self):
        self.upToDate = False  # invalidate position buffer
        self.computeTemperature()

    def getPositionsBuffer(self):
        if not self.upToDate:
            numbaAccelerated.twoArraysToOne(self.coords.xs, self.coords.ys, self.coords.wheres, self.positions)

        return self.positions

    def advect(self):
        self.coords.xs += self.coords.vxs * ComputedConstants.dt
        self.coords.ys += self.coords.vys * ComputedConstants.dt

    def middle(self):
        return (self.left + self.right) * 0.5

    def updateIndicesAccordingToWall(self, x):
        """
        Negates indices of particle initially left of wall
        :param x: wall location
        :return: None
        """
        for i in range(len(self.coords.xs)):
            if self.coords.xs[i] < x:
                self.coords.wheres[i] *= -1

    ##############################################################
    ###################        Update cell       #################
    ##############################################################

    def update(self):
        self.advect()

        self.sort()

        self.collide()

        self.wallBounce()

        self.updateConstants()

        if ComputedConstants.it + 400 % 500 == 0:
            self.improveCollisionDetectionSpeed()
