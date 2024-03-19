import math

import numpy as np
import numbaAccelerated
from constants import *
from coords import Coords
from thermo import getMeanSquareVelocity


class Cell:
    coloringPolicy = "none" # might be "coll", "vx" or "fixed"
    collision = True

    def __init__(self, nbPart, effectiveTemp, left, right, startIndex, nbPartTarget=None,v_xYVelocityProfile=None):
        """
        Create a cell
        :param nbPart: effective number of part in cell
        :param effectiveTemp: temperature required for this cell (It may be different from Cell.initTemp)
        :param left: left coordinate of cell
        :param right: right coordinate of cell
        :param startIndex: first id of particle lying in this cell
        :param nbPartTarget: target numer of particles. Useful if initial number of particle is way lower than the target
        :param v_xYVelocityProfile: v_x(y) mean velocity profile to be imposed at startup
        """

        self.left = left
        self.right = right
        self.length = self.right - self.left
        self.startIndex = startIndex
        if v_xYVelocityProfile is not None:
            self.v_xYVelocityProfile = v_xYVelocityProfile
        else :
            self.v_xYVelocityProfile = self.zeroV_xYVelocityProfile

        self.nbPart = nbPart
        if nbPartTarget is None:
            nbPartTarget = self.nbPart
        self.arraySize = int(nbPartTarget * INITSIZEEXTRARATIO)
        nb = min(int(nbPartTarget**0.5+40), 400)
        nb = 1000

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

    def zeroV_xYVelocityProfile(self,y):
        return 0.

    def randomInit(self, effectiveTemp):
        vStar = thermo.getMeanSquareVelocity(ComputedConstants.kbs, ComputedConstants.ms, effectiveTemp)

        self.coords.vxs = np.random.normal(0, vStar/2**0.5, self.arraySize)
        self.coords.vys = np.random.normal(0, vStar/2**0.5, self.arraySize)

        # enforce true temperature
        indices = np.nonzero(self.coords.wheres != DEAD)
        vStarComputed = (np.average(self.coords.vxs[indices] ** 2 + self.coords.vys[indices] ** 2)) ** 0.5
        ratio = (vStarComputed / vStar)
        self.coords.vxs /= ratio
        self.coords.vys /= ratio

        # locations and states
        if ComputedConstants.fillRatio < 0.1:
            # random init
            self.coords.xs = self.left + (np.random.random(self.arraySize)) * self.length
            for i in range(self.arraySize):
                self.coords.ys[i] = (i + 0.5) * ComputedConstants.width / self.arraySize
        else:
            # cristal like init
            L = self.length
            H = ComputedConstants.width
            deltax = math.sqrt( 2 * L*H / self.nbPart / math.sqrt(3.))
            deltay = deltax * math.sqrt(3)/2

            nx = int(0.5 + self.length / deltax)
            ny = int(0.5 + self.length / deltay)

            if (nx - 1) * ny >= self.nbPart:
                nx = nx - 1

            if nx*(ny-1)>=self.nbPart:
                ny = ny-1

            deltax = deltax * (nx / (nx+1))**0.5
            deltay = deltay * (ny / (ny+1))**0.5

            id0 = self.coords.wheres[indices[0][0]]-1
            for ind in indices[0]:
                id = self.coords.wheres[ind]-1

                i = (id-id0) % nx
                j = (id-id0) // nx
                self.coords.xs[ind] = self.left + i*deltax + deltax/4 * (-1)**(j % 2) + 2*deltax/3
                self.coords.ys[ind] = j * deltay + 2*deltay/3

        for i in range(self.arraySize):
            self.coords.vxs[i] += self.v_xYVelocityProfile(self.coords.ys[i])

        self.sort()

        self.coords.updateTuple()

    ##############################################################
    ################## Compute thermodynamic      ################
    ##############################################################

    def computeSumVelocityLeftOfWall(self,wallLocation):
        return numbaAccelerated.computeSumVelocityLeftOfWall(self.coords.xs, self.coords.vxs, self.coords.wheres,wallLocation)

    def computeXVelocity(self):
        """
        :return: the averaged X velocity of the cell
        """
        return numbaAccelerated.ComputeXVelocity(self.coords.vxs,self.coords.wheres)

    def computeXVelocityBins(self,nbBins):
        """
        :param nbBins: number of bins
        :return: the averaged X velocity in bins
        """
        bins = np.array([0.]*nbBins)

        return numbaAccelerated.ComputeXVelocityBins(self.coords.ys, ComputedConstants.width,self.coords.vxs,self.coords.wheres,bins)

    def ComputeXVelocityBeforeWall(self,nbBins,span,wallLoc):
        """
        :param nbBins: number of bins
        :param span: x span of measures (measures are performed between Xwall-span and Xwall
        :param wallLoc : location of wall
        :return: the averaged X velocity in bins
        """

        bins = np.array([0.]*nbBins)
        counts = np.array([0.] * nbBins)
        return numbaAccelerated.ComputeXVelocityBeforeWall(self.coords.xs,wallLoc,span,self.coords.vxs,self.coords.wheres,bins,counts)

    def computeKineticEnergy(self):
        """
        :param x: wall location
        :return: kinetic energies (left and right of wall)
        """
        self.ecl, self.ecr = numbaAccelerated.computeEcs(self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.ms)
        self.ec = self.ecl + self.ecr
        return self.ecl, self.ecr

    def computeTemperature(self):
        ecl, ecr = numbaAccelerated.computeEcs(self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.ms)
        self.temperature = (ecl+ecr) / (self.nbPart * ComputedConstants.kbs)
        alpha = ComputedConstants.alphaAveraging
        self.averagedTemperature = alpha * self.temperature + (1 - alpha) * self.averagedTemperature

        return self.temperature

    def computePressure(self, fup, fdown, fleft, fright):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :param fleft: last computed force on left static wall (in N), negative if not provided
        :param fright: last computed force on right static wall (in N), negative if not provided
        :return: None
        """
        nbWall = 2
        self.instantPressure = (fup + fdown) / (1 * (self.right - self.left))
        if fleft >= 0:
            self.instantPressure += fleft / ComputedConstants.width
            nbWall += 1

        if fright >= 0:
            self.instantPressure += fright / ComputedConstants.width
            nbWall += 1
        self.instantPressure /= nbWall

        alpha = ComputedConstants.vStar * ComputedConstants.dt / ComputedConstants.length

        self.averagedPressure = alpha * self.instantPressure + (1 - alpha) * self.averagedPressure

    def count(self):
        ct = numbaAccelerated.countAlive(self.coords.wheres)
        if ct / self.arraySize > 0.92:
            print("coords arrays arrays nearly saturated : ", ct / self.arraySize)
        return ct

    def countLeft(self, x):
        out = numbaAccelerated.countAliveLeft(self.coords.xs, self.coords.wheres, x)
        return out

    def countRight(self, x):
        out = numbaAccelerated.countAliveRight(self.coords.xs, self.coords.wheres, x)
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


    def applyPeriodic(self):
        numbaAccelerated.movePeriodically(self.coords.xs, ComputedConstants.length)

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
        vStarBoundary = -1
        if ComputedConstants.boundaryTemperature > 0:
            vStarBoundary = getMeanSquareVelocity(ComputedConstants.kbs, ComputedConstants.ms, ComputedConstants.boundaryTemperature)

        fup, fdown = numbaAccelerated.staticWallInterractionUpAndDown(self.coords.ys, self.coords.vxs, self.coords.vys, self.coords.wheres, ComputedConstants.width,
                                                                      ComputedConstants.dt,ComputedConstants.ms,
                                                                      vStarBoundary)

        # moving Wall
        if self.wall is not None:
            fpl, fpr = numbaAccelerated.movingWallInteraction(self.coords.xs, self.coords.vxs, self.coords.vys, self.coords.wheres, self.wall.location(),
                                                              self.wall.velocity(),
                                                              ComputedConstants.dt, ComputedConstants.ms, self.wall.mass())
            self.wall.addToForce(fpl, fpr)

        fleft = -1
        fright = -1


        # left wall
        if self.leftCell is None and not ComputedConstants.periodic:
            fleft = numbaAccelerated.staticWallInteractionLeft(self.coords.xs, self.coords.vxs,self.coords.vys, self.coords.wheres, self.left, ComputedConstants.dt,
                                                               ComputedConstants.ms)

        # right wall
        if self.rightCell is None and not ComputedConstants.periodic:
            fright = numbaAccelerated.staticWallInteractionRight(self.coords.xs, self.coords.vxs,self.coords.vys, self.coords.wheres, self.right, ComputedConstants.dt,
                                                                 ComputedConstants.ms)

        self.computePressure(fup, fdown, fleft, fright)



    def interfaceCollide(self):
        return
        if self.leftCell != None:
            nb = numbaAccelerated.detectCollisionsAtInterface(self.coords.indicesLeftOfCell,self.leftCell.coords.indicesRightOfCell, *self.coords.tpl,*self.leftCell.coords.tpl,
                                             ComputedConstants.dt,
                                             ComputedConstants.ds)

    def collide(self):
        """
        Compute collisions between particles
        nbNeighbour is the number of neighbours par particle i which are checked
        :return: None
        """
        if Cell.coloringPolicy == "coll":
            self.coords.colors *= ComputedConstants.decoloringRatio


        x = 1e9
        xMax = -1
        if self.wall is not None:
            x = self.wall.location()
            #xMax = x - ComputedConstants.ls/4


        nb =numbaAccelerated.detectAllCollisions(*self.coords.tplExtended,
                                             ComputedConstants.dt,
                                             ComputedConstants.ds,
                                             self.histo, Cell.coloringPolicy, xMax,self.left,self.right)

    def improveCollisionDetectionSpeed(self):
        """
        Improve collision detectionSpeed by estimated the smallest number of neighbor to check for collision
        :return: None
        """
        try:
            ln = len(self.histo)
            h = np.array(self.histo, dtype=float)

            h /= (np.sum(h))
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

            DeltaMax = max(int(0.5 + k + np.log(0.1 / Sk) / np.log(xk)),12)
        except RuntimeWarning:
            DeltaMax = 100

        self.histo = np.zeros(DeltaMax, dtype=int)

    def sort(self):
        return self.coords.sort()

    ##############################################################
    #################      Helper functions     ##################
    ##############################################################

    def checkCorrectSide(self):
        numbaAccelerated.checkCorrectSide(self.coords.wheres, self.coords.xs, self.wall.location())

    def updateConstants(self):
        self.upToDate = False  # invalidate position buffer
        #self.computeTemperature()

    def getPositionsBuffer(self):
        if not self.upToDate:
            numbaAccelerated.twoArraysToOne(self.coords.xs, self.coords.ys, self.coords.wheres, self.positions)

        return self.positions

    def advect(self):
        numbaAccelerated.advect(self.coords.xs, self.coords.ys, self.coords.vxs, self.coords.vys, ComputedConstants.dt,ComputedConstants.forceX, ComputedConstants.ms)

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

        if Cell.collision:
            self.sort()
            self.collide()

        self.wallBounce()

        self.updateConstants()

        if (ComputedConstants.it + 250) % 500 == 0:

            if Cell.collision:
                self.improveCollisionDetectionSpeed()
            self.count()
