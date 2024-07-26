from cell import Cell
from constants import *
import tracker
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from movingWall import MovingWall


class Domain:

    ##############################################################
    ###################        Initializing        ###############
    ##############################################################

    def __init__(self, nbCells, *, ratios=None, effectiveTemps=None, periodic=False, v_xYVelocityProfile=None):
        """

        :param nbCells: number of cells used in the simulation
        :param ratios: if provided, allows to specify different ratios of particles in each cells
        :param effectiveTemps: if provided, allows to specify different temperatures in each cells
        :param periodic: if True, periodic boundary conditions are used to link both left and right sides of domain.
        :param v_xYVelocityProfile: v_x(y) mean velocity profile to be imposed at startup
        """
        ComputedConstants.nbCells = nbCells
        ComputedConstants.periodic = periodic

        if ratios is None:
            ratios = [1. / ComputedConstants.nbCells for _ in range(nbCells)]

        if effectiveTemps is None:
            effectiveTemps = [ComputedConstants.initTemp for _ in range(nbCells)]

        nbParts = [int(ComputedConstants.nbPartTarget * ratios[i]) for i in range(nbCells)]

        # cells creation
        self.cells = []
        startIndex = 0
        for i in range(ComputedConstants.nbCells):
            left = i * ComputedConstants.length / nbCells
            right = (i + 1) * ComputedConstants.length / nbCells

            c = Cell(nbParts[i], effectiveTemps[i], left, right, startIndex, ComputedConstants.nbPartTarget // nbCells,
                     v_xYVelocityProfile=v_xYVelocityProfile)
            startIndex += nbParts[i]
            self.cells.append(c)

        ComputedConstants.nbPartCreated = startIndex

        # updating neighbors
        for i in range(1, ComputedConstants.nbCells):
            self.cells[i - 1].rightCell = self.cells[i]

        for i in range(0, ComputedConstants.nbCells - 1):
            self.cells[i + 1].leftCell = self.cells[i]

        if periodic and ComputedConstants.nbCells > 1:
            self.cells[-1].rightCell = self.cells[0]
            self.cells[0].leftCell = self.cells[-1]

        # trackers
        self.trackers = []

        # nbcollisions
        self.insideAverage = 0
        self.interfaceAverage = 0
        self.itCount = 0

        # single/multi thread default
        self.update = self._updateSingleT
        self.pool = None

        # wall
        self.wall = None

    def setMaxWorkers(self, mw):
        if mw > 1:
            self.pool = ThreadPoolExecutor(max_workers=mw)
            self.update = self._updateMultipleT
        else:
            self.update = self._updateSingleT

    def addMovingWall(self, m, x, v, imposedVelocity=None):
        self.wall = MovingWall(m, x, v, imposedVelocity)
        for c in self.cells:
            c.updateIndicesAccordingToWall(x)
            c.wall = self.wall

    def addTracker(self, id):
        self.trackers.append(tracker.Tracker(self, id))


    def reshapeCells(self):
        """
        This function moves cells boundaries to equilibrate the number of particles (usefull with moving wall)
        :return: None
        """

        L = self.cells[-1].right
        def move():

            ns = []
            nsCum = [0]
            nTot = 0

            delta = (self.cells[0].right - self.cells[0].left)/100
            k =len(self.cells)
            for c in self.cells:
                n = c.count()
                ns.append(n)
                nsCum.append(n + nsCum[-1])
                nTot += n
            nAvg = nTot/k
            nsCum.remove(0)

            xs = []
            ds = []
            for i in range(len(self.cells)):
                xs.append(self.cells[i].left)
                ds.append(self.cells[i].right - self.cells[i].left)
            xs.append(L)

            xps = [0]
            for i in range(len(self.cells)):
                alpha = 3*(1. - ns[i] /nAvg)
                xps.append( xps[-1] + ds[i] + delta * alpha)

            for i in range(len(self.cells)):
                self.cells[i].left = xps[i]
                self.cells[i].right = xps[i+1]
            return ns,xs

        ns=[]
        for _ in range(2):
            ns,xs=move()



    ##############################################################
    ################## Compute thermodynamic      ################
    ##############################################################

    def computeAverageVelocityLeftOfWall(self):
        av = 0
        for c in self.cells:
            av += c.computeSumVelocityLeftOfWall(self.wall.location())
        return av / self.countLeft()

    def computeXVelocityBins(self, nbBins):
        bin = np.array([0.] * nbBins)
        for c in self.cells:
            bin += c.computeXVelocityBins(nbBins)
        return bin / len(self.cells)

    def ComputeXVelocityBeforeWall(self, nbBins, span):

        sBins = np.array([0.] * nbBins)
        sCounts = np.array([0.] * nbBins)
        for c in self.cells:
            bins, counts = c.ComputeXVelocityBeforeWall(nbBins, span, self.wall.location())
            sBins += bins
            sCounts += counts

        for i in range(len(sCounts)):
            if sCounts[i] == 0:
                sCounts[i] = 1

        return sBins / sCounts

    def computeXVelocity(self):
        vx = 0.
        for c in self.cells:
            vx += c.computeXVelocity()
        return vx / len(self.cells)

    def computePressure(self):
        p = 0
        for c in self.cells:
            p += c.instantPressure / len(self.cells)  # average of pressure over multiple cells

        return p

    def count(self):
        count = 0
        for c in self.cells:
            count += c.count()
        return count

    def countLeft(self):
        count = 0.
        x = self.wall.location()
        for c in self.cells:
            count += c.countLeft(x)
        return count

    def countRight(self):
        count = 0.
        x = self.wall.location()
        for c in self.cells:
            count += c.countRight(x)
        return count

    def computeKineticEnergyLeftSide(self):
        ec = 0.
        for c in self.cells:
            ecl, ecr = c.computeKineticEnergy()
            ec += ecl
        return ec

    def computeKineticEnergyRightSide(self):
        ec = 0.
        for c in self.cells:
            ecl, ecr = c.computeKineticEnergy()
            ec += ecr
        return ec

    def computeKineticEnergy(self):
        ec = 0.
        for c in self.cells:
            ecl, ecr = c.computeKineticEnergy()
            ec += ecl + ecr
        return ec

    def getAveragedTemperatures(self):
        out = []
        for cell in self.cells:
            out.append(cell.averagedTemperature)
        return out

    def computeTemperature(self):
        return self.computeKineticEnergy() / ComputedConstants.nbPartTarget / ComputedConstants.kbs

    def checkSides(self):
        for c in self.cells:
            c.checkCorrectSide()

    def resetCount(self):
        self.itCount = 0

    def countCollisions(self):
        a = 1 / (1 + self.itCount)
        self.itCount += 1
        inside = 0
        interface = 0
        for c in self.cells:
            inside += c.lastNbCollide
            interface += c.lastNbCollideInterface

        self.insideAverage = self.insideAverage * (1 - a) + a * inside
        self.interfaceAverage = self.interfaceAverage * (1 - a) + a * interface

    ##############################################################
    ##################         Update      #######################
    ##############################################################

    def advectMovingWall(self):
        if self.wall is not None:
            self.wall.advect()

    def _updateMultipleT(self):
        ComputedConstants.it += 1  # to be moved upper once cells are gathered in broader class
        ComputedConstants.time += ComputedConstants.dt

        self.advectMovingWall()

        futures = []
        for c in self.cells:
            futures.append(self.pool.submit(c.update))
        wait(futures, return_when=ALL_COMPLETED)

        futures = []
        for c in self.cells:
            futures.append(self.pool.submit(c.interfaceCollide))
        wait(futures, return_when=ALL_COMPLETED)

        futures = []
        for c in self.cells:
            futures.append(self.pool.submit(c.prepareSwap))
        wait(futures, return_when=ALL_COMPLETED)

        for c in self.cells:
            c.applySwap()

        if ComputedConstants.periodic:
            for c in self.cells:
                c.applyPeriodic()

        for t in self.trackers:
            t.doMeasures()

        self.countCollisions()
        if ComputedConstants.it % 10 == 0:
            self.reshapeCells()


    def _updateSingleT(self):
        ComputedConstants.it += 1  # to be moved upper once cells are gathered in broader class
        ComputedConstants.time += ComputedConstants.dt

        self.advectMovingWall()

        for c in self.cells:
            c.update()

        for c in self.cells:
            c.interfaceCollide()

        for c in self.cells:
            c.prepareSwap()

        for c in self.cells:
            c.applySwap()

        if ComputedConstants.periodic:
            for c in self.cells:
                c.applyPeriodic()

        for t in self.trackers:
            t.doMeasures()

        self.countCollisions()

        if ComputedConstants.it % 10 == 0:
            self.reshapeCells()
