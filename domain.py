from cell import Cell
from constants import *
import tracker
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from movingWall import MovingWall


class Domain:

    ##############################################################
    ###################        Initializing        ###############
    ##############################################################

    def __init__(self, nbCells, *, ratios=None, effectiveTemps=None):
        ComputedConstants.nbCells = nbCells

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

            c = Cell(nbParts[i], effectiveTemps[i], left, right, startIndex, ComputedConstants.nbPartTarget // nbCells)
            startIndex += nbParts[i]
            self.cells.append(c)

        ComputedConstants.nbPartCreated = startIndex

        # updating neighbors
        for i in range(1, ComputedConstants.nbCells):
            self.cells[i - 1].rightCell = self.cells[i]

        for i in range(0, ComputedConstants.nbCells - 1):
            self.cells[i + 1].leftCell = self.cells[i]

        # trackers
        self.trackers = []

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

    ##############################################################
    ################## Compute thermodynamic      ################
    ##############################################################

    def computePressure(self):
        p = 0
        for c in self.cells:
            p += c.instantPressure / len(self.cells)  # average of pression over multiple cells

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

    ##############################################################
    ##################         Update      #######################
    ##############################################################

    def advectMovingWall(self):
        if self.wall is not None:
            self.wall.advect()

    def _updateMultipleT(self):
        ComputedConstants.it += 1  # to be moved upper once cells are gathered in broader class
        ComputedConstants.time += ComputedConstants.dt

        futures = []
        for c in self.cells:
            futures.append(self.pool.submit(c.update))
        wait(futures, return_when=ALL_COMPLETED)
        futures = []

        for c in self.cells:
            futures.append(self.pool.submit(c.prepareSwap))
        wait(futures, return_when=ALL_COMPLETED)

        for c in self.cells:
            c.applySwap()

        self.advectMovingWall()

        for t in self.trackers:
            t.doMeasures()

    def _updateSingleT(self):
        ComputedConstants.it += 1  # to be moved upper once cells are gathered in broader class
        ComputedConstants.time += ComputedConstants.dt

        for c in self.cells:
            c.update()

        for c in self.cells:
            c.prepareSwap()

        for c in self.cells:
            c.applySwap()

        self.advectMovingWall()

        for t in self.trackers:
            t.doMeasures()
