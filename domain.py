from cell import Cell
from constants import *
import tracker


class Domain:

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

            c = Cell(nbParts[i], effectiveTemps[i], left, right, startIndex,ComputedConstants.nbPartTarget//nbCells)
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

    def getAveragedTemperatures(self):
        out = []
        for cell in self.cells:
            out.append(cell.averagedTemperature)
        return out


    def addTracker(self, id):
        self.trackers.append(tracker.Tracker(self, id))

    def update(self):
        ComputedConstants.it += 1  # to be moved upper once cells are gathered in broader class
        ComputedConstants.time += ComputedConstants.dt

        for c in self.cells:
            c.update()

        for c in self.cells:
            c.prepareSwap()

        for c in self.cells:
            c.applySwap()

        for t in self.trackers:
            t.doMeasures()

