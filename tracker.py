import numbaAccelerated
import cell
from constants import *
import numpy as np


class Tracker:

    def __init__(self, domain, pid, meanFreePath=True):
        """

        :param domain: domain containing the cell
        :param pid: id of the particle to track
        """
        self.id = pid

        self.domain = domain
        self.currentIndex = None
        self.currentCell = None

        self.meanFreePath = meanFreePath  # if true, measurement of mean free path and time is performed

        self._updateCurrentIndex()
        x, y = self.getPosition()
        pos = np.array([x, y])
        self.previousPosAtImpacts = [pos]  # positions of particle id right after impact
        self._invalidateCurrentIndex()
        self.ds = []  # distances since last impact
        self.ts = []  # times of impacts
        self.lastPos = pos  # last position
        self.distance = 0  # distance since last impact

    def _invalidateCurrentIndex(self):
        self.currentIndex = None

    def _updateCurrentIndex(self):
        if self.currentIndex is None:
            index, idCell = -1, -1
            _cell = None
            while index == -1:
                idCell += 1
                _cell = self.domain.cells[idCell]
                index = numbaAccelerated.retieveIndex(self.id, _cell.coords.wheres)
            self.currentCell = _cell
            self.currentIndex = index

    def getPosition(self):
        return self.currentCell.getPositionsBuffer()[self.currentIndex]

    def _hasCollidedRecently(self):
        return self.currentCell.coords.colors[self.currentIndex] > ComputedConstants.decoloringRatio

    def _measureMeanFreePathAndTime(self):
        x, y = self.getPosition()
        pos = np.array([x, y])

        self.distance += numbaAccelerated.norm(pos - self.lastPos)
        self.lastPos = pos

        if self._hasCollidedRecently():
            self.previousPosAtImpacts.append(pos)
            self.ds.append(self.distance)
            self.ts.append(ComputedConstants.time)
            self.distance = 0

    def doMeasures(self):
        """
        perform all measurement required for particle id
        :return: None
        """
        self._updateCurrentIndex()

        if self.meanFreePath:
            self._measureMeanFreePathAndTime()

        self._invalidateCurrentIndex()

    def getMeanFreePathresults(self):
        """
        compute mean free path and time and their uncertainties
        :return: 4 requested values (l,u_l, tl, u_tl)
        """

        # mean free path and uncertainty
        l = np.average(np.array(self.ds))
        ul = np.std(np.array(self.ds)) / len(self.ds) ** 0.5

        # mean free time and uncertainty
        dts = np.array(self.ts)[1:] - np.array(self.ts)[0:len(self.ts) - 1]
        tl = np.average(np.array(dts))
        utl = np.std(np.array(dts) / len(dts) ** 0.5)

        return l, ul, tl, utl
