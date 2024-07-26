import numbaAccelerated
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

        self.currentIndex = 0
        self.currentCell = None
        self.currentCellIndex = 0
        self.lastCollisionTime = -1


        self.meanFreePath = meanFreePath  # if true, measurement of mean free path and time is performed

        self._updateCurrentIndex()
        self.lastNormV = (np.sum(self.getVelocity() ** 2)) ** 0.5
        self.ds = []  # distances since last impact
        self.ts = []  # times of impacts

    def _updateCurrentIndex(self):
        index, idCell = -1, self.currentCellIndex - 1
        _cell = None
        while index == -1:
            idCell += 1
            _cell = self.domain.cells[idCell % len(self.domain.cells)]
            index = numbaAccelerated.retrieveIndexs(self.id, _cell.coords.wheres, self.currentIndex)

        self.currentCell = _cell
        self.currentCellIndex = idCell % len(self.domain.cells)
        self.currentIndex = index

    def getVelocity(self):
        return np.array(
            [self.currentCell.coords.vxs[self.currentIndex], self.currentCell.coords.vys[self.currentIndex]])

    def _measureMeanFreePathAndTime(self):
        accurateTime =  self.currentCell.coords.lastColls[self.currentIndex]
        newV = (np.sum(self.getVelocity() ** 2)) ** 0.5
        if newV != self.lastNormV:
            if self.lastCollisionTime < 0:
                self.lastCollisionTime = accurateTime
                self.lastNormV = (np.sum(self.getVelocity() ** 2)) ** 0.5
            else:
                deltaT = (accurateTime - self.lastCollisionTime)
                #print(ComputedConstants.time, self.currentCell.coords.lastColls[self.currentIndex])

                self.distance = deltaT * self.lastNormV
                self.ds.append(self.distance)
                self.ts.append(deltaT)

                self.lastCollisionTime = accurateTime
                self.lastNormV = newV

    def doMeasures(self):
        """
        perform all measurement required for particle id
        :return: None
        """
        if self.meanFreePath:
            self._updateCurrentIndex()
            self._measureMeanFreePathAndTime()

    def getMeanFreePathresults(self):
        """
        compute mean free path and time and their uncertainties
        :return: 4 requested values (l,u_l, tl, u_tl)
        """

        # mean free path and uncertainty
        l = np.average(np.array(self.ds))
        ul = np.std(np.array(self.ds)) / len(self.ds) ** 0.5

        # mean free time and uncertainty

        tl = np.average(np.array(self.ts))
        utl = np.std(np.array(self.ts) / len(self.ts) ** 0.5)

        return l, ul, tl, utl
