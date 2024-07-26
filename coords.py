import numpy as np
from constants import *
import numbaAccelerated


class Coords:

    def __init__(self, size):
        self.xs = np.empty(size, dtype=float)
        self.ys = np.empty(size, dtype=float)
        self.vxs = np.empty(size, dtype=float)
        self.vys = np.empty(size, dtype=float)
        self.wheres = np.ones(size, dtype=np.int32) * DEAD
        self.lastColls = np.ones(size, dtype=float) * -1
        self.indicesLeftOfCell = np.zeros(size // 4, dtype=np.int32)-1
        self.indicesRightOfCell = np.zeros(size // 4, dtype=np.int32)-1

        self.updateTuple()

    def updateTuple(self):
        self.tpl = self.xs, self.ys, self.vxs, self.vys, self.wheres, self.lastColls
        self.tplExtended = self.xs, self.ys, self.vxs, self.vys, self.wheres, self.lastColls, self.indicesLeftOfCell, self.indicesRightOfCell



    def sort(self):
        numbaAccelerated.sortCell(self.xs, self.ys, self.vxs, self.vys, self.wheres, self.lastColls)

    def resetSwap(self):
        self.wheres[:] = DEAD
        self.alive = 0
