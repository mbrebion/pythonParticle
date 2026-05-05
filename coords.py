import numpy as np

from constants import *
from numbaAcc import sorting


class Coords:

    def __init__(self, size):
        #self.xs = np.empty(size, dtype=float)
        #self.ys = np.empty(size, dtype=float)
        #self.vxs = np.empty(size, dtype=float)
        #self.vys = np.empty(size, dtype=float)
        #self.wheres = np.ones(size, dtype=np.int32) * DEAD
        #self.lastColls = np.ones(size, dtype=float) * -1
        #self.colors = np.ones(size, dtype=float)  # new !
        #self.indicesLeftOfCell = np.zeros(size // 4, dtype=np.int32)-1
        #self.indicesRightOfCell = np.zeros(size // 4, dtype=np.int32)-1
        self.alive = 0

        self.updateTuple()


        self.tp = np.dtype([("xs",np.float64),("ys",np.float64),("vxs",np.float64),("vys",np.float64),("wheres",np.int32),("lastColls",np.float64), ("colors",np.float64),("indLeft",np.int32),("indRight",np.int32)])
        self.crd = np.zeros(size,dtype=self.tp)
        self.crd["wheres"][:] = DEAD
        self.crd["colors"][:] = 1
        self.crd["indLeft"][:] = -1
        self.crd["indRight"][:] = -1


    def updateTuple(self):
        self.tpl = self.xs, self.ys, self.vxs, self.vys, self.wheres, self.lastColls,self.colors
        self.tplExtended = self.xs, self.ys, self.vxs, self.vys, self.wheres, self.lastColls, self.colors, self.indicesLeftOfCell, self.indicesRightOfCell



    def sort(self):
        sorting.sortCell(self.crd)

    def resetSwap(self):
        self.crd["wheres"][:] = DEAD
        self.alive = 0
