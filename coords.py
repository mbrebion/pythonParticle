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
        self.colors = np.zeros(size, dtype=float)

    def sort(self):
        numbaAccelerated.sortCell(self.xs, self.ys, self.vxs, self.vys, self.wheres, self.colors)

    def addParticle(self, x, y, vx, vy, where, color):
        index = numbaAccelerated.goodIndexToInsertTo(y, self.ys, self.wheres, 0)
        self.xs[index] = x
        self.ys[index] = y
        self.vxs[index] = vx
        self.vys[index] = vy
        self.colors[index] = color
        self.wheres[index] = where

    def copyAndKillParticle(self, i, other):

        other.addParticle(self.xs[i], self.ys[i], self.vxs[i], self.vys[i], self.wheres[i], self.colors[i])

        self.wheres[i] = DEAD
        # new y ensure this particle won't move too much in list while sorting
        self.ys[i] = (i + 0.5) / len(self.xs) * ComputedConstants.width
        self.vxs[i] = 0.
        self.vys[i] = 0.
        self.colors[i] = 0.
        self.xs[i] = -1.

    def emptySelfInOther(self, other):
        self.sort()

        for i in range(len(self.xs)):
            if self.wheres[i] == DEAD:
                continue

            self.copyAndKillParticle(i, other)
