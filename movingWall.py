from constants import ComputedConstants
import numpy as np


class MovingWall:

    def __init__(self, m, x, v, imposedVelocity=None):
        """

        :param m: mass of the wall
        :param x: initial location
        :param v: initial velocity
        :param imposedVelocity:  function such as v = imposedVelocity(t), if not provided, velocity is computed according to forces
        """
        self._mass = m
        self._x = x
        self._v = v
        self._forceLeft = 0.
        self._forceRight = 0.
        self.imposedVelocity = imposedVelocity

    def setFree(self):
        self.savedImposedVelocity = self.imposedVelocity
        self.imposedVelocity = None

    def unSetFree(self):
        self.imposedVelocity = self.savedImposedVelocity


    def advect(self):
        # update velocity (enforced or computed)
        if self.imposedVelocity is not None:
            t = ComputedConstants.time
            self._v = self.imposedVelocity(t,self._x)
        else:
            self._v += (self._forceLeft + self._forceRight) * ComputedConstants.dt / self._mass
        self._forceLeft = 0
        self._forceRight = 0

        # update location
        self._x += self._v * ComputedConstants.dt

    def addToForce(self, fLeft, fRight):
        """
        add f to force which will be applied on next iteration
        force is reset to 0 after each iteration
        This patterns allows to compute force applied to the wall from several cells
        :param fLeft: force acting to left side of wall
        :param fRight: force acting to right side of wall
        :return: None
        """
        self._forceLeft += fLeft
        self._forceRight += fRight

    def kineticEnergy(self):
        return self._mass * self._v ** 2 * 0.5

    def location(self):
        return self._x

    def velocity(self):
        return self._v

    def mass(self):
        return self._mass

    def getBuffers(self):
        b1 = np.array([self._x * ComputedConstants.resX / ComputedConstants.length, 0., 0.], dtype=float)
        b2 = np.array([self._x * ComputedConstants.resX / ComputedConstants.length, ComputedConstants.resY, 0.], dtype=float)
        return b1, b2
