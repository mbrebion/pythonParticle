import time
import math
from glumpy import app, gl, gloo
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection

import domain
from domain import Domain
from constants import ComputedConstants, INITSIZEEXTRARATIO
import warnings
from shaders import *
from glumpy.graphics.collections import SegmentCollection

warnings.filterwarnings('ignore')


class Window:

    def __init__(self, nPart, P, T, L, H, ls, nbCells=1, ratios=None, effectiveTemps=None, resX=1024, resY=1024,periodic = False):
        # simulation
        X = L
        Y = H
        ls = ls
        ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

        self.domain = Domain(nbCells, effectiveTemps=effectiveTemps, ratios=ratios, periodic=periodic)

        # window
        ComputedConstants.resX = resX
        ComputedConstants.resY = resY
        self.window = app.Window(resX, resY, color=(1, 1, 1, 1))
        nTot = int(nPart * INITSIZEEXTRARATIO)
        self.program = gloo.Program(circlesVertex, circlesFragment, count=nTot)

        self.program['radius'] = self.getRadius()
        self.program['resolution'] = resX, resY
        self.program['spaceLength'] = X, Y

        self.updateProgram()

        self.t = 0
        self.nStep = 1
        self.displayPerformance = False

        self.window.on_resize = self.on_resize
        self.window.on_draw = self.on_draw

        # timing
        self.timeStep = 0
        self.duration = 3e-4

        self.createLabels()

        # wall
        transform = OrthographicProjection(Position())
        self.segments = SegmentCollection(mode="agg", linewidth='local', transform=transform)

    def run(self):
        if self.domain.wall is not None:
            p0, p1 = self.domain.wall.getBuffers()

            self.segments.append(p0, p1, linewidth=2)
            self.segments['antialias'] = 1
            self.window.attach(self.segments["transform"])
            self.window.attach(self.segments["viewport"])

        app.run()

    def updateProgram(self):
        start = 0

        if self.domain.wall is not None:
            self.segments.__delitem__(0)
            p0, p1 = self.domain.wall.getBuffers()
            self.segments.append(p0, p1, linewidth=2)

        for i in range(len(self.domain.cells)):
            cell = self.domain.cells[i]
            nb = cell.arraySize

            self.program['position'][start:start + nb] = cell.getPositionsBuffer()
            self.program['color'][start:start + nb] = cell.coords.colors
            start += nb

    def createLabels(self):
        self.labels = GlyphCollection(transform=OrthographicProjection(Position()))
        self.regular = FontManager.get("OpenSans-Regular.ttf")

        self.labels.append("_", self.regular, origin=(20, 30, 0), scale=0.5, anchor_x="left")
        self.labels.append("_", self.regular, origin=(20, 70, 0), scale=0.5, anchor_x="left")

        self.window.attach(self.labels["transform"])
        self.window.attach(self.labels["viewport"])

    def updateLabels(self):
        self.labels.__delitem__(0)
        self.labels.__delitem__(0)

        textRatio = " C. Ratio = " + str(int(self.duration / ComputedConstants.dt))
        textDuration = " C. time = " + "{:.2e}".format(self.duration * 1000) + " ms"

        self.labels.append(textRatio, self.regular, origin=(25, 30, 0), scale=0.8, anchor_x="left")
        self.labels.append(textDuration, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")

    def getRadius(self):
        return ComputedConstants.ds / ComputedConstants.length * ComputedConstants.resX / 2

    def on_resize(self, width, height):
        ComputedConstants.resX = width
        ComputedConstants.resY = height
        self.program["resolution"] = width, height

    def on_draw(self, dt):
        self.t += dt
        self.timeStep += 1
        self.window.clear()

        self.program.draw(gl.GL_POINTS)

        self.segments.draw()

        if self.displayPerformance:
            self.labels.draw()

        alpha = 0.05

        tInit = time.perf_counter()

        for i in range(self.nStep):
            self.domain.update()

        self.duration = (time.perf_counter() - tInit) / self.nStep * alpha + (1 - alpha) * self.duration

        self.updateProgram()

        if self.timeStep % 100 == 0 and self.displayPerformance:
            self.updateLabels()
            print( self.domain.computeTemperature() )


        #if self.timeStep % 50 == 0:
        #    print(ComputedConstants.time, self.domain.wall.location(), self.domain.countLeft(), self.domain.countRight(), self.domain.count())


def velocity(t):
        return -45


if __name__ == "__main__":
    window = Window(2000, 1e5, 300, 1, 1, 40e-3, nbCells=4)
    window.domain.setMaxWorkers(1)

    window.domain.addMovingWall(1000, 0.5, 40, imposedVelocity=velocity)
    ComputedConstants.dt /= 2
    window.nStep = 1
    window.displayPerformance = True
    # Cell.colorCollisions = False
    window.run()
