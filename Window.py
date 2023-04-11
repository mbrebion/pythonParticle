import time
from glumpy import app, gl, gloo
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection
import matplotlib.pyplot as plt
from cell import Cell
import warnings
import numpy as np

warnings.filterwarnings('ignore')

vertex = """
#version 400
attribute vec2 position;
attribute float color;
uniform float radius;
uniform vec2 resolution;
uniform vec2 spaceLength;

varying vec2 v_center;
varying float v_radius;
varying vec3 v_color;

void main () {
    v_center = position / spaceLength * resolution;
    v_radius = radius;
    v_color = vec3(0.,0.,0.);
    if (color <0.05){
        v_color = vec3(0.,0.,0.);
    }else{
        v_color = vec3(color,0.,0.);
    }
     
    
    gl_Position = vec4(2.0*position/spaceLength-1.0, 0.0, 1.0); 
    gl_PointSize = 3.0 + ceil(2.0*radius);    
    }
"""

fragment = """
#version 400
varying vec2 v_center;
varying float v_radius;
varying vec3 v_color;

void main() {
    vec2 p = gl_FragCoord.xy - v_center;
    float a = 1.0;
    float d = length(p)-v_radius*0.99;
    if(d > 0.0) a = exp(-d*d);
    gl_FragColor = vec4(v_color, a);    
}
"""


class Window:

    def __init__(self, nPart, P, T):
        # simulation
        X = 0.1
        Y = 0.1
        ls = 6e-3
        Cell.thermodynamicSetup(T, X, Y, P, nPart,ls)
        self.cell = Cell(1, T)
        Cell.dt /= 2

        # window
        self.resX = 1024
        self.resY = 1024
        self.window = app.Window(self.resX, self.resY, color=(1, 1, 1, 1))
        nTot = int(nPart * 1.2)
        self.program = gloo.Program(vertex, fragment, count=nTot)  # 1.2 for dead cells

        self.program['position'] = self.cell.getPositionsBuffer()
        self.program['radius'] = self.getRadius() / 2
        self.program['resolution'] = self.resX, self.resY
        self.program['color'] = self.cell.colors
        self.program['spaceLength'] = X, Y
        self.t = 0

        self.window.on_resize = self.on_resize
        self.window.on_draw = self.on_draw

        # timing
        self.timeStep = 0
        self.duration = 3e-4

        self.createLabels()

        # add trackers
        self.cell.addTracker(3*nPart//4)
        self.cell.addTracker(nPart//2)
        self.cell.addTracker(nPart//4)

        app.run()

        for t in self.cell.trackers:
            l, ul, tl, utl = t.getMeanFreePathresults()
            print("Particle ", t.id, " : ")
            print("l = ", "{:.2e}".format(l), " +/- ", "{:.2e}".format(ul), " m")
            print("tl = ", "{:.2e}".format(tl), " +/- ", "{:.2e}".format(utl), " s")
            print()

    def createLabels(self):
        self.labels = GlyphCollection(transform=OrthographicProjection(Position()))
        self.regular = FontManager.get("OpenSans-Regular.ttf")

        # self.labels.append("_", self.regular, origin=(20, 20, 0), scale=0.5, anchor_x="left")
        self.labels.append("_", self.regular, origin=(20, 30, 0), scale=0.5, anchor_x="left")
        self.labels.append("_", self.regular, origin=(20, 70, 0), scale=0.5, anchor_x="left")

        self.window.attach(self.labels["transform"])
        self.window.attach(self.labels["viewport"])

    def updateLabels(self):
        self.labels.__delitem__(0)
        self.labels.__delitem__(0)
        # self.labels.__delitem__(0)

        # textT = " T = " + str(self.cell.temperature)[0:5] + " K"
        # textP = " P = " + str(self.cell.averagedPressure)[0:5] + " Pa"
        textRatio = " C. Ratio = " + str(int(self.duration / Cell.dt))
        textDuration = " C. time = " + "{:.2e}".format(self.duration * 1000) + " ms"

        # self.labels.append(textT, self.regular, origin=(25, 110, 0), scale=0.8, anchor_x="left")
        # self.labels.append(textP, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")
        self.labels.append(textRatio, self.regular, origin=(25, 30, 0), scale=0.8, anchor_x="left")
        self.labels.append(textDuration, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")

    def getRadius(self):
        return self.cell.ds / self.cell.length * self.resX

    def on_resize(self, width, height):
        self.resX = width
        self.resY = height
        self.program["resolution"] = width, height

    def on_draw(self, dt):
        self.t += dt
        self.timeStep += 1
        self.window.clear()

        self.program.draw(gl.GL_POINTS)
        self.labels.draw()

        alpha = 0.05
        nStep = 1
        tInit = time.perf_counter()

        for i in range(nStep):
            self.cell.update()

        self.duration = (time.perf_counter() - tInit) / nStep * alpha + (1 - alpha) * self.duration

        self.program['position'] = self.cell.getPositionsBuffer()
        self.program['color'] = self.cell.colors

        if self.timeStep % 30 == 0:
            self.updateLabels()
            pass


window = Window(120, 1e5, 300)

# todolist :
#  - compute thermodynamic values such as l (mean free path)
#  - add multiple cells
#  - add moving wall
#  - checking integration with imgui for nicer outputs
