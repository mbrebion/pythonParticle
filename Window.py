import time
from glumpy import app, gl, gloo
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection
from domain import Domain
from constants import ComputedConstants, INITSIZEEXTRARATIO
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
        X = 1
        Y = 0.1
        ls = 5e-3
        ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

        # self.domain = Domain(4, effectiveTemp=[250, 250, 350, 350])
        tg = 330
        td = 270
        alpha = td/tg
        self.domain = Domain(2,effectiveTemps=[tg,td],ratios=[1-1/(1+alpha),1/(1+alpha)])
        ComputedConstants.dt /= 1.


        # window
        self.resX = 1024
        self.resY = 1024
        self.window = app.Window(self.resX, self.resY, color=(1, 1, 1, 1))
        nTot = int(nPart * INITSIZEEXTRARATIO)
        self.program = gloo.Program(vertex, fragment, count=nTot)  # 1.2 for dead cells

        self.program['radius'] = self.getRadius()
        self.program['resolution'] = self.resX, self.resY
        self.program['spaceLength'] = X, Y

        self.updateProgram()

        self.t = 0

        self.window.on_resize = self.on_resize
        self.window.on_draw = self.on_draw

        # timing
        self.timeStep = 0
        self.duration = 3e-4

        self.createLabels()

        nTracker = 0
        for i in range(nTracker):
            self.domain.addTracker(int((i + 0.5) * nPart / nTracker))



        app.run()

        ls = []
        ts = []

        for t in self.domain.trackers:
            l, ul, tl, utl = t.getMeanFreePathresults()
            ls.append(l)
            ts.append(tl)
            print("Particle ", t.id, " : ")
            print("l = ", "{:.2e}".format(l), " +/- ", "{:.2e}".format(ul), " m")
            print("tl = ", "{:.2e}".format(tl), " +/- ", "{:.2e}".format(utl), " s")
            print()

        if nTracker >= 0:
            ls = np.array(ls)
            ts = np.array(ts)

            print()
            print("l = ", "{:.2e}".format(np.average(ls)), " +/- ", "{:.2e}".format(np.std(ls) / nTracker ** 0.5), " m")
            print("t = ", "{:.2e}".format(np.average(ts)), " +/- ", "{:.2e}".format(np.std(ts) / nTracker ** 0.5), " s")



    def updateProgram(self):
        start = 0
        for i in range(len(self.domain.cells)):
            cell = self.domain.cells[i]
            nb = cell.arraySize

            self.program['position'][start:start + nb] = cell.getPositionsBuffer()
            self.program['color'][start:start + nb] = cell.coords.colors
            start += nb

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
        textRatio = " C. Ratio = " + str(int(self.duration / ComputedConstants.dt))
        textDuration = " C. time = " + "{:.2e}".format(self.duration * 1000) + " ms"

        # self.labels.append(textT, self.regular, origin=(25, 110, 0), scale=0.8, anchor_x="left")
        # self.labels.append(textP, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")
        self.labels.append(textRatio, self.regular, origin=(25, 30, 0), scale=0.8, anchor_x="left")
        self.labels.append(textDuration, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")

    def getRadius(self):
        return ComputedConstants.ds / ComputedConstants.length * self.resX / 2

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
            self.domain.update()

        self.duration = (time.perf_counter() - tInit) / nStep * alpha + (1 - alpha) * self.duration

        self.updateProgram()

        if self.timeStep % 50 == 0:
            print(self.domain.getAveragedTemperatures(),sum(self.domain.getAveragedTemperatures()))
            self.updateLabels()
            pass


window = Window(20000, 1e5, 300)

# todolist :
#  - add moving wall
#  - checking integration with imgui for nicer outputs
