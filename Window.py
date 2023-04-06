import time

from glumpy import app, gl, gloo
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection, Viewport
from cell import Cell
import warnings

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
    if (color <0.1){
        v_color = vec3(0.,0.,0.);
    }else{
        v_color = vec3(1,0.1,0.1);
    }
     
    
    gl_Position = vec4(2.0*position/spaceLength-1.0, 0.0, 1.0); 
    gl_PointSize = 2.0 + ceil(2.0*radius);    
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
    float d = length(p)-v_radius*0.95;
    if(d > 0.0) a = exp(-d*d);
    gl_FragColor = vec4(v_color, a);    
}
"""


class Window:

    def __init__(self, nPart, P, T):
        # simulation
        X = 0.1
        Y = 0.1
        Cell.thermodynamicSetup(T, X, Y, P, nPart)
        self.cell = Cell(1, T)
        Cell.dt *= 0.5

        # window
        self.resX = 1024
        self.resY = 1024
        self.window = app.Window(self.resX, self.resY, color=(1, 1, 1, 1))
        nTot = int(nPart * 1.2)
        self.program = gloo.Program(vertex, fragment, count=nTot)  # 1.2 for dead cells

        self.program['position'] = self.cell.ouputBuffer()
        self.program['radius'] = self.getRadius() / 2
        self.program['resolution'] = self.resX, self.resY
        self.program['color'] = self.cell.colors
        self.program['spaceLength'] = X, Y
        self.t = 0

        self.window.on_resize = self.on_resize
        self.window.on_draw = self.on_draw

        # timing
        self.timeStep = 0
        self.duration = 1e-4

        self.createLabels()

        app.run()

    def createLabels(self):
        self.labels = GlyphCollection(transform=OrthographicProjection(Position()))
        self.regular = FontManager.get("OpenSans-Regular.ttf")

        self.labels.append("_", self.regular, origin=(20, 20, 0), scale=0.5, anchor_x="left")
        self.labels.append("_", self.regular, origin=(20, 40, 0), scale=0.5, anchor_x="left")
        self.labels.append("_", self.regular, origin=(20, 40, 0), scale=0.5, anchor_x="left")

        self.window.attach(self.labels["transform"])
        self.window.attach(self.labels["viewport"])

    def updateLabels(self):
        self.labels.__delitem__(0)
        self.labels.__delitem__(0)
        self.labels.__delitem__(0)

        textT = " T = " + str(self.cell.temperature)[0:5] + " K"
        textP = " P = " + str(self.cell.averagedPressure)[0:5] + " Pa"
        textDuration = " C. Time = " + "{:.2e}".format(self.duration*1000) + " ms"

        self.labels.append(textT, self.regular, origin=(25, 110, 0), scale=0.8, anchor_x="left")
        self.labels.append(textP, self.regular, origin=(25, 70, 0), scale=0.8, anchor_x="left")
        self.labels.append(textDuration, self.regular, origin=(25, 30, 0), scale=0.8, anchor_x="left")

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
        self.program['position'] = self.cell.ouputBuffer()
        self.program['color'] = self.cell.colors

        alpha = 0.01
        nStep = 2
        tInit = time.perf_counter()

        for i in range(nStep):
            self.cell.update()

        self.duration = (time.perf_counter() - tInit) / nStep * alpha + (1 - alpha) * self.duration

        self.program.draw(gl.GL_POINTS)

        if self.timeStep % 10 == 0:
            self.updateLabels()
        self.labels.draw()


window = Window(1000, 1e5, 300)

# TODO : move xs, ys into single arrays ; same for vxs,vys
# TODO implement collision detection bases on radius
# TODO test results
