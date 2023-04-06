from glumpy import app, gl, gloo
import numpy as np

vertex = """
#version 120

uniform vec2 mouse;
uniform float zoom;
attribute vec2 position;
varying float v_radius;
void main () {
    v_radius = 4.0;
    gl_Position = vec4(position, 0.0, 1.0);
    gl_PointSize = 2.0*v_radius;
}

"""

fragment = """
#version 120
const float SQRT_2 = 1.4142135623730951;
uniform vec4 (0.5,0.5,0.5,1);
void main()
{
    gl_FragColor = color * (SQRT_2/2.0 - length(gl_PointCoord.xy - 0.5));
}
"""

n = 5000
window = app.Window(1024,1024, color=(1,1,1,1))
program = gloo.Program(vertex, fragment, count=n)
program['position'] = np.random.normal(0.0,0.25,(n,2))
program['zoom'] = 2.0

@window.event
def on_mouse_motion(x, y, dx, dy):
    program['mouse'] = (2.0*float(x)/window.width-1.0,
                        1.0-2.0*float(y)/window.height)

@window.event
def on_mouse_scroll(x, y, dx, dy):
    zoom = program['zoom']
    program['zoom'] = min(max(zoom *(1.0+ dy/100.0), 1.0), 50.0)

@window.event
def on_draw(dt):
    window.clear()
    #program['position'] = np.random.normal(0.0,0.25,(n,2))
    program.draw(gl.GL_POINTS)

app.run()
