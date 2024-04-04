from Window import Window
from constants import ComputedConstants
from cell import Cell


X = 0.177  # X length of simulation in m
Y = 0.1  # Y length in m
ls = X/35*4  # mean free path
T = 300  # K
P = 1e5  # Pa

window = Window(8000, P, T, X, Y, ls, nbCells=1, resX=1824, resY=1026)
window.nStep = 3
ComputedConstants.decoloringRatio = 0.7  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
Cell.coloringPolicy = "coll"  # color particles according to vx, blue if <0, red if >0, black if close to 0

window.run()

