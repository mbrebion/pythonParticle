from animations.Window import Window
from constants import ComputedConstants
from cell import Cell

X = 0.177  # X length of simulation in m
Y = 0.1  # Y length in m
ls = Y/25  # mean free path
T = 300  # K
P = 1e5  # Pa

window = Window(12000, P, T, X, Y, ls, nbCells=12, periodic=True, resX=1824, resY=1026)
ComputedConstants.forceX = 8  # N
ComputedConstants.boundaryTemperature = T

Cell.coloringPolicy = "vx"  # color particles according to vx, blue if <0, red if >0, black if close to 0

window.run()

