from animations.Window import Window
from constants import ComputedConstants
from cell import Cell


X = 0.1 # X length of simulation in m
Y = 0.1  # Y length in m
ls = X/50*1 # mean free path
T = 300  # K
P = 1e5  # Pa

window = Window(5000, P, T, X, Y, ls, nbCells=1, resX=1024, resY=1024)
window.nStep = 1

window.run()

