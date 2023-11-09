from Window import Window
from constants import ComputedConstants

ls = 2*4.8e-3  # mean free path in m
X = 0.177  # X length of simulation in m
Y = 0.1  # Y length in m
T = 300  # K
P = 1e5  # Pa

window = Window(2500, P, T, X, Y, ls, resX=1824, resY=1026, nbCells=1)
window.nStep = 2  # two time step by frame
ComputedConstants.dt = 3e-7  # fixed time step, in order to compare different parameter easily


# coloring left particles in red
coords = window.domain.cells[0].coords
for i in range(0,len(coords.colors),1):
    if coords.xs[i] < X/2:
        coords.colors[i] = 1
    else :
        coords.colors[i] = -1


window.run()


