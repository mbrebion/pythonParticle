from Window import Window
from constants import ComputedConstants
from cell import Cell

ls = 1.5e-3 # mm
scale = 1
window = Window(int(5000/scale), 1e5, 300, 0.1, 0.05, ls*scale,resX=1600,resY=800,nbCells=2)
window.nStep = 2  # two time step by frame
ComputedConstants.dt/=scale

coords = window.domain.cells[0].coords
for i in range(0,len(coords.colors),3):
    coords.colors[i] = 1

Cell.colorCollisions = False


window.run()


