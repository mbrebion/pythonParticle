from Window import Window
from constants import ComputedConstants
from cell import Cell

window = Window(6000, 1e5, 300, 4000, 12000, 200,resX=300,resY=900)


ComputedConstants.decoloringRatio = 0.85  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
window.nStep = 2  # two time step by frame


def newAdvect(self):
    self.coords.xs += self.coords.vxs * ComputedConstants.dt
    self.coords.ys += self.coords.vys * ComputedConstants.dt
    self.coords.vys += -9.81 * ComputedConstants.dt


Cell.advect = newAdvect

window.run()


