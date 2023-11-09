from Window import Window
from constants import ComputedConstants
from cell import Cell


Y = 12000  # Y length in m
X = Y*1.77  # X length of simulation in m
ls = X/35  # mean free path
T = 300  # K
P = 1e5  # Pa

window = Window(6000, P, T, X, Y, ls, resX=1824, resY=1026)


ComputedConstants.decoloringRatio = 0.85  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
window.nStep = 1  # two time step by frame


# add vertical force to particles
def newAdvect(self):
    self.coords.xs += self.coords.vxs * ComputedConstants.dt
    self.coords.ys += self.coords.vys * ComputedConstants.dt
    self.coords.vxs += -9.81 * ComputedConstants.dt


Cell.advect = newAdvect
Cell.coloringPolicy = "coll"

for i in range(4000):
    window.domain.update()

window.run()


