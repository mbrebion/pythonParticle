from Window import Window
from constants import ComputedConstants

window = Window(1000, 1e5, 300, 1e-6, 1e-6, 70e-9)

ComputedConstants.dt /= 2  # reduce time step
ComputedConstants.decoloringRatio = 0.8  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
window.nStep = 1  # two time step by frame

window.run()

