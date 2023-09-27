from Window import Window
from constants import ComputedConstants

window = Window(4000,1E5,300,0.1,0.1, 4.95416e-03,nbCells=2)

ComputedConstants.decoloringRatio = 0.7  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
window.nStep = 1  # two time step by frame
ComputedConstants.dt/=1
window.run()

