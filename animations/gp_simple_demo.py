from Window import Window
from constants import ComputedConstants

window = Window(4000,1E5,600,0.1,0.1, 0.1/25,nbCells=1,periodic=True)

ComputedConstants.decoloringRatio = 0.7  # speed at which colors return to black after collision (0: instant, 0.5, after two/three frames, 1 never)
window.nStep = 2 # two time step by frame
window.displayPerformance = True
ComputedConstants.dt /= 5
ComputedConstants.forceX = 9.81 * ComputedConstants.ms*10000*0
ComputedConstants.boundaryTemperature = 600
window.run()

