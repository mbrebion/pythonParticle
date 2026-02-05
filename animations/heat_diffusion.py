from animations.Window import Window
from constants import ComputedConstants
from cell import Cell


X = 0.2 # X length of simulation in m
Y = 0.1  # Y length in m
ls = X/25*1 # mean free path
T = 300  # K
P = 1e5  # Pa
nPart = 8000

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
ComputedConstants.dt *= 1
nbDomains = 8
tHigh = 320.
tLow = 2 * T - tHigh
alpha = tLow / tHigh
rg = alpha / (1 + alpha)
rd = 1 - rg
rg,rd = 2*rg, 2*rd

effectiveTemps = [tHigh for i in range(nbDomains)]
ratios = [rg / nbDomains for i in range(nbDomains)]
for j in range(nbDomains // 2, nbDomains):
    effectiveTemps[j] = tLow
    ratios[j] = rd / nbDomains


window = Window(nPart, P, T, X, Y, ls, nbCells=nbDomains, resX=1440, resY=720, effectiveTemps=effectiveTemps, ratios = ratios)
window.nStep = 1

window.run()

