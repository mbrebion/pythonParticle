import time

from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 1
Y = 0.1
ls = 5e-3

nPart = 64000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
ComputedConstants.dt *= 1
nbDomains = 16
tHigh = 320.
tLow = 2 * T - tHigh
alpha = tLow / tHigh
rg = alpha / (1 + alpha)
rd = 1 - rg
print(rg, rd)

effectiveTemps = [tHigh for i in range(nbDomains)]
ratios = [rg / 5 for i in range(nbDomains)]
for j in range(nbDomains // 2, nbDomains):
    effectiveTemps[j] = tLow
    ratios[j] = rd / 5

domain = Domain(nbDomains, effectiveTemps=effectiveTemps, ratios=ratios)
domain.setMaxWorkers(3)

instants = []
temps = []

it = 0
itMax = 1e5
instants.append(ComputedConstants.time)
temps.append(domain.getAveragedTemperatures())
ComputedConstants.alphaAveraging = 0.1
while it < itMax:
    it += 1
    domain.update()

    if it % 500 == 0:
        print(100 * it / itMax, "%")
        print(domain.getAveragedTemperatures(), ComputedConstants.alphaAveraging)
    if it % 500 == 0:
        instants.append(ComputedConstants.time)
        temps.append(domain.getAveragedTemperatures())


printList(instants, "ts")
printList(temps, "temps")
