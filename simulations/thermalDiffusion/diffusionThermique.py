import time

from constants import ComputedConstants
from domain import Domain


def printList(l, name,file):
    f = open(file, "a")
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    f.write(name+ "= np.array("+ out+"\n")
    f.close()


X = 1
Y = 0.1
ls = 20e-3

nPart = 256000
T = 300
P = 1e5

for ii in range(7,16):
    ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
    ComputedConstants.dt *= 1
    nbDomains = 16
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

    domain = Domain(nbDomains, effectiveTemps=effectiveTemps, ratios=ratios)
    domain.setMaxWorkers(4)

    instants = []
    temps = []

    it = 0

    tMax = 24e-3 #s

    instants.append(ComputedConstants.time)
    temps.append(domain.getAveragedTemperatures())
    ComputedConstants.alphaAveraging = 0.05
    while ComputedConstants.time <= tMax:
        it += 1
        domain.update()

        if it%20 == 0:
            domain.getAveragedTemperatures()

        if it % 1000 == 1:
            time.sleep(3)
            print(100 * ComputedConstants.time/tMax, "%")
            print(domain.getAveragedTemperatures(), ComputedConstants.alphaAveraging)
        if it % 1000 == 1:
            instants.append(ComputedConstants.time)
            temps.append(domain.getAveragedTemperatures())

    file = "runls20mm_highdiff_"+str(ii)+".py"
    printList(instants, "ts",file)
    printList(temps, "temps",file)
