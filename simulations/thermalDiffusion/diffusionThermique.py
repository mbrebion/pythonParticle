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

nPart = 1024000*2
T = 298.7
P = 1e5

for ii in range(0,8):
    ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
    ComputedConstants.dt *= 1
    nbDomains = 256
    tHigh = 320.
    tLow = 280.
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
    domain.setMaxWorkers(2)

    instants = []
    temps = []
    ns = []

    it = 0

    tMax = 24e-3 #s

    instants.append(ComputedConstants.time)
    temps.append(domain.getTemperatures(8))
    ns.append(domain.getCounts(8))

    file = "runls20mm_VHD_" + str(ii) + ".py"
    f = open(file, "a")
    f.write("import numpy as np")
    f.close()

    while ComputedConstants.time <= tMax:
        it += 1
        domain.update()

        if it % 75 == 1:
            time.sleep(3)
            print(100 * ComputedConstants.time/tMax, "%")
            print(domain.getTemperatures(8))
            print(domain.getCounts(8))
            print()
        if it % 75 == 1:
            instants.append(ComputedConstants.time)
            temps.append(domain.getTemperatures(8))
            ns.append(domain.getCounts(8))


    printList(instants, "ts",file)
    printList(temps, "temps",file)
    printList(ns, "ns", file)
