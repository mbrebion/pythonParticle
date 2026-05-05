from domain import Domain
from numbaAcc import measuring
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def printList(l, name,file):
    f = open(file, "a")
    f.write(name+ "= "+ repr(l)+"\n")
    f.close()



X = 1
Y = 0.1
ls = 20e-3

nPart = 1024e3*2
T = 298.7
P = 1e5
nc = 1024

for ii in range(0,8):
    tHigh = 320.
    tLow = 280.
    alpha = tLow / tHigh
    rg = alpha / (1 + alpha)
    rd = 1 - rg
    rg,rd = 2*rg, 2*rd

    effectiveTemps = [tHigh for i in range(nc)]
    ratios = [rg / nc for i in range(nc)]
    for j in range(nc // 2, nc):
        effectiveTemps[j] = tLow
        ratios[j] = rd / nc

    domain = Domain(nc, T, X, Y, P, nPart, ls, drOverLs=0.0025, maxWorkers=2, ratios = ratios,effectiveTemps=effectiveTemps)

    instants = []
    temps = []
    ns = []

    it = 0

    tMax = 24e-3*3/2 #s # thermaldiffusive time is half of particle diffusive time

    instants.append(domain.csts["time"])
    proto =  [0. for _ in range(20)]
    domain.computeParam(measuring.computeTemperature, extensive=False, array=proto)
    temps.append([ T for T in proto])
    domain.computeParam(measuring.countAlive, extensive=True, array=proto)
    ns.append([N for N in proto])


    file = "runls20mm_final_" + str(ii) + ".py"
    f = open(file, "a")
    f.write("from numpy import array \n \n")
    f.close()

    while domain.csts["time"] <= tMax:
        it += 1
        domain.update()

        if it % 1000 == 1:
            print(100 * domain.csts["time"]/tMax, "%")
            domain.computeParam(measuring.computeTemperature, extensive=False, array=proto)
            tps = [round(T,4) for T in proto]
            print(tps)
            domain.computeParam(measuring.countAlive, extensive=True, array=proto)
            NNs = [round(N,2) for N in proto]
            print(NNs)
            print()

            instants.append(float(domain.csts["time"]))
            temps.append(tps)
            ns.append(NNs)


    printList(np.array(instants), "ts",file)
    printList(np.array(temps), "temps",file)
    printList(np.array(ns), "ns", file)