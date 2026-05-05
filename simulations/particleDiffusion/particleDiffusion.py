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
Y = 0.25
ls = 20e-3

nPart = 1024000
T = 300
P = 1e5
nc = 160*2



leftColorRatio=0.7
rightColorRatio = 1-leftColorRatio
colorRatios = [leftColorRatio if i < nc//2 else rightColorRatio for i in range(nc)]
domain = Domain( nc, T, X, Y, P, nPart, ls, drOverLs=0.006,maxWorkers=1,colorRatios=colorRatios)

instants = []
temps = []

it = 0
tMax = 24e-3*3 #s

instants.append(domain.csts["time"])
ratios = [0. for _ in range(20)]

while domain.csts["time"] <= tMax:
    it += 1
    domain.update()

    if it % 1000 == 1:
        print(100 * domain.csts["time"]/tMax, "%")

        domain.computeParam(measuring.computeColorRatio,extensive=False, array=ratios)
        print(ratios)
        print()
        instants.append(float(domain.csts["time"]))
        temps.append([round(r,5) for r in ratios])

file = "runls20mm_final_3.py"
f = open(file, "w")
f.write("from numpy import array\n")
f.write("\n")
f.close()
printList(np.array(instants), "ts",file)
printList(np.array(temps), "temps",file)
