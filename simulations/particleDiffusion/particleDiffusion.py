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

nPart = 128000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

nbDomains = 16

leftColorRatio=0.7
rightColorRatio = 1-leftColorRatio
colorRatios = [leftColorRatio if i < nbDomains//2 else rightColorRatio for i in range(nbDomains)]


domain = Domain(nbDomains, colorRatios=colorRatios)
domain.setMaxWorkers(4)

instants = []
temps = []

it = 0
tMax = 24e-3 #s

instants.append(ComputedConstants.time)


while ComputedConstants.time <= tMax:
    it += 1
    domain.update()

    if it % 500 == 1:
        time.sleep(0.25)
        print(100 * ComputedConstants.time/tMax, "%")
        print(domain.getColorRatios())
        print()
    if it % 500 == 1:
        instants.append(ComputedConstants.time)
        temps.append(domain.getColorRatios())

file = "runls20mm_lowdiff_5.py"
printList(instants, "ts",file)
printList(temps, "temps",file)
