import os
import time

from constants import ComputedConstants
from domain import Domain


# il faudrait lancer quelques simu avec frottement aux parois, pour différentes valeurs de Y
X = 0.4*2
L = X / 2
Y = 0.1
ls = L / 25
nPart = 260000
T = 300
P = 1e5
v = 6
m = 2

# long : x=0.8m
# verylong x=1.6m
outputName = "xols" + str(round(L / ls)) + "p" + str(round(P / 1e5)) + "m" + str(m) + "v" + str(v) + "N" + str(
    nPart // 1000) + "k_T" + str(T) + "doubleLruguous.txt"
if os.path.exists(outputName):
    print("file", outputName, " already exists ; abort")
    exit()

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(65)
domain.setMaxWorkers(3)
outputColor = "purple"
domain.addMovingWall(m, X / 2, v)
print(domain.computeKineticEnergyLeftSide(), domain.computeKineticEnergyRightSide(), domain.countLeft(),
      domain.countRight())
first = True

f = open(outputName, "w")
f.write(outputColor + " , m=" + str(m) + ", v=" + str(v))
f.write(", xols=" + str(round(L / ls)) + ", N=" + str(nPart) + ", T=" + str(T) + ", P=" + str(P))
f.write(", X=" + str(X) + ", Y=" + str(Y) + " \n")

while True:
    domain.update()

    if first or ComputedConstants.it % 200 == 0:
        first = False
        ecl = domain.computeKineticEnergyLeftSide()
        ecr = domain.computeKineticEnergyRightSide()

        out = str(ComputedConstants.time) + ", " + str(domain.wall.location() / X) + ", " + str(
            domain.wall.velocity()) + ", " + str(ecl) + ", " + str(ecr) + ", " + str(domain.countLeft())
        print(out)
        f.write(out + "\n")
        f.flush()
        time.sleep(0.5)
