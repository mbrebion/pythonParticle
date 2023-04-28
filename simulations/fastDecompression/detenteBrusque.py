from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.2
Y = 0.1
ls = 4e-3

nPart = 5000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

domain = Domain(2, [0.95,0.05])

ComputedConstants.dt = 4.83046e-07
ns0 = []
ns1 = []
instants = []

it = 0
itMax = 50e3
while it < itMax:
    it += 1
    domain.update()
    if it % 500 == 0:
        print(100 * it // itMax, "%")
    if it % 100 == 0:
        instants.append(ComputedConstants.time)
        ns0.append(domain.cells[0].count())
        ns1.append(domain.cells[1].count())
        print(domain.cells[0].averagedTemperature,domain.cells[1].averagedTemperature)

printList(instants,"ts")
printList(ns0,"ns0")
printList(ns1,"ns1")
