from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.5
Y = 0.1
ls = 5e-3

nPart = 5000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
ComputedConstants.dt *= 1
nbDomains = 5
ComputedConstants.alphaAveraging = 1
domain = Domain(nbDomains)

instants = []
temps = []

it = 0
itMax = 100e3
instants.append(ComputedConstants.time)
temps.append(domain.getAveragedTemperatures())
while it < itMax:
    it += 1
    domain.update()

    if it % 100 == 0:
        print(100 * it / itMax, "%")
        print(domain.getAveragedTemperatures())
    if it % 100 == 0:
        instants.append(ComputedConstants.time)
        temps.append(domain.getAveragedTemperatures())

printList(instants, "ts")
printList(temps, "temps")