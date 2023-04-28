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
ls = 20e-3

nPart = 80000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
ComputedConstants.dt *= 1
nbDomains = 10
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

instants = []
temps = []

it = 0
itMax = 10e3
instants.append(ComputedConstants.time)
temps.append(domain.getAveragedTemperatures())
while it < itMax:
    it += 1
    domain.update()

    if it > 500:
        ComputedConstants.alphaAveraging = 0.01
    if it > 2000:
        ComputedConstants.alphaAveraging = 0.005
    if it > 5000:
        ComputedConstants.alphaAveraging = 0.001

    if it % 100 == 0:
        print(100 * it / itMax, "%")
        print(domain.getAveragedTemperatures(),ComputedConstants.alphaAveraging)
    if it % 100 == 0:
        instants.append(ComputedConstants.time)
        temps.append(domain.getAveragedTemperatures())

printList(instants, "ts")
printList(temps, "temps")
