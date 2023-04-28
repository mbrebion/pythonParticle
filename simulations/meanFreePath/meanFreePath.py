from constants import ComputedConstants
from domain import Domain
import numpy as np


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.1
Y = 0.1
ls = 5e-3

nPart = 5000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

domain = Domain(1)

nTracker = 10
for i in range(nTracker):
    domain.addTracker(int((i + 0.5) * nPart / nTracker))

ls = []
ts = []

it = 0
itMax = 5e3
while it < itMax:
    it += 1
    domain.update()
    if it % 500 == 0:
        print(100 * it // itMax, "%")

for t in domain.trackers:
    l, ul, tl, utl = t.getMeanFreePathresults()
    ls.append(l)
    ts.append(tl)
    print("Particle ", t.id, " : ")
    print("l = ", "{:.2e}".format(l), " +/- ", "{:.2e}".format(ul), " m")
    print("tl = ", "{:.2e}".format(tl), " +/- ", "{:.2e}".format(utl), " s")
    print()

ls = np.array(ls)
ts = np.array(ts)

print()
print("Averaging performed on all trackers")
print()
print("l_s = ", "{:.2e}".format(np.average(ls)), " +/- ", "{:.2e}".format(np.std(ls) / nTracker ** 0.5), " m")
print("t_ls = ", "{:.2e}".format(np.average(ts)), " +/- ", "{:.2e}".format(np.std(ts) / nTracker ** 0.5), " s")
