from constants import ComputedConstants
from domain import Domain
import numpy as np


X = 0.1
Y = 0.1
lsa = 0.005
nTracker = 1000
nPart = 256000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, lsa)

domain = Domain(32)
domain.setMaxWorkers(4)
ComputedConstants.dt /= 1

# warming
it = 0
while it < 500:
    it += 1
    domain.update()


for i in range(nTracker):
    domain.addTracker(int((i + 0.5) * nPart / nTracker))

ls = []
ts = []

it = 0
itMax = 50e3
while it < itMax:
    it += 1
    domain.update()
    if it % (itMax//20) == 0:
        print(100 * it // itMax, "%")

for t in domain.trackers:
    ls += t.ds
    ts += t.ts

ls = np.array(ls)
ts = np.array(ts)

print()
print("Averaging performed on all trackers")
print()
print(len(ls), " events occured")
print(ls)
print("l_s/l_sasked = ", "{:.3e}".format(np.average(ls)/lsa), " +/- ", "{:.3e}".format(np.std(ls) / len(ls) ** 0.5 /lsa), " m")
print("t_ls = ", "{:.2e}".format(np.average(ts)), " +/- ", "{:.2e}".format(np.std(ts) / nTracker ** 0.5), " s")
