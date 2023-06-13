import time
from domain import Domain
from constants import ComputedConstants
import numpy as np

X = 0.1
Y = 0.1
ls = 50e-3
nPart = 16000
T = 300
P = 1e5

# P (V - N b) = N k T

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(1)
domain.setMaxWorkers(1)

ps = []
ts = []

it = 0
while it < 16000:
    it += 1
    domain.update()

    ps.append(domain.computePressure())
    ts.append(domain.computeTemperature())
    if it % 400 == 0:
        print(it, ComputedConstants.time, domain.computePressure(), domain.computeTemperature())
        #time.sleep(0.2)

ps = np.array(ps)
p = np.average(ps)
up = np.std(ps) / np.sqrt(len(ps))
print("pression moyenne : ", p, " +\- ", up, " pa")

ts = np.array(ts)
t = np.average(ts)
ut = np.std(ts) / np.sqrt(len(ts))
print("temperature moyenne : ", t, " +\- ", ut, " K")

ratios = ps * X * Y / (nPart * ComputedConstants.kbs * ts)

print("PS/(NKbT) = ", np.average(ratios), " +\- ", np.std(ratios)/np.sqrt(len(ratios)))