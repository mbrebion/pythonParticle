import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from constants import Kb, MASS
from  scipy.ndimage import convolve1d

font = {'family': 'times',
        'size': 16}
matplotlib.rc('font', **font)
plt.rcParams['font.family'] = 'times'


Tg = 320 #K
Td = 280 #K
L=1 #m
N= 256000
n0 = N/L
ng = 2*n0 * Td / (Tg+Td)
nd = 2*n0 * Tg / (Tg+Td)
Teq = (ng*Tg + nd*Td) / (2*n0)

ls = 20e-3 #m
vstar = (2 *  Kb/ MASS * Teq) ** 0.5
D = ls * vstar * 2 / np.sqrt(np.pi)


def getTemperature(xs, t):
    # diffusion equation for n is solved, then T = n0 T_eq / n

    cste = (ng+nd)/2
    def harmo(p,xs,t):
        kn = np.pi/L + 2*p*np.pi/L
        taup = 1 / (D*kn**2)
        return (nd-ng)/(0.5+p)/np.pi * np.exp(-t/taup) * np.sin(kn * xs)

    n = xs * 0 + cste
    for i in range(100):
        n += harmo(i,xs,t)
    return Teq * n0 / n


######

import runls20mm_HD_1 as run
ts = run.ts
temps=[]
ns = []

temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_1 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_2 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_3 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_4 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_5 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_6 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_7 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_8 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_9 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_10 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_11 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_12 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_13 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

import runls20mm_HD_14 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))


import runls20mm_HD_15 as run
temps.append(np.copy(run.temps))
ns.append(np.copy(run.ns))

""""""

n = len(temps[0][0])
nstep = len(temps[0])


mask = [0.2, 0.6, 0.2]
mask = [0.1, 0.2, 0.4, 0.2, 0.1]
mask=[0.5, 0.5]
#mask = [1]

temps = convolve1d(temps, mask, axis=-2, )
ns = convolve1d(ns, mask, axis=-2, )
#temps = convolve1d(temps, mask, axis=-2, )
#ns = convolve1d(ns, mask, axis=-2, )


ftemps = []
utemps = []

for i in range(nstep):
    out = [0 for j in range(n)]
    uout = [0 for j in range(n)]
    for j in range(n):
        tps = np.array( [temps[k][i][j] for k in range(len(temps))] )
        out[j] = np.average(tps)
        uout[j] = 2*np.std(tps)/(len(temps)**0.5)
    ftemps.append(out)
    utemps.append(uout)

ftemps = np.array(ftemps)
utemps = np.array(utemps)

fig = plt.figure(figsize=(9, 5), dpi=120)
xs = np.array([(i + 0.5) / n * 1 for i in range(n)]) -0.5
xshr = np.array([(i + 0.5) / n * 0.05 for i in range(n*20)]) -0.5
plt.grid()
plt.xlabel("x (m)")
plt.ylabel("T (K)")

plt.xticks([-0.5,-0.25, 0, 0.25, 0.5])
indices = [ 2, 14, 40, 100]
alphas = [0.35, 0.55, 0.75, 0.95]

tau=  1/(D*(np.pi/L)**2)  # in s

# initial profile
def initial(xs):
    return np.array([Tg if x<0 else Td for x in xs])

plt.plot(xshr, xshr*0 + Teq,"--b",alpha=0.5, label = "$T_{\\rm eq} \\approx $"+str(round(Teq,1))+" K" )
plt.plot(xshr, initial(xshr),"-.k", alpha=0.7  )#, label="$t/\\tau_0$ = 0"  )
for j in range(len(indices)):
    i = indices[j]
    plt.fill_between(xs, ftemps[i] - utemps[i], ftemps[i] + utemps[i], alpha=alphas[j], facecolor='red', label="$t/\\tau_0$ = " + str(round(ts[i]/tau,3))  )
    t = ts[i]
    solTH = getTemperature(xshr, t)
    plt.plot(xshr, solTH, "-.k", alpha=0.7)

fig.tight_layout()
plt.legend()
plt.show()
