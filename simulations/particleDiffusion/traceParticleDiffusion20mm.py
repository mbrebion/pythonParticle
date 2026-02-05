import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 16}
matplotlib.rc('font', **font)

#l = 20e-3 m
#Ns = 64000

D = 8.28/2
L = 1

def getTxt(xs,t):
    Tg = 0.7
    Td = 0.3

    cste = (Tg+Td)/2
    def harmo(n,xs,t):
        kn = np.pi/L + 2*n*np.pi/L

        return (Td-Tg)/(0.5+n)/np.pi * np.exp(-D * kn**2 * t/1000) * np.sin(kn * xs)

    sol = xs * 0 + cste
    for i in range(100):
        sol += harmo(i,xs,t)
    return sol


######

import runls20mm_lowdiff_1 as run
ts = run.ts
temps=[]
temps.append(np.copy(run.temps))


import runls20mm_lowdiff_2 as run
temps.append(np.copy(run.temps))

import runls20mm_lowdiff_3 as run
temps.append(np.copy(run.temps))

import runls20mm_lowdiff_4 as run
temps.append(np.copy(run.temps))

import runls20mm_lowdiff_5 as run
temps.append( np.copy(run.temps))
"""
import runls20mm_hd_6 as run
temps.append( np.copy(run.temps))
"""
######
#temps[0][0] += 0.01
#temps[1][0] -= 0.01
ts *= 1000

n = len(temps[0][0])
nstep = len(temps[0])


alpha = 1
for k in range(len(temps)):
    for j in range(1,len(temps[0])):
        for i in range(len(temps[0][0])-2,0,-1):
            temps[k][j][i] = alpha * temps[k][j][i] + (1 - alpha) * (temps[k][j][i - 1]+temps[k][j][i+1])/2


ftemps = []
utemps = []
for i in range(nstep):
    out = [0 for j in range(n)]
    uout = [0 for j in range(n)]
    for j in range(n):
        tps = np.array( [temps[k][i][j] for k in range(len(temps))] )

        out[j] = np.average(tps)
        uout[j] = 1.5*np.std(tps) / len(temps)**0.5
    ftemps.append(out)
    utemps.append(uout)

ftemps = np.array(ftemps)
utemps = np.array(utemps)

print(ftemps[0])

print(ftemps[5])


fig = plt.figure(figsize=(9, 5), dpi=120)
xs = np.array([(i + 0.5) / n * 1 for i in range(n)]) -0.5
xshr = np.array([(i + 0.5) / n * 0.1 for i in range(n*10)]) -0.5
plt.grid()
plt.xlabel("x (m)")
plt.ylabel("$\\alpha$")


nbPlot = 4
indices = [0]
for i in range(0,nbPlot-1 ):
    indices.append(int(nstep ** ((i + 2) / nbPlot)) - 2)

#indices[-1] -= 10
tau=  1/(D*(np.pi/L)**2) * 1000 # in ms

for i in indices:
    plt.fill_between(xs, ftemps[i] - utemps[i], ftemps[i] + utemps[i], alpha=0.5 + i / nstep * 0.5, facecolor='red', label="$t/\\tau_0'$ = " + str(int(ts[i]*100/tau)/100))

    t = ts[i]+1e-3
    solTH = getTxt(xshr,t)
    plt.plot(xshr, solTH, "-.k", alpha=0.5 + i / nstep * 0.5)

fig.tight_layout()
plt.legend()
plt.show()
