import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from  scipy.ndimage import convolve1d

font = {'family': 'times',
        'size': 16}
matplotlib.rc('font', **font)

ls = 20e-3
vs = 414.03

L = 1
Tg = 0.7
Td = 0.3


D = ls * vs / np.sqrt(np.pi)
print("D = "  , round(D,4), "m^2/s")
tau = 1/(D*np.pi**2/L**2)
print("tau = "  , round(tau,8), "s")

def getTxt(xs,t):


    cste = (Tg+Td)/2
    def harmo(n,xs,t):
        kn = np.pi/L + 2*n*np.pi/L
        taun = 1/(D*kn**2)
        return (Td-Tg)/(0.5+n)/np.pi * np.exp(- t / taun) * np.sin(kn * xs)

    sol = xs * 0 + cste
    for i in range(100):
        sol += harmo(i,xs,t)
    return sol

#import runls20mm_final_1 as run
#ts = run.ts
#temps=[]
#temps.append(np.copy(run.temps))

#import runls20mm_final_2 as run
#temps.append(np.copy(run.temps))


######

import runls20mm_HD_1 as run
ts = run.ts
temps=[]
temps.append(np.copy(run.temps))

import runls20mm_HD_2 as run
temps.append(np.copy(run.temps))

import runls20mm_HD_3 as run
temps.append(np.copy(run.temps))

import runls20mm_HD_4 as run
temps.append(np.copy(run.temps))

import runls20mm_HD_5 as run
temps.append(np.copy(run.temps))

import runls20mm_HD_6 as run
temps.append(np.copy(run.temps))

n = len(temps[0][0])
nstep = len(temps[0])


#mask = [0.25, 0.5, 0.25]
#mask = [1/2 for i in range(2)]
#mask = [1]
#temps = convolve1d(temps, mask, axis=-2, )



ftemps = []
utemps = []
for i in range(nstep):
    out = [0 for j in range(n)]
    uout = [0 for j in range(n)]
    for j in range(n):
        tps = np.array( [temps[k][i][j] for k in range(len(temps))] )

        out[j] = np.average(tps)
        uout[j] = 2.45 * np.std(tps) / len(temps)**0.5
    ftemps.append(out)
    utemps.append(uout)

ftemps = np.array(ftemps)
utemps = np.array(utemps)


fig = plt.figure(figsize=(9, 5), dpi=120)
xs = np.array([(i + 0.5) / n * 1 for i in range(n)]) -0.5
xshr = np.array([(i + 0.5) / n * 0.1 for i in range(n*10)]) -0.5

plt.grid()
plt.xlabel("x (m)")
plt.ylabel("$\\alpha$")

n = len(ftemps)
indices = [n//25, n//9,  n//3, n-1]
print(indices)
alphas = [0.35, 0.55, 0.75, 0.95]


def initial(xs):
    return np.array([Tg if x<0 else Td for x in xs])

plt.plot(xshr, initial(xshr),"-.k", alpha=0.7  )#, label="$t/\\tau_0$ = 0"  )
plt.xticks([-0.5,-0.25, 0, 0.25, 0.5])

for j in range(len(indices)):
    i = indices[j]
    plt.fill_between(xs, ftemps[i] - utemps[i]-0.0005, ftemps[i] + utemps[i]+0.0005, alpha=alphas[j], facecolor='red', label="$t/\\tau_{\\rm p}$ = " + str(round(ts[i]/tau,3)) )
    solTH = getTxt(xshr,ts[i])
    plt.plot(xshr, solTH, "-.k", alpha=0.7)
    print(alphas[j])

fig.tight_layout()
plt.legend()
plt.show()
