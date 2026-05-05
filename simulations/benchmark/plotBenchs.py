import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
plt.rcParams['font.size'] = 14


fig = plt.figure(figsize=(7, 4), dpi=120)
ax = plt.axes()
ax.set_yscale('log')
ax.set_xscale('log')


ax.set_xlim([2*10**3, 2*10**8])


ns = np.array([  4000,   8000,   16000,  32000, 64000,  128000,  256000,  512000,  1024000, 2048000, 4096000, 8192000,  16384000, 32768000, 65536000, 10**8])
ncs1  = np.array([8,     16,     32,     50,     64,      100,     160,     200,     320,     512,    640,     1000,      1600,       -1,      -1,     -1])
tpts1 = np.array([0.020, 0.038,  0.078,  0.156,  0.340,  0.799,   2.05,    6.13,    16.8,    47.6,    132.,     356,      1014,       -1,      -1,     -1])

ncs2  = np.array([-1,    -1,     -1,     -1,     64,      100,     160,     256,     320,     512,     800,    1280,    1600,     2000,    3200,    3200])
tpts2 = np.array([-1,    -1,     -1,     -1,   0.24,     0.43,    0.82,    1.70,    3.75,    8.27,    18.4,    42.2,     108,      303,     854,    1663])

ncs3  = np.array([-1,    -1,     -1,     20,      40,      64,     100,    160,     200,     320,     400,     512,      800,    1000,     1600,    -1])
tpts3 = np.array([-1,    -1,     -1,    0.109,  0.203,  0.396,    0.83,   1.85,    4.54,    10.5,    27.9,    75.3,      215,     613,     1776,    -1])


##############################################
####### display of computational times #######
##############################################

if True:

    ax.set_ylim([0.01, 2000])
    # X=2, Y=1, l = 50e-3 ; T=300
    plt.errorbar(ns,tpts1,yerr = abs(tpts1)*0.07, fmt = 'none', capsize = 4, ecolor = 'red' , elinewidth = 2, capthick = 2, label="$\\ell_s = 50$ mm")
    # X=2, Y=1, l = 5e-3 ; T=300
    plt.errorbar(ns,tpts2,yerr = abs(tpts2)*0.07, fmt = 'none', capsize = 4, ecolor = 'blue', elinewidth = 2, capthick = 2,  label="$\\ell_s = 5$ mm")
    # X=0.01, Y=0.02, l = 1e-4 ; T=100
    plt.errorbar(ns,tpts3,yerr = abs(tpts3)*0.07, fmt = 'none', capsize = 4, ecolor = 'green', elinewidth = 2, capthick = 2,  label="$\\ell_s = 0.1$ mm")


    # low N region
    xs1 = np.array([400,80000])
    comp = np.array([80000,1e8])
    ys1_low = xs1 / 400*0.001
    ys1_high = xs1 / 400*0.003
    ax.plot(xs1, ys1_low, "k--", lw=1, alpha=0.8,label="$\\mathcal{O}(N)$")
    ax.plot(xs1, ys1_high, "k--", lw=1, alpha=0.8)
    ax.plot(comp, comp/400*0.001, "k--", lw=1, alpha=0.3)
    ax.plot(comp, comp/400*0.003, "k--", lw=1, alpha=0.3)

    # high N region
    ns = 2048e3
    xs1 = np.array([ns,2e8])
    comp = np.array([400,ns])
    ys1_low = xs1**1.5 / ns**1.5*55
    ys1_high = xs1**1.5 / ns**1.5*4
    ax.plot(xs1, ys1_low, "k-", lw=1, alpha=0.8,label="$\\mathcal{O}(N^{3/2})$")
    ax.plot(xs1, ys1_high, "k-", lw=1, alpha=0.8)
    ax.plot(comp, comp**1.5/ns**1.5*55, "k-", lw=1, alpha=0.3)
    ax.plot(comp, comp**1.5/ns**1.5*4, "k-", lw=1, alpha=0.3)
    plt.ylabel("$\\Delta t_{\\tau}$ (s)")

##############################################
#######      display of n_{c,B}        #######
##############################################
else:
    ax.set_ylim([8, 4000])
    # X=2, Y=1, l = 50e-3 ; T=300
    d = 1/4.
    c = 1/15
    plt.plot(ns, ncs1,"xr",label="$\\ell_s = 50$ mm", markerfacecolor="w" )
    mod = [min(d*(n*2)**0.5, c*n*(0.02*50e-3/1)**0.5) for n in ns]
    plt.plot(ns,mod, "r",linestyle=(0, (3, 6)), alpha=0.5)
    # X=2, Y=1, l = 5e-3 ; T=300
    plt.plot(ns, ncs2,"ob",label="$\\ell_s = 5$ mm" , markerfacecolor="w")
    mod = [min(d * (n * 2) ** 0.5, c * n * (0.02 * 5e-3 / 1) ** 0.5) for n in ns]
    plt.plot(ns, mod, "b",linestyle=(3, (3, 6)), alpha=0.5)
    # X=0.01, Y=0.02, l = 1e-4 ; T=100
    plt.plot(ns, ncs3,"sg",label="$\\ell_s = 0.1$ mm", markerfacecolor="w" )
    mod = [min(d * (n / 2) ** 0.5, c * n * (0.02 * 1e-4 / 0.02) ** 0.5) for n in ns]
    plt.plot(ns, mod, "--g", alpha=0.5)

    plt.ylabel("$n_{\\rm c, Best}$ ")


plt.grid()
plt.legend()
plt.xlabel("N")

plt.tight_layout()
plt.show()
