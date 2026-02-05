import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'family': 'times',
        'size': 20}

matplotlib.rc('font', **font)

etas = np.array([1e-4,       3e-4,      1e-3,      3e-3,      1e-2,      3e-2,   5e-2,     1e-1,     2e-1,     3e-1,      4e-1,     5e-1,     6e-1,     7e-1])
zs = np.array([1.000204,   1.00062,   1.00208,   1.00597,   1.01999,    1.0626, 1.109,   1.2344,   1.5522,   2.0524,    2.6980,   3.997,     5.991,    9.37])-1
etas = np.array([1e-4,       3e-4,      1e-3,      3e-3,      1e-2,      3e-2,   5e-2,     1e-1,     2e-1,     3e-1,  3.5e-1,    3.8e-1,    ])
zs = np.array([1.000204,   1.00062,   1.00208,   1.00597,   1.01999,    1.0626, 1.109,   1.2344,   1.5522,   2.0524,  2.39  ,  2.615  ,   ])-1

zths = 1/(1-etas)**2 -1
plt.loglog(etas,zths,"-b", label="EOS Helfand")

nSimple=len(etas)
zsimples = 1/(1-2*etas[:nSimple])-1
plt.loglog(etas[:nSimple],zsimples,"--r", label="EOS 2", linewidth=2)
plt.loglog(etas,zs,"+k", label="Simulation",markersize=10,markeredgewidth=2)


plt.xlabel("$\\eta_S$")
plt.ylabel("$Z-1$")
plt.legend()
plt.grid()
plt.show()