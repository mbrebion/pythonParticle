import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'family': 'times',
        'size': 20}

matplotlib.rc('font', **font)

plt.grid()
plt.xlabel("$x$")
plt.ylabel("$E_c \\times  (Hx - 2Ns) $" )
#plt.ylim(0.95,1.05)

# 12000 particles, ls=Y/25

def addPlot(name,display):
    f = open(name,"r")
    ts=[]
    xs = []
    Ecs = []
    cstes = []
    for l in f:
        t,x,Ec,Ecr,cste = l.split()
        ts.append(float(t))
        xs.append(float(x))
        Ecs.append(float(Ec))
        cstes.append((float(cste)))

    xs = np.array(xs)
    cstes = np.array(cstes)

    plt.plot(xs , cstes ,display, label = name)

addPlot("0p05mps.txt", "-.k")
addPlot("0p25mps.txt", "--g")
addPlot("1mps.txt", "--b")
addPlot("5mps.txt", "--r")
addPlot("15mps.txt", "--y")
addPlot("45mps.txt", "-.g")
addPlot("90mps.txt", "-.b")
addPlot("90mpsbis.txt", "-.k")
addPlot("180mps.txt", "-.r")

plt.legend()
plt.show()