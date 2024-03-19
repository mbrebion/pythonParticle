import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'family': 'times',
        'size': 20}

matplotlib.rc('font', **font)

plt.grid()
plt.xlabel("$X(t)/X(0)$")
plt.ylabel("$r(X)$" )
#plt.ylim(0.95,1.05)

# 12000 particles, ls=Y/25

def addPlot(name,display):
    f = open(name,"r")
    ts=[]
    xs = []
    Ecs = []
    cstes = []
    for l in f:
        data = l.split()
        xs.append(float(data[0]))
        cstes.append((float(data[2])))

    xs = np.array(xs)/xs[0]
    cstes = np.array(cstes)

    plt.plot(xs , cstes ,display, label = "N = "+ name.split("/")[-1].split("X")[0])

addPlot("./1kXo25V10.txt", "-.k")
addPlot("./2kXo25V10.txt", "-.b")
addPlot("./4kXo25V10.txt", "-.r")
addPlot("./8kXo25V10.txt", "-.g")
addPlot("./16kXo25V10.txt", "-k")
addPlot("./32kXo25V10.txt", "-b")
addPlot("./64kXo25V10.txt", "-r")
plt.legend()
plt.show()