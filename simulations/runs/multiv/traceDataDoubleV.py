import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)

files = ["512_2_28_14_50_PLA.txt","512_2_14_28_50_PLA.txt","512_2_14_14_50_PLA.txt"]

# TODO : tracer Ec - Ec,rev en fonction de X ou de t
def addPlot(name):
    f = open(name, "r")
    xs = []
    Ecs = []
    EcRevs = []
    EcMs = []
    uEcs = []
    axes = f.readline().split("\n")[0].split(",")
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = axes[2]
    col = axes[3].strip()
    hatch = axes[4].strip()

    for l in f:
        ds = l.split(",")
        x, y, ux, uy = float(ds[0]), float(ds[2]), float(ds[5]), float(ds[7])  # X , Ec
        xs.append(x)
        Ecs.append(y)
        uEcs.append(uy)
        EcMs.append(float(ds[3]))
        EcRevs.append(Ecs[0] * xs[0] / x) # adiabatic rev 2D compression

    xs = np.array(xs)
    EcRevs = np.array(EcRevs)
    EcMs = np.array(EcMs)
    Ecs = np.array(Ecs)
    uys = np.array(uEcs)

    ys = Ecs - EcRevs
    ys /= Ecs[0]
    plt.plot(xs, ys, alpha=1, label=legend,color=col)
    plt.fill_between(xs, 0,  EcMs/ Ecs[0], alpha=0.35, facecolor="none",
                     hatch=hatch, edgecolor=col)


fig, main_ax = plt.subplots(figsize=(16, 8));
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
#plt.ylim(0.999, 1.006)
plt.xlim(0.49, 1.06)


for i in range(len(files)):
    addPlot(files[i])

plt.legend(loc="upper right", fontsize="16")
plt.grid()
plt.show()
