import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)
plt.figure(figsize=(16, 8))
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)

files = ["256_4_50.txt","256_8_25.txt","256_16_10.txt","256_32_8.txt","256_16_6.txt","256_16_5.txt","256_8_4.txt"]


plt.grid()

plt.ylim(0.999, 1.08)


def addPlot(name):
    f = open(name, "r")
    xs = []
    ys = []
    uxs = []
    uys = []
    axes = f.readline().split("\n")[0].split(",")
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = axes[2]
    col = axes[3].strip()
    hatch = axes[4].strip()

    for l in f:
        x,y,ux,uy = l.split(",")
        xs.append(float(x))
        ys.append(float(y))
        uxs.append(float(ux))
        uys.append(float(uy))

    xs = np.array(xs)
    ys = np.array(ys)
    uys = np.array(uys)

    plt.fill_between(xs, (ys - uys/2)/ys[0], (ys + uys/2)/ys[0], alpha=1, facecolor="none", label=legend, hatch=hatch, edgecolor=col)



for i in range(len(files)):
    addPlot(files[i])

plt.legend(loc="upper right")
plt.show()
