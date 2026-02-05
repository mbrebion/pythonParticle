import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)


#files = ["256_64_20_12_PRA.txt","256_64_20_25_PRA.txt","512_32_20_50_PRA.txt","1024_32_20_75_PRA.txt","512_32_20_25_PLA.txt"]
files = ["256_32_20_12_0.2_0.1_PRA.txt","256_32_20_25_0.2_0.1_PRA.txt","512_32_20_50_0.2_0.1_PRA.txt","1024_32_20_75_0.2_0.1_PRA.txt","256_32_20_25_0.2_0.1_PLA.txt"]

def addPlot(name,axesPlot):
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
        ds = l.split(",")
        x, y, ux, uy= ds[0], ds[1], ds[5], ds[6]
        xs.append(float(x))
        ys.append(float(y))
        uxs.append(float(ux))
        uys.append(float(uy))

    xs = np.array(xs)

    ys = np.array(ys)
    uys = np.array(uys)

    axesPlot[0].fill_between(xs, (ys - uys / 2) / ys[0], (ys + uys / 2) / ys[0], alpha=1, facecolor="none", label=legend,
                     hatch=hatch, edgecolor=col)
    axesPlot[1].fill_between(xs, (ys - uys / 2) / ys[0], (ys + uys / 2) / ys[0], alpha=1, facecolor="none", label=legend,
                     hatch=hatch, edgecolor=col)


fig, main_ax = plt.subplots(figsize=(16, 8));
main_ax.set_box_aspect(0.5)
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
plt.ylim(0.999, 1.017)
plt.xlim(0.49, 1.01)

inset_ax = main_ax.inset_axes(
    [0.62, 0.52, 0.34, 0.44],  # [x, y, width, height] w.r.t. axes
    xlim=[0.94, 1], ylim=[1, 1.004],  # sets viewport & tells relation to main axes
    xticklabels=[], yticklabels=[]
)
axes = [main_ax,inset_ax]
main_ax.indicate_inset_zoom(inset_ax, edgecolor="black",linewidth=2)

for i in range(len(files)):
    addPlot(files[i],axes)

plt.legend(loc="lower left", fontsize="16")
plt.grid()
plt.show()
