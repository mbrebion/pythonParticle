import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)


files = ["512_2_14_12.txt","512_2_14_25.txt","512_2_14_50.txt","512_2_14_75.txt","512_2_14_25_PLA.txt"]
files = ["512_2_14_12.txt","512_2_14_25.txt","512_2_14_50.txt","512_2_14_75.txt","512_2_14_75_PLA.txt"]
def model(x,V,ls):
    X0 = 0.32 # m
    vstar = 414  # m/s
    Y = 0.4 # m
    X = x*X0 # m
    t = (X0-X)/V  # time (s)

    w0 = 2*np.pi * vstar * np.sqrt(2/3) / (2*X)   # w varies with time
    ta = X / (vstar * np.sqrt(2/3))  # acoustic time
    invtv = 3/(4*np.sqrt(np.pi)) * ls * w0**2 / vstar  # viscous damping time
    tv = 1/invtv/1.5

    RP = 1.8* t *V*V * (ls/Y)**0.5  # permanent regime
    RT = (1-np.exp(-(t*4 /ta) )) + 0.4*np.exp(-t/tv) * np.sin(w0 * t)  # transitory regime
    e = RP + RT/2
    return 1+e * (V/vstar)**2

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
    ls = 0.4/float(axes[5].strip())

    for l in f:
        ds = l.split(",")
        x, y, ux, uy= ds[0], ds[1], ds[5], ds[6]
        xs.append(float(x))
        ys.append(float(y))
        uxs.append(float(ux))
        uys.append(float(uy))

    xs = np.array(xs)
    yTH = model(xs,414/14, ls)
    ys = np.array(ys)
    uys = np.array(uys)

    axesPlot[0].fill_between(xs, (ys - uys / 2) / ys[0], (ys + uys / 2) / ys[0], alpha=1, facecolor="none", label=legend,
                     hatch=hatch, edgecolor=col)
    axesPlot[1].fill_between(xs, (ys - uys / 2) / ys[0], (ys + uys / 2) / ys[0], alpha=1, facecolor="none", label=legend,
                     hatch=hatch, edgecolor=col)

    axesPlot[0].plot(xs,yTH,color=col)
    axesPlot[1].plot(xs, yTH, color=col)


fig, main_ax = plt.subplots(figsize=(16, 8));
main_ax.set_box_aspect(0.5)
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
plt.ylim(0.998, 1.015)
plt.xlim(0.49, 1.01)

inset_ax = main_ax.inset_axes(
    [0.7, 0.55, 0.25, 0.4],  # [x, y, width, height] w.r.t. axes
    xlim=[0.97, 1], ylim=[1, 1.004],  # sets viewport & tells relation to main axes
    xticklabels=[], yticklabels=[]
)
axes = [main_ax,inset_ax]
main_ax.indicate_inset_zoom(inset_ax, edgecolor="black",linewidth=2)

for i in range(len(files)):
    addPlot(files[i],axes)

plt.legend(loc="lower left", fontsize="16")
plt.grid()
plt.show()
