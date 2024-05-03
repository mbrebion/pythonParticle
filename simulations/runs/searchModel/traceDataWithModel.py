import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import solve_de

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)

files = ["512_2_14_50_PLA.txt", "512_2_20_50_PLA.txt","512_2_28_50_PLA.txt"]
#files = ["512_2_14_25_PLA.txt","512_2_14_50_PLA.txt", "512_2_14_75_PLA.txt"]

# TODO : tracer Ec - Ec,rev en fonction de X ou de t
def addPlot(name):
    f = open(name, "r")
    xs = []
    Ecs = []
    rs = []
    urs = []
    EcRevs = []
    EcMs = []
    uEcs = []
    axes = f.readline().split("\n")[0].split(",")
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = axes[2]
    ratio = 1 / float(legend.split("=")[1].split("$")[0].split("/")[1].lstrip())
    col = axes[3].strip()
    hatch = axes[4].strip()

    lsOx = 1./float(name.split("_")[3])

    X0 = 0.32  # m
    Y = 0.4  # m
    xstar = 2 * 512000 * np.pi * (1.38107e-05 / 2) ** 2 / (
            Y * Y)  # correctif pour passer du GP au gaz réel : il s'agit du double de la compacité

    for l in f:
        ds = l.split(",")
        x,r, y, ux,ur, uy = float(ds[0]),float(ds[1]), float(ds[2]),float(ds[4]), float(ds[6]), float(ds[7])  # X , Ec
        rs.append(r)
        urs.append(ur)
        xs.append(x)
        Ecs.append(y)
        uEcs.append(uy)
        EcMs.append(float(ds[3]))
        EcRevs.append(Ecs[0] * (1-xstar) / (x-xstar))  # adiabatic rev 2D compression

    xs = np.array(xs)
    rs = np.array(rs)
    urs = np.array(urs)
    EcRevs = np.array(EcRevs)
    EcMs = np.array(EcMs)
    Ecs = np.array(Ecs)
    uys = np.array(uEcs)

    ys = Ecs - EcRevs
    ys /= Ecs[0]
    uys /= Ecs[0]
    #plt.fill_between(xs, rs - urs/2, rs + urs/2, alpha=1, facecolor="none", label=legend, hatch=hatch, edgecolor=col)
    #plt.fill_between(xs, ys - uys / 2, ys + uys / 2, alpha=1, facecolor="none", label=legend, hatch=hatch, edgecolor=col)

    # model
    gamma = ratio


    plt.plot(xs, Ecs * (xs-xstar) / (Ecs[0]*(1-xstar)), "--", color=col)

    xs,us =solve_de.getModel(gamma,lsOx)
    modelrs = 1 +  us * (xs-xstar) / (1 - xstar)

    #plt.plot(xs, modelrs, "-.", color=col)
    #plt.plot(xs, us, "--", color=col)



fig, main_ax = plt.subplots(figsize=(16, 8));
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
# plt.ylim(0.999, 1.006)
plt.xlim(0.49, 1.06)

for i in range(len(files)):
    addPlot(files[i])

plt.legend(loc="upper right", fontsize="16")
plt.grid()
plt.show()
