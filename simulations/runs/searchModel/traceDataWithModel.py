import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import solve_de

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)

#files = ["256_32_14_25_0.2_0.1_PRA.txt","256_32_20_25_0.2_0.1_PRA.txt","256_32_28_25_0.2_0.1_PRA.txt"]
files = ["512_32_14_25_0.2_0.1_PRA.txt","512_32_20_25_0.2_0.1_PRA.txt","512_32_28_25_0.2_0.1_PRA.txt"]
# TODO : tracer Ec - Ec,rev en fonction de X ou de t
def addPlot(name):
    f = open(name, "r")
    xs = []
    Ecs = []
    rs = []
    urs = []
    EcRevs = []
    EcMs = []
    uEcMs = []
    uEcs = []
    axes = f.readline().split("\n")[0].split(",")
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = axes[2]
    ratio = 1 / float(legend.split("=")[1].split("$")[0].split("/")[1].lstrip())
    col = axes[3].strip()
    hatch = axes[4].strip()

    lsOx = 1./float(name.split("_")[3])

    X0 = 0.2  # m
    Y = 0.1  # m
    xstar = 2 * 512000 * np.pi * (1.72633e-06 / 2) ** 2 / (
            X0 * Y)  # correctif pour passer du GP au gaz réel : il s'agit du double de la compacité

    for l in f:
        ds = l.split(",")
        x,r, y, ux,ur, uy = float(ds[0]),float(ds[1]), float(ds[2]),float(ds[4]), float(ds[6]), float(ds[7])  # X , Ec
        rs.append(r)
        urs.append(ur)
        xs.append(x)
        Ecs.append(y)
        uEcs.append(uy)
        EcMs.append(float(ds[3]))
        uEcMs.append(float(ds[8]))
        EcRevs.append(Ecs[0] * (1-xstar) / (x-xstar))  # adiabatic rev 2D compression

    xs = np.array(xs)
    rs = np.array(rs)
    urs = np.array(urs)
    EcRevs = np.array(EcRevs)
    Ecs = np.array(Ecs)
    uys = np.array(uEcs)
    EcMs = np.array(EcMs) / Ecs[0] /ratio**2
    uEcMs = np.array(uEcMs) / Ecs[0]/ratio**2

    ys = Ecs - EcRevs
    ys /= Ecs[0]
    uys /= Ecs[0]

    gamma = ratio
    xsmodel, us = solve_de.getModel(gamma, lsOx)
    xsmodela, usa = solve_de.getModelApprox(gamma, lsOx)

    plot = "r"
    if plot == "r" :
        # r plot
        plt.fill_between(xs, rs - urs / 2, rs + urs / 2, alpha=1, facecolor="none", label=legend, hatch=hatch,
                         edgecolor=col)

        modelrs = 1 + us * (xsmodel-xstar) / (1 - xstar)
        plt.plot(xsmodel, modelrs, "-.", color=col,lw = 3)
        modelrsa = 1 + usa * (xsmodela-xstar) / (1 - xstar)
        plt.plot(xsmodela, modelrsa, "-", color=col,lw=3)
    elif plot=="u":
        # u plot
        plt.fill_between(xs, ys - uys / 2, ys + uys / 2, alpha=1, facecolor="none", label=legend, hatch=hatch, edgecolor=col)
        plt.plot(xsmodel, us, "--", color=col,lw=3)
        plt.plot(xsmodela, usa, "-", color=col, alpha = 0.5,lw=3)
        plt.ylabel("$u = E_{c,{\\rm irr}} / E_{c,0}$")

    else:
        plt.fill_between(xs, EcMs - uEcMs / 2, EcMs + uEcMs / 2, alpha=1, facecolor="none", label=legend, hatch=hatch,
                         edgecolor=col)
        plt.ylabel("$ E_{c,M} / E_{C,0} $")





fig, main_ax = plt.subplots(figsize=(16, 8));
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
plt.xlim(0.49, 1.0)

for i in range(len(files)):
    addPlot(files[i])

plt.legend(loc="lower left", fontsize="16")
plt.xlabel("$x=X(t) / X_0$")
#plt.ylabel("$u = E_{c,i} / E_{c,0}$")
plt.grid()
plt.show()
