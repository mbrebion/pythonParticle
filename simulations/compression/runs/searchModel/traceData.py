import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import solve_de

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)


files = ["256_32_20_25_0.05_0.1_PRA.txt","256_32_20_25_0.1_0.1_PRA.txt","256_32_20_25_0.2_0.1_PRA.txt","256_32_20_25_0.4_0.1_PRA.txt","256_32_20_25_0.8_0.1_PRA.txt"]

def addPlot(name):
    f = open(name, "r")
    xs = []
    rs = []
    urs = []
    axes = f.readline().split("\n")[0].split(",")
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = "X0 = " + name.split("_")[4]
    col = axes[3].strip()
    hatch = axes[4].strip()

    for l in f:
        ds = l.split(",")
        x, r, y, ux, ur, uy = float(ds[0]), float(ds[1]), float(ds[2]), float(ds[4]), float(ds[6]), float(
            ds[7])  # X , Ec
        rs.append(r)
        urs.append(ur)
        xs.append(x)

    xs = np.array(xs)
    rs = np.array(rs)
    urs = np.array(urs)

    plt.fill_between(xs, rs - urs / 2, rs + urs / 2, alpha=1, facecolor="none", label=legend, hatch=hatch,
                     edgecolor=col)


fig, main_ax = plt.subplots(figsize=(16, 8));
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
plt.xlim(0.49, 1.0)

for i in range(len(files)):
    addPlot(files[i])

plt.legend(loc="lower left", fontsize="16")
plt.grid()
plt.show()
