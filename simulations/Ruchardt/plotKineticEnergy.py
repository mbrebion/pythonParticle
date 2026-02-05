import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plotSpectrum = False
plotVar = ""

#names = ["xols12.5p2m2v12N256k.txt","xols12.5p1m2v12N256k.txt","xols25p1m2v12N256k.txt"]
names = ["xols12.5p2m2v12N256k.txt"]
h = 0.1
L = 0.4


plt.xlabel("t [s]")
plt.ylabel("Ec [J]")

plt.grid()

for name in names:
    f = open(name, "r")
    firstline = f.readline()
    color = firstline.split(",")[0].replace(" ", "")
    xols =float(name.split("p")[0].split("xols")[1])
    P = float(name.split("m")[0].split("p")[1])
    m = float(name.split("v")[0].split("m")[1])
    legend = ", ".join(firstline.split(",")[1:])
    ts = []
    xs = []
    vs = []
    ecTots = []
    ecWalls = []
    ecLefts = []
    ecRights = []

    for l in f:
        t, x, v, ecl, ecr = l.split(",")[0:5]
        ts.append(float(t))
        xs.append(float(x) - 0.5)
        vs.append(float(v))
        ecWall = float(v) ** 2 * 0.5 * m
        ecWalls.append(ecWall)
        ec = float(ecl) + float(ecr) + ecWall
        ecLefts.append(float(ecl))
        ecRights.append(float(ecr))
        ecTots.append(ec)


    ecLefts = np.array(ecLefts)
    ecRights = np.array(ecRights)
    xs = np.array(xs)*2
    uLefts = ecLefts/ecLefts[0] - 1/(1+xs)
    uRights = ecRights/ ecRights[0] - 1/ (1 - xs)

    plt.plot(ts, uLefts, "-", color="black", label=legend)
    plt.plot(ts, uRights, "-", color="red", label=legend)

plt.legend()
plt.show()

