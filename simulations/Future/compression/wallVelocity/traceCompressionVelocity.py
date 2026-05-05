import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)
plt.figure(figsize=(16, 8))
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)

plt.grid()
plt.xlabel("$X(t)/X(0)$")
plt.ylabel("$r$")
plt.ylim(0.999, 1.05)

# ls=Y/25

def model(x, V):
    X0 = 0.9 * 0.4
    taua = X0 / (414 + abs(V))
    t = np.array((x - X0) / V)
    return 1 + (V / 414) ** 2 * (1 - np.exp(2.5 * (X0 - x) / (V * taua)))
    return 1 + (V / 414) ** 2 * np.minimum(t / taua, [1.] * len(t))


def addPlot(name, display,legend = None):
    f = open(name, "r")
    xs = []
    cstes = []
    for l in f:
        x, Ec, cste, cstep, t = l.split()
        xs.append(float(x))
        cstes.append((float(cste)))

    xs = np.array(xs) * 0.9 / 0.8 * 0.4
    xsn = np.array(xs) / xs[0]
    cstes = np.array(cstes)
    plt.plot(xsn, cstes, display, linewidth=2, label=legend)


#addPlot("../runs/vo5Xo25_8_005.txt", ".k")
addPlot("../runs/a.txt", "-k","a")
addPlot("../runs/b.txt", "-b","b")
addPlot("../runs/c.txt", "-r","c")
addPlot("../runs/d.txt", "-g","d")
addPlot("../runs/dPrime.txt", "-y","dPrime")
addPlot("../runs/e.txt", ".y","e")
addPlot("../runs/f.txt", ".r","f")
addPlot("../runs/g.txt", ".b","g")
addPlot("../runs/h.txt", "xk","h")

plt.legend(loc="lower left")
plt.show()
