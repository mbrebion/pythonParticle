import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import constants
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


names =["LQ/xols25p1m0.002v50N520k_T300_"+l+".txt" for l in ["a","b","c","d","e","f", "g", "h", "i", "j"] ]

font = {'family': 'times',
        'size': 36}
matplotlib.rc('font', **font)

f = open(names[0], "r")
firstline = f.readline()
# red , m=2, v=12,xols = 12.5, N=256000, T=300, P=1e5, X=0.4, Y=0.1

color = firstline.split(",")[0].replace(" ", "")
m = float(firstline.split(",")[1].split("=")[1].replace(" ", ""))
v = float(firstline.split(",")[2].split("=")[1].replace(" ", ""))
xols = float(firstline.split(",")[3].split("=")[1].replace(" ", ""))
N = float(firstline.split(",")[4].split("=")[1].replace(" ", ""))
T = float(firstline.split(",")[5].split("=")[1].replace(" ", ""))
P = float(firstline.split(",")[6].split("=")[1].replace(" ", ""))
X = float(firstline.split(",")[7].split("=")[1].replace(" ", ""))
Y = float(firstline.split(",")[8].split("=")[1].replace(" ", ""))

L = X / 2
S = X * Y  # total surface
mgaz = P * S / T * constants.MASS / constants.Kb
v0star = (2 * constants.Kb / constants.MASS * T) ** 0.5
Qth = (2 * m / mgaz) ** 0.5 * xols  # encore un meilleur match avec Qth*1.2
w0 = (mgaz / m) ** 0.5 * v0star / L
Vstar = np.sqrt(mgaz / (2 * m * N)) * v0star
print(Vstar, v0star)

legend = "$\\sqrt{2M/m_{\\rm gaz}}$ = " + str(round(np.sqrt(2 * m / mgaz), 1)).replace(".",
                                                                                       ",") + "; $L/l_{\\rm s}$ = " + str(
    round(xols, 2)).replace(".", ",")

f.close()

ts = []
lesVitesses = []
lesXs = []
for name in names:
    f = open(name, "r")
    firstline = f.readline()
    ts = []
    xs = []
    vs = []
    for l in f:
        t, x, v, ecl, ecr = l.split(",")[0:5]
        ts.append(float(t))
        xs.append(float(x) - 0.5)
        vs.append(float(v))

    # std : thermodynamic equilibrium :
    lesVitesses.append(np.array(vs))
    lesXs.append(np.array(xs))
    ts = np.array(ts)

lesVitesses = np.array(lesVitesses).transpose()
lesXs = np.array(lesXs).transpose()

xs = []
uxs = []
vs = []
uvs = []
for p in range(len(lesVitesses)):
    xs.append(np.average(lesXs[p]))
    uxs.append(np.std(lesXs[p]))
    vs.append(np.average(lesVitesses[p]))
    uvs.append(np.std(lesVitesses[p]))


xs=np.array(xs)
vs = np.array(vs)
uxs = np.array(uxs) #/ len(names)**0.5
uvs = np.array(uvs) #/ len(names)**0.5

period = 2 * np.pi / w0
#plt.xlim(0, 12)
n = len(vs)
print(np.std(vs[1*n//4:]))

fig, ax = plt.subplots(figsize=(16, 8))
plt.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.14)
sub_ax = inset_axes(ax, "35%", "35%", "upper right", borderpad=0.8)

period = (L/414*(2/3)**0.5)

ax.grid()
ax.fill_between(ts/period, vs-uvs/2, vs+uvs/2, alpha=1, facecolor="red", label = "")
ax.plot(ts/period, vs, color="red", label = "",lw=0.6)
#ax.fill_between(ts/period, xs-uxs/2, xs+uxs/2, alpha=1, facecolor="red", label = "")
#ax.plot(ts/period, xs, color="red", label = "",lw=0.3)
ax.set_xlabel("$t / \\tau_{\\rm a} $")
ax.set_ylabel("$V/V^*$ ")

p=0
sub_ax.plot(xs[p*n//6:], vs[p*n//6:]/Vstar, "-", color="black", label=legend, alpha=1., markeredgecolor='none')
sub_ax.set_xlabel("$x $")
sub_ax.set_ylabel("$V/V^*$ ")
sub_ax.axhline(y=0, color='grey')
sub_ax.axvline(x=0, color='grey')

sub_ax.xaxis.set_label_position('top')
sub_ax.yaxis.set_label_position('left')
sub_ax.set_xticks([])
sub_ax.set_yticks([])

plt.show()

