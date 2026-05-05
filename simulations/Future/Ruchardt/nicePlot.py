import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import constants
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

name ="HQ/xols12.5p2m2v12N256k.txt"  # transitoire BF
#name ="xols25p1m1v12N128k_T300ruguous.txt"  # transitoire BF , parois rugueuses
#name ="xols25p1m1v12N540k_T300ruguous.txt"  # transitoire BF , parois rugueuses
#name ="xols25p1m1v-12N128k_T300ruguous_reverse.txt"  # transitoire BF , parois rugueuses ; à l'envers
#name ="HQ/xols25p1m1v0N64k_T300.txt"  # retour à l'équilibre thermo : Q élevé
#name ="LQ/xols25p1m0.002v0N66k_T300.txt"  # retour à l'équilibre thermo : Q faible

font = {'family': 'times',
        'size': 34}

matplotlib.rc('font', **font)

f = open(name, "r")
firstline = f.readline()
# red , m=2, v=12,xols = 12.5, N=256000, T=300, P=1e5, X=0.4, Y=0.1

color = firstline.split(",")[0].replace(" ", "")
m    = float(firstline.split(",")[1].split("=")[1].replace(" ", ""))
v    = float(firstline.split(",")[2].split("=")[1].replace(" ", ""))
xols = float(firstline.split(",")[3].split("=")[1].replace(" ", ""))
N    = float(firstline.split(",")[4].split("=")[1].replace(" ", ""))
T    = float(firstline.split(",")[5].split("=")[1].replace(" ", ""))
P    = float(firstline.split(",")[6].split("=")[1].replace(" ", ""))
X    = float(firstline.split(",")[7].split("=")[1].replace(" ", ""))
Y    = float(firstline.split(",")[8].split("=")[1].replace(" ", ""))


L = X/2
S = X * Y  # total surface
mgaz = P * S / T * constants.MASS / constants.Kb
v0star = (2 * constants.Kb / constants.MASS * T) ** 0.5
Qth = (2*m / mgaz) ** 0.5 * xols   # encore un meilleur match avec Qth*1.2
w0 = (mgaz/m)**0.5 * v0star / L
Vstar = np.sqrt(mgaz/(2*m*N)) * v0star
ls = L/xols
ds = S / (2*np.sqrt(2)* N * ls)
s = np.pi * (ds/2)**2
xss = 2* N  *s / Y
print(Vstar,v0star,xss)

legend = "$\\sqrt{2M/m_{\\rm gaz}}$ = " + str(round(np.sqrt(2*m/mgaz),1)).replace(".",",") + "; $L/l_{\\rm s}$ = " + str(round(xols,2)).replace(".",",")
ts = []
xs = []
vs = []
ecTots = []
ecLefts = []
ecRights = []

for l in f:
    t, x, v, ecl, ecr = l.split(",")[0:5]
    ts.append(float(t))
    xs.append(float(x) - 0.5)
    vs.append(float(v))
    ecWall = float(v) ** 2 * 0.5 * m
    ec = float(ecl) + float(ecr) + ecWall
    ecLefts.append(float(ecl))
    ecRights.append(float(ecr))
    ecTots.append(ec)

# std : thermodynamic equilibrium :
vs = np.array(vs)
xs = np.array(xs)*2
ts = np.array(ts)
ecLefts = np.array(ecLefts)
ecRights = np.array(ecRights)
k = m * w0**2
ecMeca = 0.5* m * vs**2 + 0.5*k *(xs*L)**2

xss = 0
ecLeftIrrs = ecLefts - ecLefts[0]*(1-xss)/(1+xs - xss)
ecRightIrrs = ecRights - ecRights[0]*(1-xss)/(1-xs - xss)

print("name : ", name)
print("w0" , w0, "rad/s")


period = 2 * np.pi / w0
#plt.xlim(0, 12)
n = len(vs)
print(np.std(vs[1*n//4:]))

fig, ax = plt.subplots(figsize=(16, 8))
plt.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.13)
#sub_ax = inset_axes(ax, "35%", "35%", "upper right", borderpad=0.8)


ax.grid()
#ax.plot(ts[0*n//6:]/period, vs[0*n//6:]/Vstar, "-", color="red", label=legend)

# EC total
#ax.plot(ts[0*n//6:]/period, ecLefts/ecLefts[0], "-", color="red", label=legend)
#ax.plot(ts[0*n//6:]/period, ecRights/ecRights[0], "-", color="blue", label=legend)

em = ecMeca / ecMeca[0]
eil = ecLeftIrrs
eir = ecRightIrrs

eil = (ecLefts/ecLefts[0] * (1+xs) - 1)*ecLefts[0] / ecMeca[0]
eir = (ecRights/ecRights[0] * (1-xs) - 1)*ecLefts[0] / ecMeca[0]

# EC irr = Ec - Ec,rev
ax.fill_between(ts/period, eil* 0, eil, color="gray",hatch="X",facecolor="none", label="$\epsilon_g \\times E_{c,0}$")
ax.fill_between(ts/period, eil, eil+eir , color="red",hatch="+",facecolor="none", label="$\epsilon_d \\times E_{c,0}$")
ax.fill_between(ts/period, eil+eir, eil+eir + em ,  color="blue",hatch="//",facecolor="none", label="$E_{\\rm m}$")
plt.plot(ts/period, 0.5 * (1-np.exp(-w0/(101) * ts )) , "-k", label="modèle")
ax.legend(loc="lower right")

print(w0,)
ug = ecLeftIrrs/ecLefts[0]
ud = ecRightIrrs/ecRights[0]


#plt.plot(ts/period,beta,"-r",label="$\\beta$")
#plt.plot(ts/period,-4*xs/100,"-b" , label="$-4x/100$")
#plt.plot(ts/period,eg,"-r")
#plt.plot(ts/period,ed,"-k")
#plt.legend()
#ax.set_ylabel("$u_g, \, u_d $")

#ax.plot(ts/period, xs, "-", color="red", label=legend)
ax.set_xlabel("$t \\times f $")
#ax.set_ylabel("$V/V^*$ ")
ax.set_ylabel("$E / E_{ m,0}$ ")
#ax.set_xlim(200,215)
#ax.set_ylim(-2.5,4)

p=0
#sub_ax.plot(xs[p*n//6:], vs[p*n//6:]/Vstar, "-", color="black", label=legend, alpha=1., markeredgecolor='none')
#sub_ax.set_xlabel("$x $")
#sub_ax.set_ylabel("$V/V^*$ ")

#sub_ax.xaxis.set_label_position('top')
#sub_ax.yaxis.set_label_position('left')
#sub_ax.set_xticks([])
#sub_ax.set_yticks([])

plt.show()

