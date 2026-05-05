import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import constants

plotSpectrum = False
plotVar = "v"
maxPeriodPlot = 12

names=[
    "xols12.5p1m2v12N256k.txt",
    "xols12.5p2m2v12N256k.txt",
    "xols25p1m1v12N260k_T300.txt",
    "xols25p1m1v12N260k_T300_long.txt",
    "xols25p1m2v12N256k.txt",
    "xols25p1m2v12N256k_T600K.txt",
    "xols50p1m1v12N260k_T300.txt",
]


#names = ["xols25p1m2v12N256k_T600K.txt","xols12.5p2m2v12N256k.txt","xols12.5p1m2v12N256k.txt","xols25p1m2v12N256k.txt","xols25p1m1v12N1040k_T300.txt","xols25p1m1v12N256k_T300.txt"]
#names = ["xols25p1m1v12N1040k_T300.txt","xols25p1m1v12N66k_T300.txt","xols25p1m1v12N260k_T300.txt"] # mêmes conditions, mais avec N !=
#names = ["xols50p1m1v12N260k_T300.txt"] # faible visco
#names = ["xols25p1m1v12N260k_T300_long.txt", "xols25p1m2v12N260k_T300_verylong.txt","xols25p1m1v6N128k_T300_long.txt"] # runs with long cavities (0.8 et 1.6 m) ; second run has bigger mass to keep same Q
names=["HQ/xols12p1m1v12N260k_T300ruguous.txt","HQ/xols25p1m1v12N260k_T300ruguous.txt","HQ/xols50p1m1v12N260k_T300ruguous.txt","HQ/xols50p1m1v12N520k_T300ruguous.txt","HQ/xols25p1m2v6N260k_T300ruguous.txt"] # parois rugueuses
names=["xols25p1m0.25v6N260k_T300quadhalfYruguous.txt","xols25p1m0.5v6N260k_T300halfYruguous.txt","xols25p1m2v6N260k_T300doubleYruguous.txt","xols25p1m2v6N260k_T300doubleLruguous.txt","HQ/xols12p1m1v12N260k_T300ruguous.txt","HQ/xols25p1m1v12N260k_T300ruguous.txt","HQ/xols50p1m1v12N260k_T300ruguous.txt","HQ/xols50p1m1v12N520k_T300ruguous.txt","HQ/xols25p1m2v6N260k_T300ruguous.txt"]
def model(t,w,a,b,tau,dec):

    return np.exp(-t/tau) * ( a*np.cos(w*(1+dec*t) * t) + b * np.sin(w*(1+dec*t) * t) )

def guesses(m,mgaz,v0,Q,L,v0star):
    w0 = (mgaz/m)**0.5 * v0star / L
    tau = 2*Q/w0 / 4 / 10
    return [ w0, v0, 0,  tau, -0.01 ]

if plotSpectrum:
    plt.xlabel("f [Hz]")
    plt.ylabel("Amp")
else:
    plt.xlabel("t/T")
    plt.ylabel(plotVar)

plt.grid()

for name in names:
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

    S = X * Y  # total surface
    mgaz = P * S / T * constants.MASS / constants.Kb
    v0star = (2 * constants.Kb / constants.MASS * T) ** 0.5
    Qth = (2*m / mgaz) ** 0.5 * xols   # encore un meilleur match avec Qth*1.2
    L = X/2
    ls = L/xols



    legend = "$\\sqrt{2M/m_{\\rm gaz}}$ = " + str(round(np.sqrt(2*m/mgaz),1)) + ", $L/l_{\\rm s}$ = " + str(round(xols,2)) + ", $P$ = " + str(round(P/1e5,2)) +" bar, $T$ = "+ str(round(T,0)) +" K"
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
    ts = np.array(ts)
    print("name : ", name)

    # plotting
    output = 0
    if plotVar == "v":
        output = vs
    elif plotVar == "x/X":
        output = xs
    else:
        output = ecLefts


    gs = guesses(m,mgaz, output[0],Qth, X/2,v0star)
    w0g = gs[0]
    opt, pcov = curve_fit(model, ts, output, gs)

    w, a, b, tau, dec = opt
    period = 2 * np.pi / w
    plt.xlim(0, maxPeriodPlot)
    plt.plot(ts/period, output, "-", color=color, label=legend)
    plt.plot(ts/period,model(ts,w, a, b, tau,dec),"--",color=color)

    Qxp = tau * w / 2
    Qth = (2*m * Y**2 / ( mgaz * ls*L)) ** 0.5
    print("L,Y (m)", L, Y)
    print("sqrt{2M/m_gaz} = ", str( (2*m/mgaz)**0.5 )[:6] )
    print("Q,th = ", str(Qth)[:6] )
    print("Q,xp = ",str(Qxp)[:6])
    print("w0,th = ", str(w0g)[:6], "rad/s")
    print("w0,xp = ", str(w / np.sqrt(1-1/(4*Qxp**2)))[:6], "rad/s")
    print(" - - - - - - - - -")
    print()

plt.legend()
plt.show()

