import math

from domain import Domain
import numpy as np

X = 0.1
Y = 0.1
nPart = 4000
T = 300
P = 1e5
eta = 0.003
ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2
# we use the way ds is computed from ls to impose ds, without changing the code !
Z = (1.0 + (eta ** 2) / 8.0) / (1 - eta) ** 2
ls = (np.pi * ds) / (4*np.sqrt(2) * (Z - 1.0))
d = 1/4
c = 1/15

dol = 0.005

print("N_t = ", round((d/c)**2 * X / dol / ls))
print("ls/X = ", round(ls/X,5))
nc = 4
print("z_th = ", 1 / (1 - eta) ** 2)
alpha = 0.1280
print("z_th,2 = ", (1 + alpha * eta ** 2) / (1 - eta) ** 2)



def run(nb):
    domain = Domain( nc, T, X, Y, P, nPart, ls, drOverLs=dol,maxWorkers=1, hideStartupOutput=nb != 0)
    ps = []
    ntau = 1500
    while domain.csts["time"] < ntau/5*domain.csts["tau"]:
        domain.update()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    domain.resetCollisions()
    domain.resetTimes()
    it = 0

    while domain.csts["time"] < ntau*domain.csts["tau"]  :
        domain.update()
        ps.append(domain.computePressure())
        it +=1


    inCol, betweenCol = domain.countCollisions()
    percentColl = (inCol + betweenCol) / domain.csts["time"] *domain.csts["tau"] / (nPart/2) * 100
    p = np.average(ps)

    z = p * X*Y / (nPart * domain.csts["kbs"]*T)

    return z,percentColl*5/4

zs = []
percents=[]
for r in range(50):
    z,percent = run(r)
    zs.append(z)
    percents.append(percent)
    p = np.average(percents)
    up = np.std(percents) / len(percents)**0.5 *2
    z = np.average(zs)
    uz = np.std(zs) / len(zs) ** 0.5 * 2.5
    print("run nb : ", r+1)
    print("percent of Coll : ", round(p,4), "+/-", round(up,4), " %")
    print("              Z : ", round(z,6), "+/-", round(uz,6), " %")


# ici dt est peu limitant car on a déjà d > l puis dOM/d << 1 ; les particules se déplacent peu lors d'un pas de temps

##############################################@@
#eta = 0.3  ; z_th2 = 2.0643
#N = 4000 ; ls/X = 0.0051 ; dt=0.05
#percent of Coll :  97.6968 +/- 0.0354  %
#              Z :  2.041763 +/- 0.001807  %

#N = 4000 ; ls/X = 0.0051 ; dt=0.02
#percent of Coll :  97.50 +/- 0.0188  %
#              Z :  2.03806 +/- 0.0014  %

#N = 4000 ; ls/X = 0.0051 ; dt=0.005
#percent of Coll :  97.397 +/- 0.088  %
#              Z :  2.036538 +/- 0.001034  %

#N = 16000 ; ls/X = 0.00255 ; dt=0.05
#percent of Coll :  99.102 +/- 0.0153  %
#              Z :  2.057039 +/- 0.00089  %

#N = 16000 ; ls/X = 0.00255 ; dt=0.02
#percent of Coll :  98.812 +/- 0.031  %
#              Z :  2.052006 +/- 0.001783  %

#N = 16000 ; ls/X = 0.00255 ; dt=0.002
#percent of Coll :  98.66 +/- 0.0247  %
#              Z :  2.049266 +/- 0.000908  %


#N = 64000 ; ls/X = 0.00128 ; dt=0.05
#percent of Coll :  99.634 +/- 0.0099  %
#              Z :  2.064624 +/- 0.000944  %

#N = 64000 ; ls/X = 0.00128 ; dt=0.02
#percent of Coll :  99.428 +/- 0.0061  %
#              Z :  2.058911 +/- 0.001676  %

#N = 64000 ; ls/X = 0.00128 ; dt=0.005
#percent of Coll :  99.357 +/- 0.0224  %
#              Z :  2.057591 +/- 0.00244  %

#N = 256000 ; ls/X = 0.00064 ; dt=0.05 ; nc = 64
#percent of Coll :  99.9706 +/- 0.0088  %
#              Z :  2.068785 +/- 0.001901  %

#N = 256000 ; ls/X = 0.00064 ; dt=0.02 ; nc = 64
#percent of Coll :  99.788  +/- 0.002  %
#              Z :  2.066251 +/- 0.001331  %

#N = 256000 ; ls/X = 0.00064 ; dt=0.02 ; nc = 8
# percent of Coll :  99.798 +/- 0.0063  %
#              Z :  2.066663 +/- 0.00117  %


##############################################
#eta = 0.03  ; z_th2 = 1.062923
#N = 4000 ; ls/X = 0.027 ; dt=0.02
#percent of Coll :  99.611 +/- 0.01  %
#              Z :  1.062709 +/- 0.000338  %

#N = 4000 ; ls/X = 0.027 ; dt=0.005
#percent of Coll :  99.57 +/- 0.0206  %
#              Z :  1.062672 +/- 0.000413  %


#N = 16000 ; ls/X = 0.013 ; dt=0.02
#percent of Coll :  99.876 +/- 0.0153  %
#              Z :  1.062734 +/- 0.000763  %


#N = 64000 ; ls/X = 0.0068 ; dt=0.02
#percent of Coll :  99.965 +/- 0.0105  %
#              Z :  1.062972 +/- 0.00059  %

#N = 64000 ; ls/X = 0.0068 ; dt=0.005
#percent of Coll :  99.906 +/- 0.0071  %
#              Z :  1.062899 +/- 0.000201  %

#N = 256000 ; ls/X = 0.0034 ; dt=0.02
# percent of Coll :  100.0101 +/- 0.0039  %
#              Z :  1.062978 +/- 0.000167  %

#N = 256000 ; ls/X = 0.0034 ; dt=0.005
#percent of Coll :  99.9651 +/- 0.0044  %
#              Z :  1.062956 +/- 0.00034  %


##############################################
#eta = 0.003  ; z_th2 = 1.0060283
#N = 4000 ; ls/X = 0.09 ; dt=0.02
#percent of Coll :  98.9932 +/- 0.019  %
#              Z :  1.005743 +/- 0.000304  %

#N = 4000 ; ls/X = 0.09 ; dt=0.005
#percent of Coll :  99.8392 +/- 0.0201  %
#              Z :  1.005832 +/- 0.000393  %

#N=256000 ; dt = 0.02
#percent of Coll :  99.176 +/- 0.003  %
#              Z :  1.005695 +/- 0.000255  %

#N=256000 ; dt = 0.005
#percent of Coll :  99.978 +/- 0.0043  %
#              Z :  1.006029 +/- 0.000207  %