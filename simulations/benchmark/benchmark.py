import time
import math

from numbaAcc import measuring
from domain import Domain

X = .1
Y = .1
nPartTarget = 4000
T = 300
P = 1e5
ls = 5e-4


# * * * *
nc = 2
# * * * *


domain = Domain( nc, T, X, Y, P, nPartTarget, ls, drOverLs=0.01,maxWorkers=2)
nStepToTau = 1 / domain.csts["drOverLs"]  # number of it to reach mean free time (not rounded)
nPart = domain.csts["nbPartCreated"]
it = 0

# warm up
print("start warming up")
while it < 500:
    it += 1
    domain.update()

domain.setMaxWorkers(2)

domain.resetCollisions()
domain.resetTimes()

print("start sleeping")
time.sleep(5)
print("start recording")
# main loop
it = 0
t = time.perf_counter()
pressure = 0.
#while time.perf_counter() - t < 50:
while it < 200000:
    it += 1
    if   (it+1900) % 2000 ==0:
        eta = nPart * (domain.csts["ds"]/2)**2 * math.pi / (X*Y)
        p, T = pressure / it , domain.computeParam(measuring.computeTemperature,extensive=False)
        z_meas = p * X*Y / (nPart * domain.csts["kbs"]*T)
        inCol,betweenCol = domain.countCollisions()
        print(it, "/2000")
        print(p,T,nPart)
        print("Z = , ", z_meas)
        print("Z_corr = , ", z_meas*T/300)
        print("colls/expect : ", round(( (inCol+betweenCol)* nStepToTau / it) / (nPart / 2), 5), ) # should stay close to 1
        print()
    domain.update()
    pressure += domain.computePressure()
    #time.sleep(0.25)
total = time.perf_counter() - t   #- 0.25*it

# debriefing

eta = nPart * (domain.csts["ds"]/2)**2 * math.pi / (X*Y)
z_hel = 1/(1-eta)**2
alpha = 0.1280
z_hen = (1+alpha*eta**2)/(1-eta)**2


p, T = pressure / it , domain.computeParam(measuring.computeTemperature,extensive=False)

z_meas = p * X*Y / (nPart * domain.csts["kbs"]*T)

# prints linked with exactness of the simulation
print("Z_meas, helfand, henderson : " , round(z_meas,6), round(z_hel,6), round(z_hen,6) )
inCol,betweenCol = domain.countCollisions()
print("colls/expect : ", round(( (inCol+betweenCol)* nStepToTau / it) / (nPart / 2), 5), ) # should stay close to 1
print("interf col  : ", round(100* betweenCol / (inCol + betweenCol),3), " %"  )
print("* * * * *")
# prints linked with timing
tpt = total / it * nStepToTau
print("time per tau = " , round(tpt,5 ) , "s") # computational time to simulate one mean free time
print("total time =  ", round(total,5 ) , "s")
print("total it =  ", it)
print("* * * * *")
# prints linked repartition of computational efforts (in percent)
print("collide & Sort     : ", round(100*domain.collNSortTime / it * nStepToTau / tpt, 5), "%")
print("inter Coll/swap    : ", round(100 * domain.interfacesTime / it * nStepToTau / tpt, 5), "%")
print("Collide & sort     time: ", round(domain.collNSortTime / it * nStepToTau , 5), "s")
print("inter Coll/swap    time: ", round(domain.interfacesTime / it * nStepToTau, 5), "s")
