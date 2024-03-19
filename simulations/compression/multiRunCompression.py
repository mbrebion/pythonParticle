import time
from constants import ComputedConstants
from domain import Domain


def printList(l, name):
    out = "["
    for v in l:
        out += str(v) + ","
    out = out[0:-1] + "])"
    print(name, "= np.array(", out)


X = 0.4
Y = 0.4
ls = X/25
nPart = 16000
T = 300
P = 1e5
nWarmUp = 2
nbPts = 20


span = nbPts*ls
vel = 25
tAcous = 0.9*X / 414
tMax = tAcous*3
ts = [0.05,0.1,0.2,0.4,0.7,1,1.3,1.7,2,2.5,3,3.5,4]
tis = [ts[0]/2*tAcous] + [(ts[i] +ts[i-1])/2*tAcous for i in range(1,len(ts))]
nbTime = len(ts)

def velocity(t):
    if t <= ComputedConstants.dt * nWarmUp:
        return 0.
    return -vel

def sum(a,b):
    out = []
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out

def divide(a,n):
    out = []
    for i in range(len(a)):
        out.append(a[i]/n)
    return out

def startRun():
    out = []
    ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
    domain = Domain(16)
    domain.setMaxWorkers(1)

    xInit = 9 * X / 10
    domain.addMovingWall(1000, xInit, vel, imposedVelocity=velocity)

    it = 0
    for n in range(nbTime):
        while it*ComputedConstants.dt < ts[n]*tAcous:
            it += 1
            domain.update()

        bins = domain.ComputeXVelocityBeforeWall(nbPts, span)

        out.append(bins)

    return out

print("tAcous = " , tAcous)
print("instants de mesures", tis)
out = startRun()

count = 1
for i in range(10000):
    nout = startRun()
    time.sleep(2)
    out = sum(out,nout)
    count += 1
    print(i)
    print(divide(out,count))
    print()
    print()


