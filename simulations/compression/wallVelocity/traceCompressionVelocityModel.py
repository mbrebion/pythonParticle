import matplotlib.pyplot as plt
import matplotlib
import numpy as np
font = {'family': 'times',
        'size': 26}

matplotlib.rc('font', **font)
plt.figure(figsize=(16,8))
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)

plt.grid()
plt.xlabel("$X(t)/X(0)$")
plt.ylabel("$r$" )
plt.ylim(0.99,1.15)

# ls=Y/25

def model(x,V):
    X0 = 0.9*0.4
    taua = X0 / (414 + abs(V))
    t = np.array((x - X0) / V)
    return 1 + (1*V/414)**2  * (1- np.exp(2.5*(X0 - x)/(V * taua ))) * x/X0
    #return 1 + (V/414)**2 * np.minimum( 2.5*t / taua,[1.]*len(t)) * x/X0

def addPlot(name,display):
    f = open(name,"r")
    ts=[]
    xs = []
    Ecs = []
    cstes = []
    velocity = float( name.split("V")[1].split("X")[0])
    for l in f:
        x,Ec,cste,cstep = l.split()
        #xs.append(abs((float(x)-0.8))/velocity)
        xs.append(float(x))
        cstes.append((float(cste)))

    xs = np.array(xs)*0.9/0.8*0.4
    xsn = np.array(xs)/xs[0]
    cstes = np.array(cstes)
    plt.plot(xsn, cstes, display,markersize=8, label="|V| = " + name.split("/")[-1].split("X")[0][1:] + " m/s")
    plt.plot(xsn, model(xs, -velocity), "-",linewidth=2, color=display[1:])



#addPlot("./V5Xo25.txt", ".k")
addPlot("./V10Xo25.txt", "xy")
addPlot("./V25Xo25.txt", "xr")
addPlot("./V50Xo25.txt", "xb")
addPlot("./V100Xo25.txt", "xk")
addPlot("./V150Xo25.txt", "xg")
plt.legend(loc = "upper right")
plt.show()