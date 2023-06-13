import matplotlib.pyplot as plt
import numpy as np



plt.grid()
plt.xlabel("x/x0")
plt.ylabel("E_c * x / (E_c *x)_0" )
#plt.ylim(0.8,1.25)
plt.ylim(0.95,1.05)

def fun(ec,x,corr):
    return ec * (x**(1)*0.1 -corr)

def addPlot(name,display,corr):

    f = open(name,"r")
    ts=[]
    xs = []
    Ecs = []
    Ss = []
    for l in f:
        t,x,Ec,Ecr = l.split()
        ts.append(float(t))
        xs.append(float(x))
        Ecs.append(float(Ec))
        Ss.append(1*float(x))
    Ecs = np.array(Ecs)
    xs = np.array(xs)


    init = fun(Ecs[0], xs[0],corr)
    plt.plot((xs*0.1-corr)/(xs[0]*0.1-corr) , fun(Ecs,xs,corr) / init ,display, label = name)

addPlot("dc2mm.txt", "--g",0.000076691*2*4/5)
addPlot("dc2mmLowRes.txt", "-.r",0.000076691*4/5)
#addPlot("dc5mm.txt", "-k")
#addPlot("dc10mm.txt", "--r")
#addPlot("dc10mmBis.txt", "+")
#addPlot("dc20mm.txt", "-.b")
#addPlot("dc20mmLowRes.txt", "*")


plt.legend()
plt.show()