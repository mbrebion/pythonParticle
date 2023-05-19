import matplotlib.pyplot as plt
import numpy as np

f = open("dataCompression.txt","r")
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

init = Ecs[0] * xs[0]

plt.grid()
plt.xlabel("t [s]")
plt.ylabel("E_c * x" )
plt.plot(ts , Ecs *xs / init ,"-k")
plt.show()