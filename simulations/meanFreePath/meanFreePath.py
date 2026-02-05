import matplotlib.pyplot as plt

from constants import ComputedConstants
from domain import Domain
import numpy as np


X = 0.1
Y = 0.1
lsa = 0.005
nPart = 64000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, lsa)

domain = Domain(32)
domain.setMaxWorkers(1)
ComputedConstants.dt /= 1


it = 0
imax = 25000
while it < imax:
    it += 1
    if (it*10) % imax == 0:
        print(it)
    domain.update()


ls =[]
i = 0
while ComputedConstants.lls[i] != 0.:
    if ComputedConstants.lls[i]<0.1 and ComputedConstants.lls[i]>0:
        ls.append(ComputedConstants.lls[i])
    i+=1
ls = np.array(ls)

print()

print()
print(len(ls), " events occured")
print( "ls = " , round(np.average(ls)*1000,3), "+/-" ,  round(np.std(ls)*1000/len(ls)**0.5,3), "mm")
import matplotlib.pyplot
plt.hist(ls,"rice")
plt.show()
