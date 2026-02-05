import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import solve_de
font = {'family': 'times',
        'size': 26}
matplotlib.rc('font', **font)

def addPlot(gamma,lsOx,col):

    xsmodel, us = solve_de.getModel(gamma, lsOx)
    xsmodela, usa = solve_de.getModelOpposite(-gamma, lsOx)


    plt.plot(xsmodel, us, "--", color=col,lw=3)
    plt.plot(xsmodela, usa, "-", color=col ,lw=3)
    plt.ylabel("$u = E_{c,{\\rm irr}} / E_{c,0}$")



fig, main_ax = plt.subplots(figsize=(16, 8));
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.11)
plt.xlim(0.49, 1.5)


addPlot(0.05,1/25,"blue")

plt.legend(loc="lower left", fontsize="16")
plt.xlabel("$x=X(t) / X_0$")
plt.grid()
plt.show()
