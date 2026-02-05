import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import thermo
import constants
from scipy.optimize import curve_fit
font = {'family': 'times',
        'size': 20}

matplotlib.rc('font', **font)

#common data
H = 0.1 #m

def model(y,vmax):
    return 4*vmax/(H)**2* y * ((H)-y)

def eta0(l,T,Ns):
    """
    l: mean free path
    T: temperature
    Ns: Nb of particles
    :return:viscosity provided by boltzmann theory
    """
    sigma = H**2 / (2*2**0.5 * l * Ns)
    nu = Ns * np.pi * (sigma/2)**2 / H**2

    kbs = thermo.getKbSimu(1e5,H**2,T,Ns)
    ms = thermo.getMSimu(constants.MASS,constants.Kb,kbs)
    g2 = (1-7*nu)/(1-nu)**2
    etaB = 1/(2*sigma) * (ms*kbs*T/np.pi)**0.5
    return etaB #* (1/g2 + 2*nu + (1+8/np.pi) * g2 * nu**2)

def getEta(vmax,T):
    rhog = constants.MASS * 1e5 / (constants.Kb*T) * 9.81*50
    return rhog * (2*H) ** 2 / (8 * vmax)


"""
l=0.125e-4 m : zero profile
"""

T0 = np.average([313,313.3])
l0 = 0.125e-3 #m
vxs0 = []


#2024
vxs0.append(np.array([6.68,19.33,30.88,41.33,49.7,57.77,64.73,71.11,75.15,78.58,80.84,81.91]))
#4096
vxs0.append(np.array([7.52,19.93,31.08,41.55,50.72,57.98,64.77,70.06,76.0,78.75,80.95,81.58]))

"""
l=5e-4 m : First profile
"""
#T1 = np.average([319,315,313])
T1 = np.average([312.6,315,313])
l1 = 5e-4 #m
vxs1 = []

# 1024 k
#vxs1.append(np.array([1.87,5.06,8.14,10.55,12.92,14.97,16.35,17.39,18.25,19.35,20.45,21.0]))
#vxs1.append(np.array([2.29,5.95,8.39,10.41,12.88,14.5,16.12,17.9,18.81,19.42,20.39,21.1]))

#2024
#vxs1.append(np.array([2.05,5.13,7.79,10.06,12.18,13.99,15.54,17.38,18.9,19.44,19.61,19.97]))
#4096
vxs1.append(np.array([1.83,4.73,7.51,10.3,12.6,14.65,16.38,17.66,18.67,19.52,19.98,20.03]))
#8M
vxs1.append(np.array([1.82,4.93,7.83,10.51,12.62,14.66,16.08,17.75,18.94,19.95,20.32,20.63]))


"""
l=2e-3 m : First profile
"""
T2 = np.average([311,311,313,313])
l2 = 2e-3 #m
vxs2 = []

#  512 k
#vxs2.append(np.array([0.54,1.59,1.82,3.11,2.72,3.41,3.69,3.21,3.62,3.61,3.61,3.79]))
# 1024 k
#vxs2.append(np.array([0.63,1.41,2.1,2.77,3.36,3.82,4.22,4.52,4.74,4.92,5.09,5.2]))
# 2048 k
#vxs2.append(np.array([0.59,1.47,2.24,2.81,3.39,3.96,4.41,4.76,4.79,4.77,5.11,5.1]))
# 4048 k
vxs2.append(np.array([0.56,1.53,2.26,2.76,3.26,3.62,4.26,4.78,4.84,5.14,5.14,5.06]))
#8M
vxs2.append(np.array([0.66,1.25,1.97,2.67,3.17,3.55,3.85,4.41,4.71,4.86,4.98,5.06]))


fig = plt.figure(figsize=(9, 5), dpi=120)
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')

def addProfileToPlot(vxs,T,l,color):
    vx = np.array([0.]*len(vxs[0]))
    uvx = np.array([0.] * len(vxs[0]))
    nb = 0
    nbv = len(vx)
    ys = np.array([H*(i+0.5) / nbv/2 for i in range(nbv)])

    vmaxs = []
    for vs in vxs:
        popt, pcov = curve_fit(model, ys, vs, p0=[20.])
        vmaxs.append(popt[0])
        vx+= vs
        uvx += vs**2
        nb+=1
    vx/=nb
    uvx = np.sqrt(uvx /nb - vx**2) / np.sqrt(nb) # uncertainty

    popt, pcov = curve_fit(model, ys, vx, p0=[53.])
    vmax = popt[0]  # vmax pour la moyenne

    plt.plot(model(ys, vmax), ys / H, color, linewidth=2,label="$l_s$="+str(round(l*1e3,4))+" mm")
    plt.errorbar(vx, ys/H,xerr=uvx, color="r", linestyle='',capsize=4, elinewidth=2)



    Ns=48e3 # Ns usefull only if advanced model is used to compute the viscosity
    etas = [getEta(v,T) for v in vmaxs]
    print("for ls = ", l*1e3 , " mm : ")
    print("vmax = " , np.average(vmaxs), "m/s")
    print("eta sim/th", str(round(np.average(etas),6))+" +/-"+ str(round(np.std(etas)/np.sqrt(len(etas)),6)), round(eta0(l, T, Ns),6))


addProfileToPlot(vxs0,T0,l0,"-b")
addProfileToPlot(vxs1,T1,l1,"--g")
addProfileToPlot(vxs2,T2,l2,"-.k")
#addProfileToPlot(vxs3,T3,l3,"r")
#addProfileToPlot(vxs4,T4,l4,"y")

plt.grid()
plt.grid(visible=True, which='minor', color='grey', linestyle='--')
plt.legend(loc="upper left")
plt.yticks([0.05,0.1, 0.5],[5e-2, 1e-1, 5e-1])
plt.ylabel("$y/H$ ")
plt.xlabel("$v(y)$ [m/s]")
plt.tight_layout()
plt.show()