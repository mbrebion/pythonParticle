import matplotlib.pyplot as plt
import numpy as np

f = open("dataOscillation.txt","r")
ts=[]
xs = []
ecgs = []
ecws = []

for l in f:
    t,x,ecg,ecw = l.split()
    ts.append(float(t)*1000)
    xs.append(float(x)*10)
    ecgs.append(float(ecg)/10000)
    ecws.append(float(ecw)/10000)

ecgs = np.array(ecgs)
ecws = np.array(ecws)
ts = np.array(ts)

recgs = ecgs / max(ecgs)
recws = ecws / max(ecws)

plt.grid()
plt.plot(ts,xs,"r-")
plt.plot(ts,ecws,"k--")
plt.plot(ts,ecgs,"b-.")
plt.plot(ts,ecgs+ecws,"k-")

m = 10 # kg
vstar = 414 # m/s
S = 1 * 1 # m^2
P0 = 1e5 # Pa
L = 2.5 # m
H = 1 # m
gamma = 2
tau = (m*vstar) / (2*S*P0)*12
w0 = (2*P0*H * gamma / L)**0.5
print("T = ", 2*3.14159/w0 * 1000, " ms")
print("tau = " , tau * 1000, " ms")
plt.plot(ts,25+3*np.exp(-ts/tau/1000) ) # amplitude théorique ????
plt.xlabel("t (ms)")
plt.ylabel("x (dm)")


plt.show()