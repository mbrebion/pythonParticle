import numpy as np
import matplotlib.pyplot as plt

f = open("data.txt", 'r')
ts = []
vs = []
for l in f :
    t,v = l.split(" ")
    ts.append(float(t))
    vs.append(float(v))

ts = np.array(ts)
vs = np.array(vs)

tAc = 2*0.1 / 340

plt.plot(ts/tAc, vs, "-r")
plt.xlabel("t / t_{acous}")
plt.ylabel("vMax (m/s)")
plt.grid()
plt.show()


amps = np.fft.fft(vs)
freqs = np.fft.fftfreq(len(vs),(ts[1] - ts[0])/tAc)
plt.plot(freqs,abs(amps), "-r")
plt.xlabel("freqs (normalized)")
plt.ylabel("amps")
plt.grid()
plt.show()