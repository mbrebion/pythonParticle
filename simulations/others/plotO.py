import numpy as np
import matplotlib.pyplot as plt

ncs   = np.array([   16,   32,     64,     128,    256,   512,   1024,   2048,   4096])
colls = np.array([ 3.56, 2.43,   1.66,    1.21,   0.74,  0.46,   0.36,   0.33,   0.34])  #s
swaps = np.array([0.062, 0.06,  0.066,   0.071,  0.074,  0.082, 0.097,   0.12,   0.16]) #s

def model(nc):
    alpha = 53
    beta  = 0.34
    return alpha / nc + beta

plt.plot(ncs,colls, "xk")
plt.plot(ncs,swaps-0.06,"ob")
print(swaps-0.06)
plt.plot(ncs,model(ncs), "-r")
plt.grid()
plt.xlabel("N")
plt.ylabel("coll&Sort (s)")
plt.show()