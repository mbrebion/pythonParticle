import math
import time

import constants
from domain import Domain
from constants import ComputedConstants

X = 2
Y = 1
scale = 1
nPart = 64000
T = 300
P = 1e5

ls = 2e-2
ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

nc = 128
domain = Domain(nc)
domain.setMaxWorkers(1)

it = 0
n = round(1/constants.drOverLs)


while it < 2:
    it += 1
    domain.update()

domain.resetCountCollisions()
domain.resetTimes()

it = 0
t = time.perf_counter()
while time.perf_counter() - t < 10.:
    it += 1
    domain.update()
total = time.perf_counter() - t

print("colls/expect : ", (domain.totalCollisions *n /it) / (nPart//2))
print("time per tau = " , round(total/it*1e3*n,3 ) , "ms")
print("total time =  ", round(total*1e3,3 ) , "ms")
print("total it =  ", it)
print("")
print("collNSort time : ", round(domain.collNSortTime,4)/it, "s")
print("interfaceColl time : ", round(domain.interfaceCollTime,4)/it, "s")
print("swap time : ", round(domain.swapTime,4)/it, "s")
print("")
print("collNSort/nbCheck * 1e9 : " , domain.collNSortTime / domain.totalCollisions * 1e9)
print("ratio : " ,  domain.totalCollisions  / (nPart**2/nc) /it *1000)



#N = 512k ; ls = 1e-2
#best nc                                  |
#dom/l          0.02    0.02    0.02    0.02    0.02   0.01
#nc              32      64     128      256     512   1024
# Delta_t (s)    4.2    2.87    2.0      1.7     1.8    3.9


#N = 64k ; ls = 2e-2
#best nc                                  |
#dom/l          0.02    0.02    0.02     0.02    0.02   0.02
#nc               4       8      16       32      64    128
# Delta_t (s)    0.73   0.53    0.38     0.26    0.22   0.24




################################################

#N = 8M ; ls = 2e-2
#-> if dom/l is adapted, nc is capped by swapping                               |
#dom/l          0.02       0.02     0.02    0.02    0.02    0.02      0.01    0.01     0.005
#nc             16         32       64     128      256     512       512     1024      2048
# Delta_t (s)                                       213     131       147      107       115
# C&S (s)      21.7      12.1       7.6     5.4     3.66    2.45     2.66      1.8       1.6
# iC   (s)      0.005     0.011     0.018   0.023   0.038   0.047    0.044     0.064     0.136
# swap (s)      0.073     0.081     0.09    0.11    0.12    0.12     0.22      0.25      0.54


#N = 8M ; ls = 2e-3
#-> nc is capped by swapping                                           |
#nc             16         32       64       128      256     512     1024   2048    4096
# Delta_t (s)                                        41.6    27.8    23.8   24      29.7
# C&S (s)      3.56       2.43      1.66     1.21     0.74    0.45    0.36   0.33    0.34
# iC  (s)      0.0024     0.0023    0.0038   0.0067   0.011   0.017   0.021  0.033   0.059
# swap(s)      0.062      0.06      0.066    0.071    0.075   0.084   0.097  0.12   0.16
# C&S / nbcheck                      7.7      12       15      19     32
# ratio        0.21       0.21      0.21     0.198    0.19    0.18   0.169  0.154    0.138

#N = 8M ; ls = 1e-3
# best nc, capped by swapping     |
#nc            256      512     1024    2048      4096
# Delta_t (s)  33       23.8     21.5     23       29
# C&S (s)       0.58     0.38     0.32     0.32     0.35
# iC  (s)       0.014    0.019    0.025    0.043    0.083
# swap(s)       0.071    0.076    0.084    0.104    0.146
