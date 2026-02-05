import math
import time

from domain import Domain
from constants import ComputedConstants

X = 0.1
Y = 0.1
nPart = 1024000
T = 300
P = 1e5

#eta = 0.05
#ds = math.sqrt(eta * X * Y / nPart / math.pi) * 2
#ComputedConstants.thermodynamicSetupFixedDiameter(T, X, Y, P, nPart, ds)

ls = 1e-3
ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)

domain = Domain(256)
domain.setMaxWorkers(4)

it = 0
n = 16
while it < n//4:
    it += 1
    domain.update()

time.sleep(0.5)

it = 0
t = time.perf_counter()
while it < n:
    it += 1
    domain.update()
total = time.perf_counter() - t

print("time per it = " , round(total/n*1e3,3 ) , "ms")
print("ratio = " , round(total/n*1e3,3 ) / 1209 )
# ref count = 1209 ms

