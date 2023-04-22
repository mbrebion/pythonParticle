from constants import ComputedConstants
from domain import Domain


X = 0.1
Y = 0.1
ls = 5e-3

nPart = 2000
T = 300
P = 1e5

ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
domain = Domain(1)

it = 0
while it < 50e3:
    it += 1
    domain.update()

print(domain.cells[0].printHisto())