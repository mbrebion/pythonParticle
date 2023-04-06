import numpy as np

from cell import Cell
import time
from numbaAccelerated import sortCell, isCollidingFast


def run():
    nPart = 50
    Cell.thermodynamicSetup(300, 0.1, 0.1, 1e5, nPart)

    cell = Cell(1, 300)
    tinit = time.time()
    n = 1000
    for i in range(n):
        swaps = cell.update()
        if i % 100 == 0:
            print((i * 100) // n, "%", "average pressure : ", int(cell.averagedPressure), " average temperature",
                  cell.temperature)

    deltat = time.time() - tinit
    print("nb de collision par pas de temps et par particule :", str(cell.nbCollision/nPart/n), " s-1")
    print("durée de simu :", str(deltat)[:5], " s")
    cell.plot()


def testSort():
    ys = np.arange(10, 80, 1)
    xs = np.arange(10, 80, 1)
    vxs = np.arange(10, 80, 1)
    vys = np.arange(10, 80, 1)
    wheres = np.array([1] * len(xs))

    np.random.shuffle(ys)

    print(ys)
    sortCell(xs, ys, vxs, vys, wheres)
    print()
    print(ys)


def testCollide():
    x1, y1 = 1., 0.
    x2, y2 = 0.2, 1.
    vx1, vy1 = 1., 0.
    vx2, vy2 = 0., 1.
    dt = 2
    d = 0.14
    print(isCollidingFast(x1, y1, x2, y2, vx1, vy1, vx2, vy2, dt, d))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
    #testCollide()
    # testSort()
