import time
from cell import Cell
import thermo
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from simulations.Future.movingWall import MovingWall

Z = 1  # m ; S*Z = V
INITSIZEEXTRARATIO = 1.3  # because of dynamic reshaping of cells, large value is not required anymore
DIAMETER = 0.37e-9  # m ; effective diameter of average air particle
MASS = 4.83e-26  # kg ; mean mass of air particle
Kb = 1.38e-23  # USI ; Boltzmann constant
DEAD = 0
LEFT = 1
RIGHT = 2


class Domain:

    ##############################################################
    ###################        Initializing        ###############
    ##############################################################

    def __init__(self, nbCells, initTemp, length, width, initPressure, nbPartTarget, ls , *, ratios=None, effectiveTemps=None, periodic=False, v_xYVelocityProfile=None,colorRatios = None, drOverLs = 0.02, maxWorkers = 1, hideStartupOutput=False):
        """
        :param nbCells: number of cells used in the simulation
        :param initTemp: mean temperature used in simulation
        :param length: length of domain (between left and right walls)
        :param width: width of domain (between lower and upper walls)
        :param initPressure: mean pressure used in simulation (in Pa)
        (pressure to be obtained with nbPartTarget particles)
        :param nbPartTarget: target number of particle for all cell
        (used to compute simulation values for mass, diameter and boltzmann constant)
        The actual number of particles may then differ in cells due to rounding effects
        :param ls: mean free path required (in m)
        :param ratios: (optional), allows to specify different ratios (average = 1.) of particles in each cells
        :param effectiveTemps: (optional), allows to specify different temperatures in each cells
        :param periodic: (optional, default = False) if True, periodic boundary conditions are used to link both left and right sides of domain.
        :param v_xYVelocityProfile: (optional) v_x(y) mean velocity profile to be imposed at startup
        :param colorRatios: (optional) : list of colorRatios  for the number of particles in cells
        :param drOverLs: (optional, default = 0.02), during a time step, particles are streamed of
        (in average) drOverLs * ls ; big values reduces the number time steps needed to complete a simulation,
        at the cost of accuracy ; while low values help treating collisions in their natural order
        :param maxWorkers (optional, default = 1), max number of threads to be used ; both creation of cells and updates
        may benefit from multithreading, event without the GIL-free version of python.
        """

        self.setConstants(initTemp, length, width, initPressure, nbPartTarget, ls, nbCells, periodic,drOverLs, hideStartupOutput)

        self.v_xYVelocityProfile = v_xYVelocityProfile

        self.ratios = ratios

        if ratios is None:
            self.ratios = [1. / nbCells for _ in range(nbCells)]

        self.effectiveTemps = effectiveTemps
        if effectiveTemps is None:
            self.effectiveTemps = [initTemp for _ in range(nbCells)]

        self.colorRatios = colorRatios
        if colorRatios is None:
            self.colorRatios = [1. for _ in range(nbCells)] # 100% of part in white


        self.initialNbParts = [int(nbPartTarget * self.ratios[i]) for i in range(nbCells)]

        # single/multi thread default
        self.maxWorkers = maxWorkers
        self.myMap = map
        self.pool = None
        self.redIter = range(1, nbCells - 1, 2)
        self.blackIter = range(0, nbCells - 1, 2)
        self.iter = range(0, nbCells, 1)
        self.reversedIter = range(nbCells - 1, -1, -1)
        self.setMaxWorkers(self.maxWorkers)

        # cells creation
        self.cells = [0 for _ in range(nbCells)]
        self.startIndices = [ sum(self.initialNbParts[0:i]) for i in range(nbCells) ]
        self.csts["nbPartCreated"] = sum(self.initialNbParts)

        # loop here to create cells
        list(self.myMap(self.createCell,range(nbCells)))



        # updating neighbors
        for i in range(1, nbCells):
            self.cells[i - 1].rightCell = self.cells[i]

        for i in range(0, nbCells - 1):
            self.cells[i + 1].leftCell = self.cells[i]

        if periodic and nbCells > 1:
            self.cells[-1].rightCell = self.cells[0]
            self.cells[0].leftCell = self.cells[-1]


        self.setPeriodic(periodic)

        # wall
        self.wall = None

        # timing
        self.collNSortTime = 0.
        self.interfacesTime = 0.

    def createCell(self,i):
        length = self.csts["length"]
        nbCells = len(self.cells)

        left = i * length / nbCells
        right = (i + 1) * length / nbCells

        c = Cell(self.initialNbParts[i], self.effectiveTemps[i], left, right, self.startIndices[i] , self.csts,
                 self.csts["nbPartCreated"] // nbCells,v_xYVelocityProfile=self.v_xYVelocityProfile, colorRatio=self.colorRatios[i])

        self.cells[i] = c

    def setForceX(self,forceX):
        self.csts["forceX"] = forceX

    def setPeriodic(self,bool):
        self.csts["periodic"] = bool
        if bool:
            self.redIter = range(-1,len(self.cells)-1,2)
        else:
            self.redIter = range(1, len(self.cells)-1, 2)

    def setConstants(self,initTemp, length, width, initPressure, nbPartTarget, ls,nbCells,periodic,drOverLs = 0.02, hideSartupOutput = False):

        tp = np.dtype(
            [("ls", np.float64), ("width", np.float64), ("length", np.float64), ("initTemp", np.float64),
             ("initPressure", np.float64), ("nbPartTarget", np.int32), ("kbs", np.float64), ("ms", np.float64),
             ("ds", np.float64), ("vStar", np.float64), ("vAv", np.float64), ("dt", np.float64)
             , ("tau", np.float64), ("fillRatio", np.float64),("nbCells", np.int32) ,
             ("periodic", np.bool),("nbPartCreated", np.int32),("it", np.int32),("time", np.float64),
             ("forceX", np.float64),("drOverLs", np.float64),("rpc", np.float64),("xwall",np.float64),
             ("vxwall",np.float64), ("mwall",np.float64), ("isWall", np.bool),
             ("wallForceLeft", np.float64),("wallForceRight", np.float64) ]  )
        csts = np.array(0, dtype=tp)
        csts["ls"] = ls
        csts["width"] = width
        csts["length"] = length
        csts["initTemp"] = initTemp
        csts["initPressure"] = initPressure
        csts["nbPartTarget"] = nbPartTarget
        csts["kbs"] = thermo.getKbSimu(initPressure, width*length*Z, initTemp, nbPartTarget)
        csts["ms"] = thermo.getMSimu(MASS, Kb, csts["kbs"])
        d = thermo.getDiameter(width*length,nbPartTarget, ls)
        csts["ds"] = thermo.getDiameterHenderson(width * length, nbPartTarget, ls)
        csts["vStar"] = thermo.getMeanSquareVelocity(csts["kbs"], csts["ms"], initTemp)
        csts["vAv"] = csts["vStar"] * np.sqrt(np.pi) / 2
        csts["dt"] = thermo.getDtCollision(csts["vAv"], ls, drOverLs)
        csts["tau"] = ls / csts["vAv"]
        csts["fillRatio"] = (csts["ds"] / 2) ** 2 * np.pi * nbPartTarget / (length*width)
        csts["nbCells"] = nbCells
        csts["periodic"] = periodic
        csts["nbPartCreated"] = nbPartTarget
        csts["it"] = 0
        csts["time"] = 0.
        csts["drOverLs"] = drOverLs
        csts["forceX"] = 0.
        csts["rpc"] = 0.  # if set to 1, radius of disks is taken into account when computing interractions with walls
        # else, it should be set to 0.

        # wall settings
        csts["xwall"] = 0.
        csts["vxwall"] = 0.
        csts["mwall"] = 0.
        csts["isWall"] = False
        csts["wallForceLeft"] = 0.
        csts["wallForceRight"] = 0.

        self.csts = csts

        bestnc = min((nbPartTarget * length / width) ** 0.5 / 4, nbPartTarget*(drOverLs*ls/width)**0.5/15)
        bestr = 0.02
        if length / bestnc < 20 * ls * bestr:
            bestr = length / (bestnc * 20 * ls)
        if not hideSartupOutput :
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("total gas mass = ", "{:.2e}".format(csts["ms"] * nbPartTarget), "kg")
            print("diameter = ", "{:.5e}".format(csts["ds"]), "m")
            print("uncorrected diameter = ", "{:.5e}".format(d), "m")
            print("v* = ", "{:.5e}".format(csts["vStar"]), "m/s")
            print("dOM/d = v*dt/d = ", "{:.2e}".format(csts["vStar"] * csts["dt"] / csts["ds"]))
            print("ls : ", "{:.5e}".format(ls), " m")
            print("tau : ", "{:.2e}".format(csts["tau"]), " s")
            print("d/l : ", "{:.2e}".format(csts["ds"] / ls), " ")
            print("dt : ", "{:.5e}".format(csts["dt"]), " s")
            print("fill ratio s_parts/S : ", "{:.5e}".format(csts["fillRatio"]))
            print()
            print("recommanded settings : ")
            print("  -     nbCell = ", round(bestnc))
            print("  -   drOverLs = ", round(bestr, 6))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print()


    def getCst(self,name):
        return self.csts[name][0]

    def setMaxWorkers(self, mw):
        self.maxWorkers = mw

        if mw > 1:
            self.pool = ThreadPoolExecutor(max_workers=mw)
            self.myMap = self.pool.map
        else:
            self.myMap = map

    def addMovingWall(self, m, x, v, imposedVelocity=None):
        self.wall = MovingWall(m, x, v, imposedVelocity)
        for c in self.cells:
            c.updateIndicesAccordingToWall(x)
            c.wall = self.wall

    def removeMovingWall(self):
        self.wall = None
        for c in self.cells:
            c.coords.wheres = np.abs(c.coords.wheres)
            c.wall = None



    def reshapeCells(self):
        """
        This function moves cells boundaries to equilibrate the number of particles (usefull with moving wall)
        :return: None
        """
        return
        L = self.cells[-1].right
        def move():

            ns = []
            nsCum = [0]
            nTot = 0

            delta = (self.cells[0].right - self.cells[0].left)/100
            k =len(self.cells)
            for c in self.cells:
                n = c.count()
                ns.append(n)
                nsCum.append(n + nsCum[-1])
                nTot += n
            nAvg = nTot/k
            nsCum.remove(0)

            xs = []
            ds = []
            for i in range(len(self.cells)):
                xs.append(self.cells[i].left)
                ds.append(self.cells[i].right - self.cells[i].left)
            xs.append(L)

            xps = [0]
            for i in range(len(self.cells)):
                alpha = 3*(1. - ns[i] /nAvg)
                xps.append( xps[-1] + ds[i] + delta * alpha)

            for i in range(len(self.cells)):
                self.cells[i].left = xps[i]
                self.cells[i].right = xps[i+1]
            return ns,xs


    ##############################################################
    ################## Compute thermodynamic      ################
    ##############################################################

    def computeParam(self,func,array = [0.],extensive = True, additionalParam=None):
        """
        :param func: the numba func to by applied for computation
        :param array: the np array to store values for all cells, may be smaller than cell number
        :param extensive: if True, values may be summed, else, averaged
        :param additionalParam: if not None, this variable is sent to func
        :return: param for all domain
        """
        r = len(self.cells)//len(array)

        for i in range(len(array)):
            acc = 0.
            for j in range(i*r,i*r+r):
                if additionalParam is None :
                    acc += func(self.cells[j].crd, self.csts)
                else:

                    acc += func(self.cells[j].crd, self.csts, additionalParam)
            array[i] = acc if extensive else acc/r
        return sum(array) if extensive else sum(array) / len(array)


    def countCollisions(self):
        sumInside = 0
        sumInterf = 0
        for c in self.cells:
            sumInside += c.sumCollide
            sumInterf += c.sumCollideInterface
        return sumInside, sumInterf

    def resetCollisions(self):
        for c in self.cells:
            c.sumCollide = 0
            c.sumCollideInterface = 0

    def computePressure(self):
        p = 0
        for c in self.cells:
            p+= c.pressure
        return p / len(self.cells)

    ##############################################################
    ##################         Update      #######################
    ##############################################################

    def streamMovingWall(self):
        if self.wall is not None:
            self.wall.stream()


    def updateInterface(self,i):
        # update interface between cells i and i+1 :
        # collision between particles of both cells
        # move from cell to swap
        # move from swap to cell

        self.cells[i+1].interfaceCollide()
        self.cells[i].prepareSwapToRight()
        self.cells[i+1].prepareSwapToLeft()
        self.cells[i].applySwapFromRight()
        self.cells[i+1].applySwapFromLeft()

    def updateCell(self,i):
        self.cells[i].update()



    def customMap(self, task,iter):
        n = len(iter)
        parallelTasks = self.maxWorkers*2
        def singleBatch(iter):
            for i in iter:
                task(i)

        futures = []
        for i in range(parallelTasks):
            batchIter = iter[(i*n)//parallelTasks:((i+1) *n)//parallelTasks]
            if self.maxWorkers>1:
                futures.append(self.pool.submit(singleBatch,batchIter))
            else:
                singleBatch(batchIter)
        wait(futures)


    def update(self):

        self.csts["it"] += 1  # to be moved upper once cells are gathered in broader class
        self.csts["time"] += self.csts["dt"]

        self.streamMovingWall()

        t = time.perf_counter()

        self.customMap(self.updateCell, self.iter if self.csts["it"] % 2 == 0 else self.reversedIter)

        t1 = time.perf_counter()
        self.collNSortTime += t1 - t

        # red black ordering
        self.customMap(self.updateInterface, self.redIter)
        self.customMap(self.updateInterface, self.blackIter)

        t2 = time.perf_counter()
        self.interfacesTime += t2 - t1


        if self.csts["periodic"]:
            for c in self.cells:
                c.applyPeriodic()


        if self.csts["it"] % 10 == 0:
            self.reshapeCells()


    def resetTimes(self):
        self.collNSortTime = 0.
        self.interfacesTime = 0.
        #self.csts["time"] = 0.
        self.csts["it"] = 0
