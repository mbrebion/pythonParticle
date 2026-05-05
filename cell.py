from numbaAcc import colliding,measuring,swapping,helping,sorting
import numpy as np
from thermo import getMeanSquareVelocity



class Cell:
    coloringPolicy = "none"  # might be "coll", "vx" or "fixed"
    collision = True

    def __init__(self, nbPart, effectiveTemp, left, right, startIndex,csts, nbPartTarget = None,
                 v_xYVelocityProfile = None, colorRatio = 1., boundaryTempUp = -1, boundaryTempDown = -1):
        """
        Create a cell
        :param nbPart: effective number of part in cell
        :param effectiveTemp: temperature required for this cell (It may be different from Cell.initTemp)
        :param left: left coordinate of cell
        :param right: right coordinate of cell
        :param startIndex: first id of particle lying in this cell
        :param csts: structured numpy array containing all physical constants
        :param nbPartTarget: target numer of particles. Useful if initial number of particle is way lower than the target
        :param v_xYVelocityProfile: v_x(y) mean velocity profile to be imposed at startup
        :param colorRatio : ratio of particles to colorize in white (1)
        """
        self.boundaryTempUp = boundaryTempUp
        self.boundaryTempDown = boundaryTempDown
        self.sumCollide = 0 # cumulative sum of all collisions within this cell
        self.sumCollideInterface = 0 # cumulative sum of all collisions with previous cell (at interface)


        self.left = left
        self.right = right
        self.length = self.right - self.left
        self.startIndex = startIndex
        self.csts = csts
        self.pressure = 0.
        self.mx = 0.

        if v_xYVelocityProfile is not None:
            self.v_xYVelocityProfile = v_xYVelocityProfile
        else:
            self.v_xYVelocityProfile = self.zeroV_xYVelocityProfile

        self.nbPart = nbPart
        if nbPartTarget is None:
            nbPartTarget = self.nbPart
        from domain import INITSIZEEXTRARATIO
        self.arraySize = int(nbPartTarget * INITSIZEEXTRARATIO)


        # creation of arrays
        self.crd = self.createArrays(self.arraySize)
        
        # creation of swap arrays
        swapSize = int(0.05 * self.arraySize)
        self.leftSwapCrd, self.amountToSwapLeft   = self.createArrays(swapSize), 0
        self.rightSwapCrd, self.amountToSwapRight = self.createArrays(swapSize), 0
        self.temp = self.createArrays(1)


        # living particles

        indices = np.linspace(0, self.arraySize - 1, self.nbPart, dtype=np.int32)
        assert(self.nbPart == len(np.unique(indices)))
        self.crd["wheres"][indices] = range(1, self.nbPart + 1)
        self.crd["wheres"][indices] += self.startIndex

        # output buffer
        self.upToDate = False
        self.positions = np.zeros((self.arraySize, 2), dtype=float)  # not used for computations but for opengl draws

        # init of locations and velocities
        self.randomInit(effectiveTemp,colorRatio)

        # neighboring cells
        self.leftCell = None
        self.rightCell = None

        # wall
        self.wall = None

    def createArrays(self,size):
        tp = np.dtype([("xs", np.float64), ("ys", np.float64), ("vxs", np.float64), ("vys", np.float64), ("wheres", np.int32),
             ("lastColls", np.float64), ("colors", np.float64), ("indLeft", np.int32), ("indRight", np.int32)])
        crd = np.zeros(size, dtype=tp, order='F')
        from domain import DEAD
        crd["wheres"][:] = DEAD
        crd["colors"][:] = 1.
        crd["indLeft"][:] = -1
        crd["indRight"][:] = -1
        
        return crd

    def zeroV_xYVelocityProfile(self, y):
        """
        initial velocity profile against y coordinate
        zero by default ; can be overridden
        """
        return 0.

    def randomInit(self, effectiveTemp,colorRatio):

        vStar = getMeanSquareVelocity(self.csts["kbs"], self.csts["ms"], effectiveTemp)

        self.crd["vxs"] = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)
        self.crd["vys"] = np.random.normal(0, vStar / 2 ** 0.5, self.arraySize)

        self.crd["colors"] = np.random.choice([0.,1.], self.arraySize, p=[1-colorRatio, colorRatio])

        # enforce true temperature and zero mean velocity
        from domain import  DEAD
        indices = np.nonzero(self.crd["wheres"] != DEAD)
        self.crd["vxs"][indices] -= np.average(self.crd["vxs"][indices])
        self.crd["vys"][indices] -= np.average(self.crd["vys"][indices])
        ratio = (np.average(self.crd["vxs"][indices] ** 2 + self.crd["vys"][indices] ** 2)) ** 0.5 / vStar
        self.crd["vxs"] /= ratio
        self.crd["vys"] /= ratio

        # locations and states
        if self.csts["fillRatio"] < 0.1:

            # random init
            self.crd["xs"] = self.left + (np.random.random(self.arraySize)) * self.length
            for i in range(self.arraySize):
                self.crd["ys"][i] = (i + 0.5) * self.csts["width"] / self.arraySize
        else:
            #print("cristal like init")
            # cristal like init
            L = self.length
            H = self.csts["width"]
            deltax = np.sqrt(2 * L * H / self.nbPart / np.sqrt(3.))
            deltay = deltax * np.sqrt(3) / 2

            nx = int(0.5 + self.length / deltax)
            ny = int(0.5 + self.length / deltay)

            if (nx - 1) * ny >= self.nbPart:
                nx = nx - 1

            if nx * (ny - 1) >= self.nbPart:
                ny = ny - 1

            deltax = deltax * (nx / (nx + 1)) ** 0.5
            deltay = deltay * (ny / (ny + 1)) ** 0.5

            id0 = self.crd["wheres"][indices[0][0]] - 1
            for ind in indices[0]:
                id = self.crd["wheres"][ind] - 1

                i = (id - id0) % nx
                j = (id - id0) // nx
                self.crd["xs"][ind] = self.left + i * deltax + deltax / 4 * (-1) ** (j % 2) + 2 * deltax / 3
                self.crd["ys"][ind] = j * deltay + 2 * deltay / 3

        for i in range(self.arraySize):
            self.crd["vxs"][i] += self.v_xYVelocityProfile(self.crd["ys"][i])

        sorting.sortCell(self.crd,self.temp)



    def count(self):
        ct = measuring.countAlive(self.crd, self.csts)
        if ct / self.arraySize > 0.92:
            print("coords arrays arrays nearly saturated : ", ct / self.arraySize)
            exit()
        return ct

    def computePressure(self, fup, fdown, fleft, fright):
        """
        update instant and average pressure
        :param fup: last computed force on upper wall (in N)
        :param fdown: last computed force on lower wall (in N)
        :param fleft: last computed force on left static wall (in N), negative if not provided
        :param fright: last computed force on right static wall (in N), negative if not provided
        :return: None
        """
        self.pressure = (fup + fdown) / (2 * (self.right - self.left))  # two walls up and down


    ##############################################################
    ##################          Swapping        ##################
    ##############################################################

    def applySwapFromLeft(self):
        """
        swap particles between cells
        :return: None
        """
        swapping.moveSwapToNeighbor(self.leftSwapCrd, self.leftCell.crd, self.amountToSwapLeft, self.csts["width"])


    def applySwapFromRight(self):
        """
        swap particles between cells
        :return: None
        """
        swapping.moveSwapToNeighbor(self.rightSwapCrd, self.rightCell.crd, self.amountToSwapRight, self.csts["width"])
        #                             swapCrd                ,crd,               amount,                  ymax


    def applyPeriodic(self):
        helping.movePeriodically(self.crd, self.left, self.right)

    def prepareSwapToLeft(self):
        """
        identify particles to be swapped and move them to swap arrays
        :return: None
        """
        self.amountToSwapLeft = swapping.moveToSwap(self.crd, self.leftSwapCrd, self.left, False)

    def prepareSwapToRight(self):
        """
        identify particles to be swapped and move them to swap arrays
        :return: None
        """
        self.amountToSwapRight = swapping.moveToSwap(self.crd, self.rightSwapCrd, self.right, True)


    ##############################################################
    #################   Wall and Collisions      #################
    ##############################################################

    def wallBounce(self):

        # up and down
        vStarBoundary = -1
        if self.boundaryTempUp > 0:
            vStarBoundary = getMeanSquareVelocity(self.csts["kbs"] , self.csts["ms"],self.boundaryTempUp)


        fup =  colliding.staticWallInterractionUp(self.crd, self.csts,vStarBoundary)

        vStarBoundary = -1
        if self.boundaryTempDown > 0:
            vStarBoundary = getMeanSquareVelocity(self.csts["kbs"], self.csts["ms"],self.boundaryTempDown)

        fdown = colliding.staticWallInterractionDown(self.crd, self.csts, vStarBoundary)

        # moving Wall
        if self.wall is not None:
            newX, newV = colliding.movingWallInteraction(self.crd, self.csts,self.wall._x, self.wall.velocity(), self.wall.mass())
            self.wall.setLocVel(newX, newV)

        fleft = -1
        fright = -1

        # left wall
        if self.leftCell is None and not self.csts["periodic"]:
            fleft = colliding.staticWallInteractionLeft(self.crd, self.csts, self.left)

        # right wall
        if self.rightCell is None and not self.csts["periodic"]:
            fright = colliding.staticWallInteractionRight(self.crd,self.csts, self.right)

        self.computePressure(fup, fdown, fleft, fright)


    def interfaceCollide(self):
        self.sumCollideInterface += colliding.detectCollisionsAtInterface(self.crd, self.leftCell.crd,self.csts)

    ##############################################################
    #################      Helper functions     ##################
    ##############################################################


    def getPositionsBuffer(self):
        if not self.upToDate:
            helping.twoArraysToOne(self.crd, self.positions)

        return self.positions

    def stream(self):
        helping.stream(self.crd, self.csts)

    def middle(self):
        return (self.left + self.right) * 0.5

    def updateIndicesAccordingToWall(self, x):
        """
        Negates indices of particle initially left of wall
        :param x: wall location
        :return: None
        """
        for i in range(len(self.crd["xs"])):
            if self.crd["xs"][i] < x:
                self.crd["wheres"][i] *= -1

    ##############################################################
    ###################        Update cell       #################
    ##############################################################

    def update(self):

        self.stream()

        self.wallBounce()

        sorting.sortCell(self.crd,self.temp)

        colls = colliding.detectAllCollisions(self.crd, self.csts,self.left,self.right)
        if colls < 0:
            print("cell's too thin")
            print("consider :")
            print(" - reducing the number of cells or,")
            print(" - reducing the timestep r = dt/tau ")
            exit()
        self.sumCollide += colls


        if self.csts["it"] % 4 == 0 :
            self.count() # this permits to check if arrays are nearly saturated
