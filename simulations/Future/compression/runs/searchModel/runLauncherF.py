import time

import numpy as np

from constants import ComputedConstants
from domain import Domain

##############
### inputs ###
##############

X = 0.2
Y = 0.1
ls = X / 25
nPart = 256000
T = 300
P = 1e5
nbDomain = 32
vstarov = 100
maxWorker = 3
wallInit = 8 * X / 10

##############
nbAverage = 4  # number of identical runs launched to output average and uncertainties
idleRatio = .25  # extra % time spent in sleep to reduce CPU heating
###############
### outputs ###
###############

def getOutPutsNames(dd):
    """
    return xlabel, ylabel, legend for run with dd settings, and color
    :param dd: dict of settings
    :return: list of strings
    """

    return ["$X(t)/X(0)$", "$r$", "$|v/v^*| = 1/"+str(dd["vstarov"])+" $", "blue"]

def getOutPutsAsList(d, dd):
    """
    list containing all values to be outputed ; uncertanties may be added to this list of more than one run is asked
    :param d: domain
    :param d: data dict
    :return: list
    """
    x = d.wall.location() / dd["wallInit"]
    Ns = d.countLeft() * np.pi * (ComputedConstants.ds / 2) ** 2
    S = d.wall.location() * dd["Y"]
    ecTotal = d.computeKineticEnergyLeftSide()
    ecMacro = d.computeAverageVelocityLeftOfWall()**2 * 0.5 * d.countLeft() * ComputedConstants.ms
    c = ecTotal * S * (1 - 2 * Ns / S)

    output = [x,c/d.initialC,ecTotal,ecMacro,ComputedConstants.time]
    print(output)
    return output


def outputFileName(dd):
    return (str(int(dd["nPart"] / 1000)) + "_" + str(dd["nbDomain"]) + "_" + str(dd["vstarov"])+ "_" + str(int(dd["X"]/dd["ls"]+0.1))+"_" + str(dd["X"]) +"_" + str(dd["Y"]) + "_PRA.txt")

##################
### run params ###
##################

def outputCriterion(d, dd):
    """
    decide whether data should be output : code to output 200 lines per run
    :param d: domain
    :param d: data dict
    :return: True if data is to be output ; False else
    """
    nbItTotalEstim = int( (dd["wallInit"]/2 * dd["vstarov"] / ComputedConstants.vStar) / ComputedConstants.dt + 0.1 )
    return ComputedConstants.it * 200 % nbItTotalEstim == 0


def runCriterion(d, dd):
    """
    decide whether simulation should go on
    :param d: domain
    :param d: data dict
    :return: True if simulation should go on ; else False
    """
    return d.wall.location() > 5 * dd["wallInit"] / 10


def initRun(d, dd):
    """
    prepare run properly
    :param d: domain
    :param d: data dict
    :return: None
    """

    def velocity(t,x):
        if t <= ComputedConstants.dt * 100:
            return 0.

        return -ComputedConstants.vStar / dd["vstarov"]


    xInit = 8 * dd["X"] / 10
    d.addMovingWall(1000, xInit, 40, imposedVelocity=velocity)
    d.update()
    Ns = d.countLeft() * np.pi * (ComputedConstants.ds / 2) ** 2
    S = dd["wallInit"] * dd["Y"]
    d.initialC = d.computeKineticEnergyLeftSide() * S * (1 - 2 * Ns / S)


####################
### do the magic ###
####################

mdd = {"X": X, "Y": Y, "ls": ls, "nPart": nPart, "T": T, "P": P, "nbDomain": nbDomain, "vstarov": vstarov, "wallInit": wallInit}


def launchAll():
    globalOutputData = []
    # start runs
    for k in range(nbAverage):
        globalOutputData.append(launchRun(mdd))

    # produce statistics
    nbVars = len(globalOutputData[0][0])
    globalAverages = []
    globalUncertainties = []
    for i in range(len(globalOutputData[0])):
        avs = np.array([0.] * nbVars)
        for j in range(nbAverage):
            avs += globalOutputData[j][i] / nbAverage
        globalAverages.append(avs)

        ucs = np.array([0.] * nbVars)
        for j in range(nbAverage):
            ucs += (globalOutputData[j][i] - avs) ** 2
        globalUncertainties.append(ucs ** 0.5 / nbAverage ** 0.5)

    # write to file
    f = open(outputFileName(mdd), "w")
    f.write(", ".join(getOutPutsNames(mdd)) +"\n")
    for i in range(len(globalAverages)):
        out = ", ".join([str(e) for e in np.concatenate((globalAverages[i],globalUncertainties[i])) ])
        f.write( out +"\n")
    f.close()


def launchRun(dd):
    ComputedConstants.thermodynamicSetup(T, X, Y, P, nPart, ls)
    domain = Domain(nbDomain)
    domain.setMaxWorkers(maxWorker)
    initRun(domain, dd)

    outputSet = []
    outputSet.append(np.array(getOutPutsAsList(domain, dd)))
    begin = time.time()

    while runCriterion(domain, dd):
        domain.update()
        if outputCriterion(domain, dd):
            outputSet.append(np.array(getOutPutsAsList(domain, dd)))
            duration = time.time() - begin
            time.sleep(duration*idleRatio)
            begin = time.time()

    outputSet.append(np.array(getOutPutsAsList(domain, dd)))

    return outputSet


launchAll()
