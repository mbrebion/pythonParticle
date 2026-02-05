import dearpygui.dearpygui as dpg


import numpy as np

from constants import ComputedConstants
from domain import Domain

plots  = []


def createDynamicPlotWindow():
    with dpg.window(tag="GraphWindow", show=False, label="Graphiques",width = 600,height=400,pos=(50,50)):
        with dpg.group(horizontal=True,tag="checkBoxBar"):
            pass

    plots.append(VelocityHistograms())
    plots.append(Temperatures())
    plots.append(Laplaces())
    plots.append(WallVelocity())
    updateWindowSize()


def updateWindowSize():
    height = 0
    for plot in plots:
        tag = plot.tag
        if dpg.get_item_configuration(tag)["show"]:
            height += dpg.get_item_configuration(tag)["height"]*1.05
    dpg.configure_item("GraphWindow", height=(height+45))

# plots

class WallVelocity:
    def __init__(self):
        self.tag = "plot_wall_velocity"
        self.vWall=[]
        self.times=[]


        with dpg.plot(tag=self.tag, label="wall velocity", height=200, width=-1,parent = "GraphWindow"):
            dpg.add_plot_axis(dpg.mvXAxis, label="t (ms)", tag=self.tag+"vx_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="v(t) (m/s)", tag=self.tag+"vy_axis")
            dpg.add_line_series([], [], label="", parent=self.tag+"vy_axis", tag=self.tag+"yaxis")
            dpg.bind_item_theme(self.tag, "plot_theme_DriveWall")

        addCheckBox(self.tag, False)
    def update(self,d:Domain):
        if d.wall is not None:
            self.times.append(ComputedConstants.time*1000)
            self.vWall.append(d.wall.velocity())
            dpg.set_axis_limits(self.tag+"vx_axis", 0, self.times[-1]) # time in ms
            dpg.set_axis_limits(self.tag+"vy_axis", min(0, min(self.vWall)) * 1.2, max(0, max(self.vWall)) * 1.2)
            dpg.set_value(self.tag+"yaxis", [self.times, self.vWall])


class Temperatures:
    def __init__(self):
        self.tag = "plot_temperature"
        self.tempLeft=[]
        self.tempRight = []
        self.times=[]
        addCheckBox(self.tag)

        with dpg.plot(tag=self.tag, label="Temperatures", height=200, width=-1,parent = "GraphWindow"):
            dpg.add_plot_axis(dpg.mvXAxis, label="t (ms)", tag=self.tag+"vx_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="T(t) (K)", tag=self.tag+"vy_axis")
            dpg.add_line_series([], [], label="T (left)", parent=self.tag+"vy_axis", tag=self.tag+"left")
            dpg.add_line_series([], [], label="T (right)", parent=self.tag + "vy_axis", tag=self.tag + "right")
            #dpg.bind_item_theme(self.tag, "plot_theme_DriveWall")
            dpg.add_plot_legend()

    def update(self,d:Domain):
        self.times.append(ComputedConstants.time * 1000)
        dpg.set_axis_limits(self.tag + "vx_axis", 0, self.times[-1])  # time in ms
        if d.wall is not None:
            self.tempLeft.append(d.computeKineticEnergyLeftSide() / d.countLeft() / ComputedConstants.kbs)
            self.tempRight.append(d.computeKineticEnergyRightSide() / d.countRight() / ComputedConstants.kbs)
            dpg.set_axis_limits(self.tag+"vy_axis", 0, max(max(self.tempLeft),max(self.tempRight))* 1.2)
            dpg.set_value(self.tag+"left", [self.times, self.tempLeft])
            dpg.set_value(self.tag +"right", [self.times, self.tempRight])
        else:
            self.tempRight.append(d.computeKineticEnergy() / d.count() / ComputedConstants.kbs)
            self.tempLeft.append(self.tempRight[-1])
            dpg.set_axis_limits(self.tag+"vy_axis", 0, max(self.tempRight)* 1.2)
            dpg.set_value(self.tag+"left", [self.times, self.tempRight])
            dpg.set_value(self.tag + "right", [self.times, self.tempRight])


class Laplaces:
    def __init__(self):
        self.tag = "plot_laplace"
        self.initLeft = 1
        self.initRight=1
        self.first=True
        self.laplaceLeft=[]
        self.laplaceRight = []
        self.times=[]


        with dpg.plot(tag=self.tag, label="r = TV^(gamma-1) / T[0]V[0]^(gamma-1)", height=200, width=-1,parent = "GraphWindow"):
            dpg.add_plot_axis(dpg.mvXAxis, label="t (ms)", tag=self.tag+"vx_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="r(t) ", tag=self.tag+"vy_axis")
            dpg.add_line_series([], [], label="r (left)", parent=self.tag+"vy_axis", tag=self.tag+"left")
            dpg.add_line_series([], [], label="r (right)", parent=self.tag + "vy_axis", tag=self.tag + "right")
            #dpg.bind_item_theme(self.tag, "plot_theme_DriveWall")
            dpg.add_plot_legend()
        addCheckBox(self.tag, False)

    def update(self,d:Domain):
        if d.wall is not None:
            if self.first :
                self.first = False
                self.initLeft = d.computeKineticEnergyLeftSide() / d.countLeft() / ComputedConstants.kbs * d.wall.location()
                self.initRight = d.computeKineticEnergyRightSide() / d.countRight() / ComputedConstants.kbs * (ComputedConstants.width- d.wall.location())

            self.times.append(ComputedConstants.time*1000)
            self.laplaceLeft.append(d.computeKineticEnergyLeftSide() / d.countLeft() / ComputedConstants.kbs * d.wall.location() / self.initLeft)
            self.laplaceRight.append(d.computeKineticEnergyRightSide() / d.countRight() / ComputedConstants.kbs * (ComputedConstants.width- d.wall.location()) / self.initRight)
            dpg.set_axis_limits(self.tag+"vx_axis", 0, self.times[-1]) # time in ms
            dpg.set_axis_limits(self.tag+"vy_axis", 0, max(max(self.laplaceLeft),max(self.laplaceRight))* 1.2)
            dpg.set_value(self.tag+"left", [self.times, self.laplaceLeft])
            dpg.set_value(self.tag +"right", [self.times, self.laplaceRight])


class VelocityHistograms:
    def __init__(self):
        self.tag = "plot_velHisto"
        addCheckBox(self.tag)
        self.maxNorm = 1e-5
        self.maxCount = 1e-5

        with dpg.plot(tag=self.tag, label="x_velocity distribution", height=200, width=-1, parent="GraphWindow"):
            dpg.add_plot_axis(dpg.mvXAxis, label="v (m/s)", tag=self.tag + "vx_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="% ", tag=self.tag + "vy_axis")
            dpg.add_line_series([], [], label="r (left)", parent=self.tag + "vy_axis", tag=self.tag + "left")
            dpg.add_line_series([], [], label="r (right)", parent=self.tag + "vy_axis", tag=self.tag + "right")
            dpg.add_plot_legend()
    def update(self,d:Domain):
        velLeft, velRight = d.getAllVelocityNorms()
        histR, bin_edgesR = np.histogram(velRight, bins=int(len(velRight) ** 0.5 / 4), density=True)

        if d.wall is not None:
            histL, bin_edgesL = np.histogram(velLeft, bins=int(len(velLeft)**0.5/4) ,density = True)
            nmaxNorm = max(self.maxNorm*0.98,max(velLeft)*1.05,max(velRight)*1.05)
            nmaxCount = max(self.maxCount*0.98,max(histL)*1.05,max(histR)*1.05)
            dpg.set_value(self.tag+"left", [bin_edgesL,histL])
            dpg.set_value(self.tag +"right", [bin_edgesR, histR])
        else:
            nmaxNorm = max(self.maxNorm * 0.98,  max(velRight) * 1.05)
            nmaxCount = max(self.maxCount * 0.98, max(histR) * 1.05)
            dpg.set_value(self.tag + "left", [bin_edgesR, histR])
            dpg.set_value(self.tag + "right", [bin_edgesR, histR])

        alpha = 0.02
        self.maxNorm = (1-alpha) * self.maxNorm + alpha * nmaxNorm
        self.maxCount = (1 - alpha) * self.maxCount + alpha * nmaxCount

        dpg.set_axis_limits(self.tag+"vx_axis", -self.maxNorm, self.maxNorm) # time in ms
        dpg.set_axis_limits(self.tag+"vy_axis", 0, self.maxCount)



def updateData(d:Domain):
    for plot in plots:
        plot.update(d)

def addCheckBox(tag,shown = True):
    def callback(sender):
        tag = sender.split("__")[0]
        dpg.configure_item(tag,show = dpg.get_value(sender))
        updateWindowSize()

    dpg.add_checkbox(tag=tag + "__cb", label=tag, callback=callback, parent="checkBoxBar", default_value=shown)
    if not shown:
        dpg.configure_item(tag, show=False)

