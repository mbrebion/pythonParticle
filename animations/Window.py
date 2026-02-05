import numpy as np
import dearpygui.dearpygui as dpg
from domain import Domain
from constants import ComputedConstants
from numbaAccelerated import twoArraysToTwo
import warnings
from dynamicPlots import createDynamicPlotWindow,updateData

#warnings.filterwarnings('ignore')
dpg.create_context()


class Window:

    def __init__(self, nPart, P, T, L, H, ls, nbCells=1, ratios=None, effectiveTemps=None, resX=1024, resY=1024,
                 periodic=False):
        # simulation
        self.X = L
        self.Y = H
        ls = ls
        ComputedConstants.thermodynamicSetup(T, self.X, self.Y, P, nPart, ls)
        self.play = True
        self.freeWall = False
        self.showPlot = False

        self.posX = np.zeros(nPart, dtype="f")
        self.posY = np.zeros(nPart, dtype="f")
        self.vWall= [0]
        self.times = [0]
        self.nStep = 2

        self.domain = Domain(nbCells, effectiveTemps=effectiveTemps, ratios=ratios, periodic=periodic)
        self.wallTarget = 0.5
        ComputedConstants.resX = resX
        ComputedConstants.resY = resY

        #Particles, wall and target for wall
        with dpg.theme(tag="plot_theme_Particles"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (250, 250, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, self.getRadius(), category=dpg.mvThemeCat_Plots)

        with dpg.theme(tag="plot_theme_Wall"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 4, category=dpg.mvThemeCat_Plots)

            with dpg.theme(tag="plot_theme_DriveWall"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (128, 128, 0), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

        with dpg.window(tag="MainWindow"):

            # main controls
            with dpg.group(horizontal=True,tag="buttonBar"):
                def showPlots(sender):
                    self.showPlot = not self.showPlot
                    if self.showPlot:
                        dpg.configure_item("GraphWindow", show=True)
                    else:
                        dpg.configure_item("GraphWindow", show=False)

                def pause(sender):
                    self.play = not self.play
                    if self.play:
                        dpg.configure_item("pp", label="||")
                    else:
                        dpg.configure_item("pp", label="|>")

                def withWall(sender):

                    if dpg.get_value("withWall"):
                        self.addWall()
                    else:
                        self.removeWall()

                def speedSlider(sender):
                    self.nStep = dpg.get_value("speedSlider")

                dpg.add_button(tag="pp", label="||", callback=pause)
                dpg.add_slider_int(tag="speedSlider", width=60,label="Vitesse",min_value=1,max_value=20,callback=speedSlider)
                dpg.set_value("speedSlider",1)
                dpg.add_spacer(width=30)
                dpg.add_checkbox(tag="showPlots", label="show plots", callback=showPlots)
                dpg.add_spacer(width=30)
                dpg.add_checkbox(tag="withWall", label="wall", callback=withWall)

            # Main simulation plot
            with dpg.plot(label="", height=-1, width=-1, tag="simulationPlot"):
                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis")
                dpg.set_axis_limits(dpg.last_item(), 0, self.X)
                dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")
                dpg.set_axis_limits(dpg.last_item(), 0, self.Y)

                # scatter plot for colors > 0
                dpg.add_scatter_series([], [], label="", parent="y_axis", tag="plotParticles")
                dpg.bind_item_theme("plotParticles", "plot_theme_Particles")


            createDynamicPlotWindow()

        self.updateProgram()

    def removeWall(self):
        self.domain.removeMovingWall()
        dpg.delete_item("dline1")
        dpg.delete_item("wallb")
        dpg.delete_item("wall")
        dpg.delete_item("spacerWall")
    def addWall(self):
        self.domain.addMovingWall(10, self.X/2, 0, imposedVelocity=self.wallVelocity)
        def freeWall(sender):
            self.freeWall = not self.freeWall
            if self.freeWall:
                self.domain.wall.setFree()
                dpg.configure_item("dline1", show=False)
            else:
                self.domain.wall.unSetFree()
                dpg.configure_item("dline1", show=True)

        dpg.add_spacer(width=30,parent = "buttonBar",tag="spacerWall")
        dpg.add_checkbox(tag="wallb", label="free wall", callback=freeWall,parent = "buttonBar")



        def updateWallTarget(val):
            self.wallTarget = dpg.get_value(val)

        dpg.add_drag_line(tag="dline1", default_value=self.X * self.wallTarget, callback=updateWallTarget,parent = "simulationPlot")
        dpg.add_line_series([0.5, 0.5], [0, self.Y], tag="wall", parent="y_axis")
        dpg.bind_item_theme("wall", "plot_theme_Wall")
        dpg.bind_item_theme("dline1", "plot_theme_DriveWall")

    def run(self):
        dpg.create_viewport(title='Particle simulation', width=ComputedConstants.resX, height=ComputedConstants.resY)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("MainWindow", True)

        while dpg.is_dearpygui_running():
            self.updateProgram()

            # update particles positions
            self.posX *=0
            self.posY *= 0
            self.posX -= 1e5
            self.posY -= 1e5

            for c in self.domain.cells:
                twoArraysToTwo(c.coords.xs, c.coords.ys, c.coords.wheres, self.posX, self.posY)
            dpg.set_value('plotParticles', [self.posX, self.posY])

            # update wall position
            if self.domain.wall != None:
                dpg.set_value("wall", [[self.domain.wall.location(), self.domain.wall.location()], [0, self.Y]])



            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def updateProgram(self):
        if self.play:
            for i in range(self.nStep):
                self.domain.update()
                if ComputedConstants.it % 100 == 1:
                    print(self.domain.getAveragedTemperatures())
                if i%4 == 0:
                    updateData(self.domain)

    def getRadius(self):
        radius = (ComputedConstants.ds / ComputedConstants.length * ComputedConstants.resX / 2 *1.05)
        return radius

    def wallVelocity(self, t,x):
        dec = self.wallTarget - self.domain.wall.location()
        v = 100 * np.arctan(dec * 10)
        return v


if __name__ == "__main__":
    window = Window(10000, 1e5, 300, 1, 1, 2e-2, nbCells=1)
    window.domain.setMaxWorkers(1)

    ComputedConstants.dt /= 1
    window.nStep = 1
    window.run()
