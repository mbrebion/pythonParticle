import numpy as np
import dearpygui.dearpygui as dpg
from domain import Domain
from constants import ComputedConstants
from numbaAccelerated import twoArraysToTwo
import warnings

warnings.filterwarnings('ignore')
dpg.create_context()


# dpg.show_documentation()
# dpg.show_metrics()


# TODO
# - hide all wall stuff and buttons if no wall available
# - script using Window should be able to add new buttons and callbacks to existing window
# - displaying the correct marker size

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

        self.posX = np.zeros(nPart, dtype="f")
        self.posY = np.zeros(nPart, dtype="f")
        self.negX = np.zeros(nPart, dtype="f")
        self.negY = np.zeros(nPart, dtype="f")

        self.domain = Domain(nbCells, effectiveTemps=effectiveTemps, ratios=ratios, periodic=periodic)
        self.wallTarget = 0.8
        # window
        ComputedConstants.resX = resX
        ComputedConstants.resY = resY

        with dpg.theme(tag="plot_theme_positive"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (60, 250, 200), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4, category=dpg.mvThemeCat_Plots)

        with dpg.theme(tag="plot_theme_negative"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 50, 20), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4, category=dpg.mvThemeCat_Plots)


        with dpg.window(tag="Scatter"):
            with dpg.group(horizontal=True):

                def pause(sender):
                    self.play = not self.play
                    if self.play:
                        dpg.configure_item("pp", label="pause")
                    else:
                        dpg.configure_item("pp", label="play")

                def freeWall(sender):
                    self.freeWall = not self.freeWall
                    if self.freeWall:
                        dpg.configure_item("wallb", label="control wall")
                        self.domain.wall.setFree()
                    else:
                        dpg.configure_item("wallb", label="free wall")
                        self.domain.wall.unSetFree()

                dpg.add_button(tag="pp", label="pause", callback=pause)
                dpg.add_button(tag="wallb", label="free wall", callback=freeWall)

            # create plot
            with dpg.plot(label="Line Series", height=-1, width=-1):
                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis")
                dpg.set_axis_limits(dpg.last_item(), 0, self.X)
                dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")
                dpg.set_axis_limits(dpg.last_item(), 0, self.Y)

                # scatter plot for colors > 0
                dpg.add_scatter_series([], [], label="0.5 + 0.5 * sin(x)", parent="y_axis", tag="plotPositive")
                dpg.bind_item_theme("plotPositive", "plot_theme_positive")
                # scatter plot for colors < 0
                dpg.add_scatter_series([], [], label="0.5 + 0.5 * sin(x)", parent="y_axis", tag="plotNegative")
                dpg.bind_item_theme("plotNegative", "plot_theme_negative")

                def updateWallTarget(val):
                    self.wallTarget = dpg.get_value(val)

                dpg.add_drag_line(tag="dline1", color=[255, 0, 0, 255], vertical=True, default_value=self.X * 0.8,
                                  callback=updateWallTarget)
                dpg.add_line_series([0.5, 0.5], [0, self.Y], tag="wall", parent="y_axis")

        self.t = 0
        self.nStep = 1
        self.displayPerformance = False

        # timing
        self.timeStep = 0
        self.duration = 3e-4

        self.updateProgram()

    def run(self):
        dpg.create_viewport(title='Custom Title', width=ComputedConstants.resX, height=ComputedConstants.resY)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Scatter", True)

        while dpg.is_dearpygui_running():
            self.updateProgram()
            # TODO :
            # - try to adapt markersize to particle size

            # update particles positions
            self.posX *= 0
            self.posX -= 1e5
            self.posY *= 0
            self.posY -= 1e5
            self.negX *= 0
            self.negX -= 1e5
            self.negY *= 0
            self.negY -= 1e5

            for c in self.domain.cells:
                twoArraysToTwo(c.coords.xs, c.coords.ys, c.coords.wheres, c.coords.colors, 1, self.posX, self.posY)
                twoArraysToTwo(c.coords.xs, c.coords.ys, c.coords.wheres, c.coords.colors, -1, self.negX, self.negY)
            print(c.coords.colors)
            dpg.set_value('plotPositive', [self.posX, self.posY])
            dpg.set_value('plotNegative', [self.negX, self.negY])

            # update wall position
            if self.domain.wall != None:
                dpg.set_value("wall", [[self.domain.wall.location(), self.domain.wall.location()], [0, self.Y]])

            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def updateProgram(self):
        if self.play:
            for i in range(self.nStep):
                self.domain.update()

    def getRadius(self):
        return ComputedConstants.ds / ComputedConstants.length * ComputedConstants.resX / 2

    def velocity(self, t):
        dec = self.wallTarget - self.domain.wall.location()

        v = 200 * np.arctan(dec * 10)
        return v


if __name__ == "__main__":
    window = Window(4000, 1e5, 300, 1, 1, 40e-2, nbCells=1)
    window.domain.setMaxWorkers(1)

    window.domain.addMovingWall(10, 0.5, 40, imposedVelocity=window.velocity)
    ComputedConstants.dt /= 10
    window.nStep = 1
    window.displayPerformance = True
    window.run()
