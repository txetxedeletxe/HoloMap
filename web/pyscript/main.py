from holomap import HoloMapFacade, HoloMapConfig

from pyscript import document, window, display
import pyscript.web as pysweb
from pyodide.ffi import create_proxy

import matplotlib as mpl
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt

import scipy

from io import StringIO

import collections.abc as  abc

class HoloMapWebEventHandler:

    def __init__(self, document, holomap : HoloMapFacade, *,plot_format="svg"):

        self.document = document
        self.holomap = holomap
        self.plot_format = plot_format

        self._acquire_HTML_elements()

    def _acquire_HTML_elements(self):

        # Get elements
        # Input and output pannels
        self.input_container = self.document.querySelector(".HM-input-container")
        self.output_container = self.document.querySelector(".HM-output-container")
        self.config_container = self.document.querySelector(".HM-config-container")

        # Primitive Domain selection Buttons
        self.domain = self.input_container.querySelectorAll("input[name=primitive_domain]")

        # Transformation control panels
        self.preliminary_transformations_cp = self.input_container.querySelector("input[type=text]").parentElement
        self.transformations_cp = self.output_container.querySelector("input[type=text]").parentElement

        # Plot containers
        self.left_plot_container = self.input_container.querySelector(".HM-plot_display")
        self.right_plot_container = self.output_container.querySelector(".HM-plot_display")

        # Controls
        ## Mesh
        self.alpha_resolution = self.config_container.querySelector("#alpha_res")
        self.beta_resolution = self.config_container.querySelector("#beta_res")

        self.sampling_method = self.config_container.querySelectorAll("input[name=sampling_method]")

        self.alpha_accumulation_values = self.config_container.querySelector("#alpha_accumulation")
        self.beta_accumulation_values = self.config_container.querySelector("#beta_accumulation")
        self.alpha_concentration = self.config_container.querySelector("#alpha_concentration")
        self.beta_concentration = self.config_container.querySelector("#beta_concentration")

        self.mesh_accumulation_points = self.config_container.querySelector("#mesh_accumulation_points")
        self.mesh_accumulation_sharpness = self.config_container.querySelector("#mesh_accumulation_sharpness")

        ## Plot
        self.markersize = self.config_container.querySelector("#markersize")
        self.linewidth = self.config_container.querySelector("#linewidth")

        self.marker_color_mode = self.config_container.querySelector("#marker_color_mode")
        self.marker_color = self.config_container.querySelector("#marker_color")
        self.marker_colormap = self.config_container.querySelector("#marker_colormap")
        self.grid_color_mode = self.config_container.querySelector("#grid_color_mode")
        self.grid_color = self.config_container.querySelector("#grid_color")
        self.grid_colormap = self.config_container.querySelector("#grid_colormap")

        self.paint_parameter = self.config_container.querySelectorAll("input[name=paint_parameter]")

        self.plot_style = self.config_container.querySelector("#plotstyle")

        ## Axes
        self.axis_linewdith = self.config_container.querySelector("#axis_linewdith")
        self.axis_color = self.config_container.querySelector("#axis_color")
        self.axis_tickrate = self.config_container.querySelector("#axis_tickrate")

        self.axis_scale = self.config_container.querySelector("#axis_scale")

        self.show_ticks = self.config_container.querySelector("#show_ticks")
        self.show_grid = self.config_container.querySelector("#show_grid")
        self.show_spines = self.config_container.querySelector("#show_spines")

        self.render_plot = self.config_container.querySelector("#render_plot")

        # Update configuration
        self._update_config()

    def update_elements(self):
        # Add plot styles
        for ch in self.plot_style.children: ch.remove()

        for style in ["default",*plt.style.available]:
            self.plot_style.append(pysweb.option(style,value=style)._dom_element)

        for cmap in mpl.colormaps:
            self.marker_colormap.append(pysweb.option(cmap,value=cmap)._dom_element)
            self.grid_colormap.append(pysweb.option(cmap,value=cmap)._dom_element)


    def attach_listeners(self):
        # Set event handlers
        for button in self.domain:
            button.addEventListener("click",create_proxy(self.change_primitive_domain))

        for button in self.sampling_method:
            button.addEventListener("click",create_proxy(self.change_sampling_method))

        for button in self.paint_parameter:
            button.addEventListener("click",create_proxy(self.change_paint_parameter))

        self.preliminary_transformations_cp.addEventListener("change",create_proxy(self.update_transformations))
        self.transformations_cp.addEventListener("change",create_proxy(self.update_transformations))

        self.marker_color_mode.addEventListener("change",create_proxy(self.update_colormode))
        self.grid_color_mode.addEventListener("change",create_proxy(self.update_colormode))

        self.render_plot.addEventListener("click",create_proxy(self.update))


    # Listeners
    def change_sampling_method(self, event):
        for button in self.sampling_method:
            button.removeAttribute("active")

        event.target.setAttribute("active","")

        value : str = event.target.value
        self.holomap.config.mesh_config.sampling_method = value.lower()

    def change_paint_parameter(self, event):
        for button in self.paint_parameter:
            button.removeAttribute("active")

        event.target.setAttribute("active","")

        value : str = event.target.value
        self.holomap.config.plot_config.paint_parameter = value.lower()

    def update_colormode(self, event):
        if "marker" in event.target.id:
            color = self.marker_color
            colormap = self.marker_colormap
        else:
            color = self.grid_color
            colormap = self.grid_colormap

        #display(str(color))

        if event.target.value == "single":
            colormap.style.setProperty("display","none")
            color.style.removeProperty("display")
        else:
            color.style.setProperty("display","none")
            colormap.style.removeProperty("display")

    def change_primitive_domain(self, event):
        for button in self.domain:
            button.removeAttribute("active")

        event.target.setAttribute("active","")

        value : str = event.target.value
        self.holomap.config.domain_config.primitive_domain = value.replace("-","_").lower()

        self.update()

    def update_transformations(self, event):
        # Obtain mappings
        mappings = []
        to_be_removed = []
        for ta in event.currentTarget.children:
            # window.console.log(ta)
            if not ta.value:
                to_be_removed.append(ta)
            else:
                mappings.append(ta.value)

        for tbr in to_be_removed:
            tbr.remove()

        # Add new field
        new_input = pysweb.input_(type="text",placeholder="f(z)")
        event.currentTarget.append(new_input._dom_element)

        # Update Mappings
        if self.input_container.contains(event.currentTarget):
            self.holomap.config.domain_config.primitive_domain_mappings = mappings
        else:
            self.holomap.config.domain_config.mappings = mappings


    # Other methods
    def update(self, event=None):
        self._update_config()
        self.redraw_plots()

    def _update_config(self):
        self.holomap.config.mesh_config.alpha_resolution = int(self.alpha_resolution.value)
        self.holomap.config.mesh_config.beta_resolution = int(self.beta_resolution.value)
        self.holomap.config.mesh_config.alpha_accumulate_values = list(map(float,filter(bool,self.alpha_accumulation_values.value.split(","))))
        self.holomap.config.mesh_config.beta_accumulate_values = list(map(float,filter(bool,self.beta_accumulation_values.value.split(","))))
        self.holomap.config.mesh_config.alpha_accumulate_concentration = float(self.alpha_concentration.value)
        self.holomap.config.mesh_config.beta_accumulate_concentration = float(self.beta_concentration.value)
        self.holomap.config.mesh_config.mesh_accumulate_points = list(map(complex,filter(bool,self.mesh_accumulation_points.value.split(","))))
        self.holomap.config.mesh_config.mesh_accumulate_sharpness = float(self.mesh_accumulation_sharpness.value)

        self.holomap.config.plot_config.plot_style = self.plot_style.value or "default"
        self.holomap.config.plot_config.markersize = 2**float(self.markersize.value)
        self.holomap.config.plot_config.linewidth = 2**float(self.linewidth.value)
        self.holomap.config.plot_config.points_color = self.marker_color.value if self.marker_color_mode.value == "single" else self.marker_colormap.value
        self.holomap.config.plot_config.grid_color = self.grid_color.value if self.grid_color_mode.value == "single" else self.grid_colormap.value

        self.holomap.config.axes_config.axis_linewidth = 2**float(self.axis_linewdith.value)
        self.holomap.config.axes_config.axis_line_color = self.axis_color.value
        self.holomap.config.axes_config.axis_scale = 2**float(self.axis_scale.value)
        self.holomap.config.axes_config.axis_tickrate = 2**float(self.axis_tickrate.value)
        self.holomap.config.axes_config.show_ticks = self.show_ticks.checked
        self.holomap.config.axes_config.show_grid = self.show_grid.checked
        self.holomap.config.axes_config.show_spines = self.show_spines.checked


    def redraw_plots(self):
        # Remove previous plots if they exist
        left_plot_svg = self.left_plot_container.querySelector("svg")
        right_plot_svg = self.right_plot_container.querySelector("svg")

        if left_plot_svg: left_plot_svg.remove()
        if right_plot_svg: right_plot_svg.remove()

        # Instantiate figures
        fig_init = mpl_figure.Figure(figsize=(4,4),dpi=self.holomap.config.figure_config.dpi,layout="tight")
        fig_trans = mpl_figure.Figure(figsize=(4,4),dpi=self.holomap.config.figure_config.dpi,layout="tight")

        # Make axes
        ax_init, ax_trans = fig_init.add_subplot(1,1,1), fig_trans.add_subplot(1,1,1)

        # Do the plotting
        self.holomap.plot_mesh(ax_init, ax_trans)

        # Save and serialize the figures
        io_init, io_trans = StringIO(), StringIO()
        fig_init.savefig(io_init,format=self.plot_format), fig_trans.savefig(io_trans,format=self.plot_format)

        # Insert images into document
        self.left_plot_container.insertAdjacentHTML("afterbegin",io_init.getvalue())
        self.right_plot_container.insertAdjacentHTML("afterbegin",io_trans.getvalue())

        # Remove unnecessary attributes
        left_plot_svg = self.left_plot_container.querySelector("svg")
        left_plot_svg.removeAttribute("width")
        left_plot_svg.removeAttribute("height")

        right_plot_svg = self.right_plot_container.querySelector("svg")
        right_plot_svg.removeAttribute("width")
        right_plot_svg.removeAttribute("height")



event_handler = HoloMapWebEventHandler(document, HoloMapFacade(HoloMapConfig()))
event_handler.update_elements()
event_handler.redraw_plots()
event_handler.attach_listeners()
