from src.domain import RadialComplexDomain, QuadrantsComplexDomain
from src.mesh import build_domain_mesh
from src.mesh.mesh import ComplexToMesh2D
from src.mesh_plotter import MeshPlotter

import numpy as np
import sympy

import matplotlib as mpl
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt

import dataclasses
from dataclasses import dataclass, field
from dataclassparse_txetx import ConfigGroupDataclass, SelfParsingDataclass

import typing

@dataclass(kw_only=True)
class HoloMapConfig(SelfParsingDataclass):

    @dataclass
    class DomainConfig(ConfigGroupDataclass):
        _config_group_title = "DOMAIN"

        mappings : tuple[str,...] = field(default_factory=tuple,metadata={"help":"Functions to apply to the initial domain (as a function of z).","nargs":"+"})

        _: dataclasses.KW_ONLY
        primitive_domain : typing.Literal["disk","half_plane","half_disk","quadrant"] = field(default="disk",metadata={"help":"Primitive domain to use as a primer for the starting domain."})
        primitive_domain_mappings : tuple[str,...] = field(default_factory=tuple,metadata={"help":"Mappings to apply to the primitive domain to obtain the starting domain.","nargs":"+"})

        epsilon : float = field(default=1e-5,metadata={"help":"Size of margin to leave to domain boundaries. This is used to emulate open domains."})            

    @dataclass(kw_only=True)
    class MeshConfig(ConfigGroupDataclass):
        _config_group_title = "MESH"

        alpha_resolution : int = field(default=16,metadata={"help":"Resolution in alpha-space at which to sample the domain."})
        beta_resolution : int = field(default=16,metadata={"help":"Resolution in beta-space at which to sample the domain."})
        sampling_method : typing.Literal["uniform","random"] = field(default="uniform",metadata={"help":"Method for sampling the domain in parameter space."})

        alpha_accumulate_values : tuple[float,...] = field(default_factory=tuple, metadata={"help":"""Accumulate "alpha" domain sampling parameter at given values (between 0 and 1).""","nargs":"+"})
        beta_accumulate_values : tuple[float,...] = field(default_factory=tuple, metadata={"help":"""Accumulate "beta" domain sampling parameter at given values (between 0 and 1).""","nargs":"+"})
        alpha_accumulate_concentration : float = field(default=4, metadata={"help":"""Concentration coefficient for alpha accumulation. Higher values produce more concentrated meshes."""})
        beta_accumulate_concentration : float = field(default=4, metadata={"help":"""Concentration coefficient for beta accumulation. Higher values produce more concentrated meshes."""})

        mesh_accumulate_points : tuple[complex,...] = field(default_factory=tuple, metadata={"help":"""Locations in the complex plane which attract mesh points in order to produce accumulation around them.""","nargs":"+"})
        mesh_accumulate_sharpness : float = field(default=2, metadata={"help":"""Sharpness factor for gaussian accumulation."""})

    @dataclass(kw_only=True)
    class PlotConfig(ConfigGroupDataclass):
        _config_group_title = "PLOT"

        plot_style : str = field(default="default",metadata={"help":"""Style of plotting to use.""","choices":("default",*plt.style.available)})
        markersize : float = field(default=1,metadata={"help":"""Size of markers of mesh-points."""})
        linewidth : float = field(default=0.1,metadata={"help":"""Width of grid-lines."""})
        points_color : str = field(default="#0000ff",metadata={"help":"""Color to paint the mesh-points. Valid formats are: hex RGB string (single color), or a matplotlib.Colormap name."""})
        grid_color : str = field(default="#000000",metadata={"help":"""Color to paint the grid-lines. Valid formats are: hex RGB string (single color), or a matplotlib.Colormap name."""})
        paint_parameter : typing.Literal["alpha","beta"] = field(default="beta",metadata={"help":"""Parameter to which the color index in the colormap is associated. Only effective when a colomap is used."""})            

    @dataclass(kw_only=True)
    class FigureConfig(ConfigGroupDataclass):
        _config_group_title = "FIGURE"

        only_transformed_mesh : bool = field(default=False,metadata={"help":"""Plot only the transformed (final) mesh."""})
        dpi : float = field(default=192,metadata={"help":"""DPI at which to render the plot."""})

    @dataclass(kw_only=True)
    class AxesConfig(ConfigGroupDataclass):
        _config_group_title = "AXES"

        axis_linewidth : float = field(default=0.5,metadata={"help":"""Width of the X-Y axis lines. Set to 0 to hide."""})
        axis_line_color : str = field(default="#000000",metadata={"help":"""Width of the X-Y axis lines in hex RGB format."""})
        axis_scale : float = field(default=2,metadata={"help":"""Scale of both X-Y axes, higher values show larger areas of the plot."""})
        axis_tickrate : float = field(default=1,metadata={"help":"""Intervals at which to place ticks (when ticks are shown)."""})
        show_ticks : bool = field(default=False,metadata={"help":"""Show ticks at tickrate interval."""})
        show_grid : bool = field(default=False,metadata={"help":"""Show grid at tickrate interval."""})
        show_spines : bool = field(default=False,metadata={"help":"""Show border spines around the plot."""})

    # Class members
    domain_config : DomainConfig = field(default_factory=DomainConfig)
    mesh_config : MeshConfig = field(default_factory=MeshConfig)
    plot_config : PlotConfig = field(default_factory=PlotConfig)
    figure_config : FigureConfig = field(default_factory=FigureConfig)
    axes_config : AxesConfig = field(default_factory=AxesConfig)


class HoloMapFacade:

    # Add cache fields
    def __init__(self, config : HoloMapConfig):
        self.config = config
        
    def make_figure(self) -> mpl_figure.Figure:
        plt.style.use(self.config.plot_config.plot_style) # Set style

        if self.config.figure_config.only_transformed_mesh:
            fig = plt.figure(figsize=(4,4),dpi=self.config.figure_config.dpi,layout="tight")
            ax_init, ax_trans = None, fig.add_subplot(1,1,1)
        else:
            fig = plt.figure(figsize=(8,4),dpi=self.config.figure_config.dpi,layout="tight")
            ax_init, ax_trans = fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)

        self.plot_mesh(ax_init,ax_trans)

        return fig
    
    def plot_mesh(self, ax_init : mpl_axes.Axes = None, ax_trans : mpl_axes.Axes = None):
        plt.style.use(self.config.plot_config.plot_style) # Set style

        # Get mappings
        mappings = [sympy.lambdify(sympy.Symbol("z"), f, "numpy") if isinstance(f,str) else f for f in self.config.domain_config.mappings]
        primitive_domain_mappings = [sympy.lambdify(sympy.Symbol("z"), f, "numpy") if isinstance(f,str) else f for f in self.config.domain_config.primitive_domain_mappings]

        # Get color/colormap
        points_color = self.config.plot_config.points_color if self.config.plot_config.points_color.startswith("#") else mpl.colormaps[self.config.plot_config.points_color]
        grid_color = self.config.plot_config.grid_color if self.config.plot_config.grid_color.startswith("#") else mpl.colormaps[self.config.plot_config.grid_color]

        # Starting Domain
        match self.config.domain_config.primitive_domain:
            case "disk": domain = RadialComplexDomain(epsilon=self.config.domain_config.epsilon)
            case "half_disk": domain = RadialComplexDomain(angle_range=(0,np.pi),include_limits_angle=False,include_limits_radius=False,epsilon=self.config.domain_config.epsilon)
            case "quadrant": domain = QuadrantsComplexDomain(epsilon=self.config.domain_config.epsilon)
            case "half_plane": domain = QuadrantsComplexDomain(reflect_x=True,epsilon=self.config.domain_config.epsilon)

        # Mesh
        init_mesh = build_domain_mesh(
            domain,
            self.config.mesh_config.alpha_resolution,
            self.config.mesh_config.beta_resolution,
            sampling_method=self.config.mesh_config.sampling_method,
            mesh_accumulate_points=self.config.mesh_config.mesh_accumulate_points,
            mesh_accumulate_args=dict(sharpness=self.config.mesh_config.mesh_accumulate_sharpness),
            alpha_accumulate_values=self.config.mesh_config.alpha_accumulate_values,
            beta_accumulate_values=self.config.mesh_config.beta_accumulate_values,
            parameter_accumulation_args=dict(
                alpha_concentration=self.config.mesh_config.alpha_accumulate_concentration, 
                beta_concentration=self.config.mesh_config.beta_accumulate_concentration),
            transformations=primitive_domain_mappings)
        
        init_mesh = ComplexToMesh2D(init_mesh)
        trans_mesh = init_mesh.transfom_mesh(mappings) # Transform mesh

        # Get points
        init_2D = init_mesh.get_mesh_points()
        trans_2D = trans_mesh.get_mesh_points()

        # Get mesh plotter
        mesh_plotter = MeshPlotter(
            markersize=self.config.plot_config.markersize,
            linewidth=self.config.plot_config.linewidth,
            points_color=points_color,
            grid_color=grid_color,
            paint_parameter=self.config.plot_config.paint_parameter)
        
        if ax_init is not None:
            mesh_plotter.plot_mesh(init_2D,ax_init)
            self._restyle_axes(ax_init)
        if ax_trans is not None:
            mesh_plotter.plot_mesh(trans_2D,ax_trans)
            self._restyle_axes(ax_trans)

        
    def _restyle_axes(self, axs : mpl_axes.Axes):
        axs.axhline(color=self.config.axes_config.axis_line_color,linewidth=self.config.axes_config.axis_linewidth)
        axs.axvline(color=self.config.axes_config.axis_line_color,linewidth=self.config.axes_config.axis_linewidth)

        axs.set_xlim(-self.config.axes_config.axis_scale,self.config.axes_config.axis_scale)
        axs.set_ylim(-self.config.axes_config.axis_scale,self.config.axes_config.axis_scale)

        if self.config.axes_config.show_ticks:
            ticks = np.linspace(-int(self.config.axes_config.axis_scale),int(self.config.axes_config.axis_scale),
                                int(2*int(self.config.axes_config.axis_scale)/self.config.axes_config.axis_tickrate + 1),
                                endpoint=True)
            axs.set_xticks(ticks,ticks)
            axs.set_yticks(ticks,ticks)
        else:
            axs.tick_params(axis="both",which="both",bottom=False,left=False,right=False,top=False,labelbottom=False,labelleft=False,)

        if not self.config.axes_config.show_spines: plt.setp(axs.spines.values(),visible=False)
        axs.grid(visible=self.config.axes_config.show_grid,which="major")


if __name__ == "__main__":
    holomap_config = HoloMapConfig.parse_args()
    holomap_facade = HoloMapFacade(holomap_config)
    
    fig = holomap_facade.make_figure()
    plt.show()
