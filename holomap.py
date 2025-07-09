import matplotlib.axes
from src.domain import RadialComplexDomain, CuadrantsComplexDomain
from src.mesh import build_domain_mesh
from src.mesh.mesh import ComplexToMesh2D
from src.mesh_plotter import MeshPlotter

import sys 
import numpy as np
import sympy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color


import argparse

def _build_parser():
    parser = argparse.ArgumentParser()

    domain_parser = parser.add_argument_group("DOMAIN")
    domain_parser.add_argument("mappings", nargs="+", help="Functions to apply to the initial domain (as a function of z).")
    domain_parser.add_argument("--primitive_domain", choices=("disk","half_plane","half_disk","quadrant"), default="disk", help="Primitive domain to use as a primer for the starting domain.")
    domain_parser.add_argument("--primitive_domain_mappings", nargs="+", help="Mappings to apply to the primitive domain to obtain the starting domain.")

    domain_parser.add_argument("--epsilon", type=float, default=1e-5, help="Size of margin to leave to domain boundaries. This is used to emulate open domains.")

    mesh_parser = parser.add_argument_group("MESH")
    mesh_parser.add_argument("--alpha_resolution", type=int, default=15, help="Resolution in alpha-space at which to sample the domain.")
    mesh_parser.add_argument("--beta_resolution", type=int, default=15, help="Resolution in beta-space at which to sample the domain.")
    mesh_parser.add_argument("--sampling_method", default="linear", choices=("linear","random"), help="Resolution in beta-space at which to sample the domain.")

    mesh_parser.add_argument("--alpha_accumulate_values", type=float, nargs="+", default=tuple(), help="""Accumulate "alpha" domain sampling parameter at given values (between 0 and 1).""")
    mesh_parser.add_argument("--beta_accumulate_values", type=float, nargs="+", default=tuple(), help="""Accumulate "beta" domain sampling parameter at given values (between 0 and 1).""")
    mesh_parser.add_argument("--alpha_accumulate_concentration", type=float, default=4, help="""Concentration coefficient for alpha accumulation. Higher values produce more concentrated meshes.""")
    mesh_parser.add_argument("--beta_accumulate_concentration", type=float, default=4, help="""Concentration coefficient for beta accumulation. Higher values produce more concentrated meshes.""")

    mesh_parser.add_argument("--mesh_accumulate_points", type=complex, nargs="+", default=tuple(), help="Locations in the complex plane which attract mesh points in order to produce accumulation around them.")
    mesh_parser.add_argument("--mesh_accumulate_sharpness", type=float, default=2, help="Sharpness factor for gaussian accumulation.")
    
    plot_parser = parser.add_argument_group("PLOT")
    plot_parser.add_argument("--plot_style", choices=plt.style.available, default="default", help="Style of plotting to use.")
    plot_parser.add_argument("--markersize", type=float, default=1, help="Size of markers of mesh-points.")
    plot_parser.add_argument("--linewidth", type=float, default=0.1, help="Width of grid-lines.")
    plot_parser.add_argument("--points_color", default="#0000ff", help="Color to paint the mesh-points. Valid formats are: hex RGB string (single color), or a matplotlib.Colormap name.")
    plot_parser.add_argument("--grid_color", default="#0000ff", help="Color to paint the grid-lines. Valid formats are: hex RGB string (single color), or a matplotlib.Colormap name.")
    plot_parser.add_argument("--paint_parameter", choices=("alpha","beta"), default="beta", help="Parameter to which the color index in the colormap is associated. Only effective when a colomap is used.")

    figure_parser = parser.add_argument_group("FIGURE")
    figure_parser.add_argument("--only_transformed_mesh", action="store_true", help="Plot only the transformed (final) mesh.")
    figure_parser.add_argument("--dpi", type=float, default=192, help="DPI at which to render the plot.")

    axes_parser = parser.add_argument_group("AXES")
    axes_parser.add_argument("--axis_linewidth", type=float, default=0.5, help="Width of the X-Y axis lines. Set to 0 to hide.")
    axes_parser.add_argument("--axis_line_color", default="#000000", help="Width of the X-Y axis lines in hex RGB format.")
    axes_parser.add_argument("--axis_scale", type=float, default=2, help="Scale of both X-Y axes, higher values show larger areas of the plot.")
    axes_parser.add_argument("--axis_tickrate", type=float, default=1, help="Intervals at which to place ticks (when ticks are shown).")
    axes_parser.add_argument("--show_ticks", action="store_true", help="Show ticks at tickrate interval.")
    axes_parser.add_argument("--show_grid", action="store_true", help="Show grid at tickrate interval.")
    axes_parser.add_argument("--show_spines", action="store_true", help="Show border spines around the plot.")
    

    debug_parser = parser.add_argument_group("DEBUG")
    debug_parser.add_argument("--hide_warnings", action="store_true", help="Do not show warning messages (some error messages can also be blocked).")

    parser.set_defaults(
        hide_warnings=False,
        switch_paint_direction=False,
        only_transformed_mesh=False,
        show_ticks=False,
        show_grid=False,
        show_spines=False)

    return parser


def _restyle_axes(
        axs : matplotlib.axes.Axes,
        *,
        axis_linewidth : float = 0.5,
        axis_line_color : str = "#000000",
        axis_scale : float = 2,
        axis_tickrate : float = 1,
        show_ticks : bool = False,
        show_grid : bool = False,
        show_spines : bool = False,
        ):
    axs.axhline(color=axis_line_color,linewidth=axis_linewidth)
    axs.axvline(color=axis_line_color,linewidth=axis_linewidth)

    axs.set_xlim(-axis_scale,axis_scale)
    axs.set_ylim(-axis_scale,axis_scale)

    if show_ticks:
        ticks = np.linspace(-int(axis_scale),int(axis_scale),int(2*int(axis_scale)/axis_tickrate + 1),endpoint=True)
        axs.set_xticks(ticks,ticks)
        axs.set_yticks(ticks,ticks)
    else:
        axs.tick_params(axis="both",which="both",bottom=False,left=False,right=False,top=False,labelbottom=False,labelleft=False,)

    if not show_spines: plt.setp(axs.spines.values(),visible=False)
    axs.grid(visible=show_grid,which="major")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.hide_warnings:
        sys.stderr = open("/dev/null","w")

    # Parse mappings
    transformations = map(sympy.sympify,args.mappings)
    transformations = [sympy.lambdify(sympy.Symbol("z"), f, "numpy") for f in transformations]

    if args.primitive_domain_mappings is not None:
        initial_transofmations = map(sympy.sympify,args.primitive_domain_mappings)
        initial_transofmations = [sympy.lambdify(sympy.Symbol("z"), f, "numpy") for f in initial_transofmations]
    else:
        initial_transofmations = None

    # Starting Domain
    match args.primitive_domain:
        case "disk": domain = RadialComplexDomain(epsilon=args.epsilon)
        case "half_disk": domain = RadialComplexDomain(angle_range=(0,np.pi),include_limits_angle=False,include_limits_radius=False,epsilon=args.epsilon)
        case "quadrant": domain = CuadrantsComplexDomain(epsilon=args.epsilon)
        case "half_plane": domain = CuadrantsComplexDomain(reflect_x=True,epsilon=args.epsilon)

    # Mesh
    init_mesh = build_domain_mesh(
        domain,
        args.alpha_resolution,
        args.beta_resolution,
        sampling_method=args.sampling_method,
        mesh_accumulate_points=args.mesh_accumulate_points,
        mesh_accumulate_args=dict(sharpness=args.mesh_accumulate_sharpness),
        alpha_accumulate_values=args.alpha_accumulate_values,
        beta_accumulate_values=args.beta_accumulate_values,
        parameter_accumulation_args=dict(alpha_concentration=args.alpha_accumulate_concentration, beta_concentration=args.beta_accumulate_concentration),
        transformations=initial_transofmations)
    init_mesh = ComplexToMesh2D(init_mesh)

    transformed_mesh = init_mesh.transfom_mesh(transformations) # Transform mesh

    # Get points
    init_2D = init_mesh.get_mesh_points()
    transformed_2D = transformed_mesh.get_mesh_points()

    # Plot sets 
    # compute colors
    color = args.points_color if args.points_color.startswith("#") else matplotlib.colormaps[args.points_color]
    grid_color = args.grid_color if args.grid_color.startswith("#") else matplotlib.colormaps[args.grid_color]

    mesh_plotter = MeshPlotter(
        markersize=args.markersize,
        linewidth=args.linewidth,
        points_color=color,
        grid_color=grid_color,
        paint_parameter=args.paint_parameter)

    # PLOT
    plt.style.use(args.plot_style) # Set style
    
    if not args.only_transformed_mesh:
        fig, (initial_mesh_ax, transformed_mesh_ax) = plt.subplots(1,2,figsize=(8,4),dpi=args.dpi) 

        mesh_plotter.plot_mesh(init_2D,initial_mesh_ax) # Plot initial mesh
        _restyle_axes(initial_mesh_ax,
                      axis_linewidth=args.axis_linewidth,
                      axis_line_color=args.axis_line_color,
                      axis_scale=args.axis_scale,
                      show_ticks=args.show_ticks,
                      show_grid=args.show_grid,
                      show_spines=args.show_spines)
    else:
        fig, transformed_mesh_ax = plt.subplots(1,1,figsize=(4,4),dpi=args.dpi) 

    mesh_plotter.plot_mesh(transformed_2D,transformed_mesh_ax) # Plot transformed mesh
    _restyle_axes(transformed_mesh_ax,
                      axis_linewidth=args.axis_linewidth,
                      axis_line_color=args.axis_line_color,
                      axis_scale=args.axis_scale,
                      show_ticks=args.show_ticks,
                      show_grid=args.show_grid,
                      show_spines=args.show_spines)
    
    fig.tight_layout()
    plt.show()
