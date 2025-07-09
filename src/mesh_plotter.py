import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, hex2color

from typing import Union

class MeshPlotter:
    def __init__(self,*,
        markersize : float = 1,
        linewidth : float = 0.1,
        points_color : Union[str,Colormap] = "#0000ff",
        grid_color : Union[str,Colormap] = "#0000ff",
        paint_parameter : str = "beta",
    ):
        self.markersize = markersize
        self.linewidth = linewidth
        self.points_color = points_color
        self.grid_color = grid_color
        self.paint_parameter = paint_parameter.lower()

        if self.paint_parameter not in ("alpha","beta"):
            raise ValueError("""Argument "paint_parameter" ({}) not valid, value must be "alpha" or "beta".""".format(self.paint_parameter))

    def plot_mesh(self, points : np.ndarray, ax : Axes = None):
        
        if ax is None: ax = plt.gca()

        # Lines
        alpha_lines = np.stack((points[:-1,:,None],points[1:,:,None]),axis=2).reshape((-1,2,2))
        beta_lines = np.stack((points[:,:-1,None],points[:,1:,None]),axis=2).reshape((-1,2,2))

        # Compute colors
        points_color = self._get_color_mesh(self.points_color,points)
        grid_color = self._get_color_mesh(self.grid_color,points)

        alpha_color = (grid_color[:-1,:]+grid_color[1:,:])/2 if grid_color.shape[0] > 1 else grid_color
        beta_color = (grid_color[:,:-1]+grid_color[:,1:])/2 if grid_color.shape[1] > 1 else grid_color

        points_color = np.reshape(points_color,shape=(-1,3))
        alpha_color = np.reshape(alpha_color,shape=(-1,3))
        beta_color = np.reshape(beta_color,shape=(-1,3))

        # Plot
        alpha_lines = LineCollection(alpha_lines,color=alpha_color,linewidth=self.linewidth)
        beta_lines = LineCollection(beta_lines,color=beta_color,linewidth=self.linewidth)

        ax.add_collection(alpha_lines)
        ax.add_collection(beta_lines)

        ax.scatter(points[:,:,0], points[:,:,1], s=self.markersize, c=points_color, zorder=2)


    def _get_color_mesh(self, color : Union[str,Colormap], mesh : np.ndarray) -> np.ndarray:
        if isinstance(color, Colormap):
            color_dim = int(self.paint_parameter == "beta")

            color = color(np.linspace(0,1,mesh.shape[color_dim]))[None,:,0:3]
            color = np.repeat(color,mesh.shape[1-color_dim],axis=0)

            if self.paint_parameter == "alpha": color = color.transpose((1,0,2))
            
        else:
            color = np.array([*hex2color(color)])[None,None,:]
        
        return color
