from holomap import  HoloMapFacade

import matplotlib as mpl
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

import pyscript
from pyscript.ffi import create_proxy

from io import StringIO

import typing

pyscript.window.console.log("hi")

class HoloMapWebFacade(HoloMapFacade):

    def update_config(self, config_path : str, config_value : typing.Any):
        config_namespace = self.config
        config_path = config_path.split(".")

        for cp in config_path[:-1]:
            config_namespace = getattr(config_namespace,cp)

        setattr(config_namespace,config_path[-1],config_value)

    def get_plot_data(self,format="svg") -> typing.Tuple[bytes, bytes]:
        fig_init = mpl_figure.Figure(figsize=(4,4),dpi=self.config.figure_config.dpi,layout="tight")
        fig_trans = mpl_figure.Figure(figsize=(4,4),dpi=self.config.figure_config.dpi,layout="tight")

        ax_init, ax_trans = fig_init.add_subplot(1,1,1), fig_trans.add_subplot(1,1,1)
        
        self.plot_mesh(ax_init, ax_trans)

        io_init, io_trans = StringIO(), StringIO()
        fig_init.savefig(io_init,format=format), fig_trans.savefig(io_trans,format=format)

        pyscript.window.console.log(io_init.getvalue())

holomap_facade = HoloMapWebFacade()

def update_config(config_path : str, config_value : typing.Any): holomap_facade.update_config(config_path,config_value)
def get_plot_data(format="svg"): return holomap_facade.get_plot_data(format)

pyscript.document.querySelector("#disk-button").addEventListener("click",lambda x: pyscript.window.console.log("hi"))

__export__ = ["update_config","get_plot_data"]

