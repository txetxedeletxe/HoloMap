from .domain import ComplexDomain, OpenDomain

import numpy as np

from typing import Union, Tuple

# Concrete classes
class RadialComplexDomain(OpenDomain,ComplexDomain):
    def __init__(self, 
                 radius_range=(0,1), 
                 angle_range = (0,2*np.pi),
                 *,
                 include_limits_radius : Union[bool,Tuple[bool,bool]] = (True,False),
                 include_limits_angle : Union[bool,Tuple[bool,bool]] = True,
                 epsilon : float = 1e-5):
        OpenDomain.__init__(self,self,
            include_limits_alpha=include_limits_radius,
            include_limits_beta=include_limits_angle,
            epsilon=epsilon)

        self.radius_range = radius_range
        self.angle_range = angle_range

    def get_points(self, alpha : np.ndarray, beta : np.ndarray) -> np.ndarray:
        radial_points = np.interp(alpha,(0,1),self.radius_range)
        angular_points = np.exp(np.interp(beta,(0,1),self.angle_range)*1j)

        points = radial_points[:,None] * angular_points[None,:]
        return points

class CuadrantsComplexDomain(OpenDomain,ComplexDomain):
    def __init__(self, 
                 quadrant : int = 1,
                 *,
                 reflect_x : bool = False,
                 reflect_y : bool = False,
                 epsilon = 1e-5):
        OpenDomain.__init__(self,self,epsilon=epsilon)
        
        self.quadrant = quadrant

        self.reflect_x = reflect_x
        self.reflect_y = reflect_y

    def get_points(self, alpha : np.ndarray, beta : np.ndarray) -> np.ndarray:
        
        alpha_range = (0,1) if self.quadrant in (1,4) else (-1,0)
        beta_range = (0,1) if self.quadrant in (1,2) else (-1,0)
        
        if self.reflect_x: alpha_range = (-1,1)
        if self.reflect_y: beta_range = (-1,1)
        
        x = np.interp(alpha,(0,1),alpha_range)
        y = np.interp(beta,(0,1),beta_range)

        x = 1/(1-x) - 1/(1+x)
        y = 1/(1-y) - 1/(1+y)

        points = x[:,None] + (y*1j)[None,:]

        return points
        

