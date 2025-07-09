from .mesh import TransformableMesh, WrappedMesh
from ..domain.domain import Domain

import numpy as np

from typing import Tuple

class DomainMesh(TransformableMesh):
    def __init__(self, 
                 domain : Domain,
                 alpha_resolution : int,
                 beta_resolution : int):

        self.domain =  domain
        self.alpha_resolution = alpha_resolution
        self.beta_resolution = beta_resolution

    def get_mesh_points(self):
        return self.domain.get_points(*self._sample_alpha_beta())
    
    def _sample_alpha_beta(self) -> Tuple[np.ndarray,np.ndarray]:
        raise NotImplementedError()

class WrappedDomainMesh(DomainMesh):
    def __init__(self, base_domain_mesh : DomainMesh):
        self.base_domain_mesh = base_domain_mesh
        DomainMesh.__init__(self, 
            base_domain_mesh.domain,
            base_domain_mesh.alpha_resolution,
            base_domain_mesh.beta_resolution)
        WrappedMesh.__init__(self,base_domain_mesh)

    def _sample_alpha_beta(self) -> Tuple[np.ndarray,np.ndarray]:
        return self.base_domain_mesh._sample_alpha_beta()

class LinearSamplingDomainMesh(DomainMesh):
    def __init__(self, 
                 domain : Domain,
                 alpha_resolution : int,
                 beta_resolution : int):

        DomainMesh.__init__(self,domain,alpha_resolution,beta_resolution)

    def _sample_alpha_beta(self) -> Tuple[np.ndarray,np.ndarray]:
        alpha_mesh = np.linspace(0,1,self.alpha_resolution)
        beta_mesh = np.linspace(0,1,self.beta_resolution)
        
        return alpha_mesh, beta_mesh
    
class RandomSamplingDomainMesh(DomainMesh):
    def __init__(self, 
                 domain : Domain,
                 alpha_resolution : int,
                 beta_resolution : int):

        DomainMesh.__init__(self,domain,alpha_resolution,beta_resolution)

    def _sample_alpha_beta(self) -> Tuple[np.ndarray,np.ndarray]:
        alpha_mesh = np.sort(np.random.random(size=self.alpha_resolution))
        beta_mesh = np.sort(np.random.random(size=self.beta_resolution))
        
        return alpha_mesh, beta_mesh
    

