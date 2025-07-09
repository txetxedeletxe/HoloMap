import numpy as np
import numpy.typing as npt
from scipy.stats import beta

from typing import List, Callable

# Base classes
class Mesh:
    # API
    def get_mesh_points(self) -> np.ndarray: raise NotImplementedError()
    def transfom_mesh(self, transformations : List[Callable]) -> "Mesh": raise NotImplementedError()

    # helper methods that might be needed for some implementations
    def _point_norm(self, points : np.ndarray): raise NotImplementedError()

class ComplexMesh(Mesh):
    def _point_norm(self, points : np.ndarray):
        return np.abs(points)

class Mesh2D(Mesh):
    def _point_norm(self, points : np.ndarray):
        return np.sqrt(points[:,:,0]**2 + points[:,:,1]**2)

class WrappedMesh(Mesh):
    def __init__(self, base_mesh : Mesh):
        self.base_mesh = base_mesh

    def get_mesh_points(self):
        return self.base_mesh.get_mesh_points()
    
    def transfom_mesh(self, transformations : List[Callable]) -> Mesh:
        return self.base_mesh.transfom_mesh(transformations)
    
    def _point_norm(self, points : np.ndarray):
        return self.base_mesh._point_norm(points)

class TransformedMesh(WrappedMesh):
    def __init__(self, base_mesh : Mesh, transformations : List[Callable] = None):
        WrappedMesh.__init__(self,base_mesh)
        self.get_mesh_points, self.__get_mesh_points = self.__get_mesh_points, self.get_mesh_points

        self.transformations = list() if transformations is None else list(transformations)

    def __get_mesh_points(self):
        mesh_points = self.__get_mesh_points()
        
        for t in self.transformations:
            mesh_points = t(mesh_points)

        return mesh_points
    
    def transfom_mesh(self, transformations : List[Callable]) -> Mesh:
        return TransformedMesh(self,transformations)
    
class CachedMesh(WrappedMesh):
    def __init__(self, base_mesh : Mesh):
        WrappedMesh.__init__(self,base_mesh)
        self.get_mesh_points, self.__get_mesh_points = self.__get_mesh_points, self.get_mesh_points

        self._mesh_points : np.ndarray = None

    def __get_mesh_points(self) -> np.ndarray:
        if self._mesh_points is None: self._mesh_points = self.__get_mesh_points()
        return np.copy(self._mesh_points)

class ComplexToMesh2D(Mesh2D,WrappedMesh):
    def __init__(self, base_mesh : ComplexMesh):
        WrappedMesh.__init__(self,base_mesh)
        self.transfom_mesh, self.__transfom_mesh = self.__transfom_mesh, self.transfom_mesh
        self.get_mesh_points, self.__get_mesh_points = self.__get_mesh_points, self.get_mesh_points
        
    def __get_mesh_points(self):
        mesh_points = self.__get_mesh_points()
        real_part, imag_part = np.real(mesh_points), np.imag(mesh_points)
        return np.stack((real_part,imag_part), axis=2)

    def __transfom_mesh(self, transformations : List[Callable]) -> Mesh:
        transformed_mesh = self.__transfom_mesh(transformations)
        return ComplexToMesh2D(transformed_mesh)

# Subclass this to have transform_mesh functionality
class TransformableMesh(TransformedMesh):
    def __init__(self): pass