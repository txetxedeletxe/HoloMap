from .mesh import Mesh, TransformableMesh, WrappedMesh

import numpy as np
import numpy.typing as npt

class AccumulationMesh(TransformableMesh):
    def __init__(self, 
                 base_mesh : Mesh,
                 accumulate_points : npt.ArrayLike = None):
        
        WrappedMesh.__init__(self,base_mesh)
        self.get_mesh_points, self.__get_mesh_points = self.__get_mesh_points, self.get_mesh_points

        self.accumulate_points = np.array() if accumulate_points is None else np.asarray(accumulate_points, copy=True)

    def __get_mesh_points(self):
        mesh_points = self.__get_mesh_points()
        return self._accumulate_mesh(mesh_points) if self.accumulate_points.size else mesh_points
    
    def _accumulate_mesh(self, mesh_points : np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

class DistanceModulatedAccumulationMesh(AccumulationMesh):
    def _accumulate_mesh(self, mesh_points : np.ndarray) -> np.ndarray:
        diff = self.accumulate_points[:,None,None] - mesh_points[None,:]
        mesh_points = mesh_points + np.mean(diff*self._distance_factor(self._point_norm(diff)),axis=0)

        return mesh_points

    def _distance_factor(self, d : np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

class GaussianAccumulationMesh(DistanceModulatedAccumulationMesh):
    def __init__(self, 
                 base_mesh : Mesh,
                 accumulate_points : npt.ArrayLike = None,
                 *,
                 sharpness : float = 1):
        DistanceModulatedAccumulationMesh.__init__(self,base_mesh,accumulate_points)

        self.sharpness = sharpness

    def _distance_factor(self, d : np.ndarray) -> np.ndarray:
        return np.exp(-(d*self.sharpness)**2)

    
