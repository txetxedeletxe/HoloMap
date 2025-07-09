from ..domain.domain import Domain, ComplexDomain

from .mesh import ComplexMesh, CachedMesh, ComplexToMesh2D
from .domain_mesh import LinearSamplingDomainMesh, RandomSamplingDomainMesh
from .domain_accumulation_mesh import DomainBetaAccumulationMesh
from .accumulation_mesh import GaussianAccumulationMesh

import numpy.typing as npt

from typing import List, Callable

def build_domain_mesh(
        domain : Domain, 
        alpha_resolution : int,
        beta_resolution : int,
        *,
        sampling_method : str = "linear",
        alpha_accumulate_values : npt.ArrayLike = None,
        beta_accumulate_values : npt.ArrayLike = None,
        parameter_accumulation_method : str = "beta",
        parameter_accumulation_args : dict = None,
        mesh_accumulate_points : npt.ArrayLike = None,
        mesh_accumulate_method : str = "gaussian",
        mesh_accumulate_args : dict = None,
        transformations : List[Callable] = None,
        use_cache : bool = False,
        ):

    # Get base class
    match sampling_method.lower():
        case "linear": mesh_base_class = LinearSamplingDomainMesh
        case "random": mesh_base_class = RandomSamplingDomainMesh
        case _: raise ValueError("""The allowed sampling methods are: "linear" and "random".""")

    if isinstance(domain,ComplexDomain):
        mesh_base_class = type("Complex{}".format(mesh_base_class.__name__),(ComplexMesh,mesh_base_class),dict())

    # Instantiate Mesh
    domain_mesh = mesh_base_class(domain,alpha_resolution,beta_resolution)
   

    if alpha_accumulate_values is not None or beta_accumulate_values is not None:
        parameter_accumulation_args = parameter_accumulation_args or dict()
        match parameter_accumulation_method.lower():
            case "beta": 
                domain_mesh = DomainBetaAccumulationMesh(domain_mesh,
                    alpha_accumulate_values=alpha_accumulate_values,
                    beta_accumulate_values=beta_accumulate_values,
                    **parameter_accumulation_args)
                
            case _: raise ValueError("""The allowed parameter accumuation methods are: "beta".""")

    if mesh_accumulate_points is not None:
        mesh_accumulate_args = mesh_accumulate_args or dict()
        match mesh_accumulate_method.lower():
            case "gaussian": domain_mesh = GaussianAccumulationMesh(domain_mesh,
                    accumulate_points=mesh_accumulate_points,
                    **mesh_accumulate_args)
                
            case _: raise ValueError("""The allowed mesh accumuation methods are: "gaussian".""")

    if transformations is not None:
        domain_mesh = domain_mesh.transfom_mesh(transformations)

    if use_cache:
        domain_mesh = CachedMesh(domain_mesh)

    

    return domain_mesh



    
    
    


