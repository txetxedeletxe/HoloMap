from .domain_mesh import DomainMesh, WrappedDomainMesh

import numpy as np
import numpy.typing as npt

import scipy.stats

from typing import Tuple

class DomainAccumulationMesh(WrappedDomainMesh):
        def __init__(self, 
                 base_domain_mesh : DomainMesh,
                 *,
                 alpha_accumulate_values : npt.ArrayLike = None,
                 beta_accumulate_values : npt.ArrayLike = None):
            
            WrappedDomainMesh.__init__(self,base_domain_mesh)
            self._sample_alpha_beta, self.__sample_alpha_beta = self.__sample_alpha_beta, self._sample_alpha_beta

            self.alpha_accumulate_values = np.array() if alpha_accumulate_values is None else np.asarray(alpha_accumulate_values, copy=True)
            self.beta_accumulate_values = np.array() if beta_accumulate_values is None else np.asarray(beta_accumulate_values, copy=True)

        def __sample_alpha_beta(self) -> Tuple[np.ndarray,np.ndarray]:
            alpha_mesh, beta_mesh = self.__sample_alpha_beta()
            alpha_mesh, beta_mesh = self._accumulate_parameter(alpha_mesh, beta_mesh)

            return alpha_mesh, beta_mesh

        def _accumulate_parameter(self, alpha_mesh : np.ndarray, beta_mesh: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
            raise NotImplementedError()


class DomainBetaAccumulationMesh(DomainAccumulationMesh):
        def __init__(self, 
                base_domain_mesh : DomainMesh,
                *,
                alpha_accumulate_values : npt.ArrayLike = None,
                beta_accumulate_values : npt.ArrayLike = None,
                alpha_concentration : float = 4,
                beta_concentration : float = 4):
            
            DomainAccumulationMesh.__init__(self, 
                base_domain_mesh, 
                alpha_accumulate_values=alpha_accumulate_values,
                beta_accumulate_values=beta_accumulate_values)

            self.alpha_concentration, self.beta_concentration = alpha_concentration, beta_concentration

            self.rv_alpha = self._build_rv(self.alpha_accumulate_values,alpha_concentration)
            self.rv_beta = self._build_rv(self.beta_accumulate_values,beta_concentration)

        def _build_rv(self, accumulate_val : np.ndarray, concentration : float):
            if not accumulate_val.size:
                return scipy.stats.Uniform(a=0,b=1)

            # Compute beta parameters
            a = np.full_like(accumulate_val, concentration)
            b = np.full_like(accumulate_val, concentration)

            lower_half = accumulate_val < 0.5
            a[lower_half] = (1/(1-accumulate_val[lower_half])*((b[lower_half]-2)*accumulate_val[lower_half] + 1))
            b[~lower_half] = (1/accumulate_val[~lower_half]*(a[~lower_half]*(1 - accumulate_val[~lower_half]) - 1)) + 2

            # Instantiate beta random variables
            beta_distribution = scipy.stats.make_distribution(scipy.stats.beta)
            beta_rvs = [beta_distribution(a=ai,b=bi) for ai,bi in zip(a,b)]

            mixture_rv = scipy.stats.Mixture([scipy.stats.Uniform(a=0,b=1)] + beta_rvs) # Mix rvs
            return mixture_rv

        def _accumulate_parameter(self, alpha_mesh : np.ndarray, beta_mesh: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:            
            alpha_mesh = self.rv_alpha.icdf(alpha_mesh)
            beta_mesh = self.rv_beta.icdf(beta_mesh)

            return alpha_mesh, beta_mesh
