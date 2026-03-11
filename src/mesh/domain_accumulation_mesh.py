from .domain_mesh import DomainMesh, WrappedDomainMesh

import numpy as np
import numpy.typing as npt

import scipy.stats

import collections.abc as abc

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

    def __sample_alpha_beta(self) -> tuple[np.ndarray,np.ndarray]:
        alpha_mesh, beta_mesh = self.__sample_alpha_beta()
        alpha_mesh, beta_mesh = self._accumulate_parameter(alpha_mesh, beta_mesh)

        return alpha_mesh, beta_mesh

    def _accumulate_parameter(self, alpha_mesh : np.ndarray, beta_mesh: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        raise NotImplementedError()


class DomainBetaAccumulationMesh(DomainAccumulationMesh):

    class MixtureOfBetas(scipy.stats.rv_continuous):
        def __init__(self, beta_params : abc.Sequence[tuple[float,float]], *, beta_weights : float | npt.ArrayLike = 1, uniform_weight : float = 1):
            scipy.stats.rv_continuous.__init__(self,a=0,b=1)

            self.beta_params = np.asarray(beta_params)
            self.beta_weights = np.asarray(beta_weights)
            self.uniform_weight = uniform_weight

        def _cdf(self, x, *args):
            cdf = np.asarray([scipy.stats.beta.cdf(x,*ab,*args) for ab in self.beta_params] + [scipy.stats.uniform.cdf(x,*args)])
            return self._mix(cdf)
        
        def _pdf(self, x, *args):
            pdf = np.asarray([scipy.stats.beta.pdf(x,*ab,*args) for ab in self.beta_params] + [scipy.stats.uniform.pdf(x,*args)])
            return self._mix(pdf)
            
        def _mix(self,vals):
            weights = np.full(self.beta_params.shape[0],self.beta_weights) if not self.beta_weights.shape else self.beta_weights
            weights = np.asarray([*weights,self.uniform_weight])

            avg = np.average(vals,axis=0,weights=weights)
            return avg

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
            return DomainBetaAccumulationMesh.MixtureOfBetas([])

        # Compute beta parameters
        a = np.full_like(accumulate_val, concentration)
        b = np.full_like(accumulate_val, concentration)

        lower_half = accumulate_val < 0.5
        a[lower_half] = (1/(1-accumulate_val[lower_half])*((b[lower_half]-2)*accumulate_val[lower_half] + 1))
        b[~lower_half] = (1/accumulate_val[~lower_half]*(a[~lower_half]*(1 - accumulate_val[~lower_half]) - 1)) + 2
        
        ab = np.stack((a,b),axis=1)

        # Instantiate beta random variables
        mixture_rv = DomainBetaAccumulationMesh.MixtureOfBetas(ab)
        return mixture_rv

    def _accumulate_parameter(self, alpha_mesh : np.ndarray, beta_mesh: np.ndarray) -> tuple[np.ndarray,np.ndarray]:            
        alpha_mesh = self.rv_alpha.ppf(alpha_mesh)
        beta_mesh = self.rv_beta.ppf(beta_mesh)

        return alpha_mesh, beta_mesh
