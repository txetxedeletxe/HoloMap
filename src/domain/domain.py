import numpy as np

from typing import Tuple, Union

# Base class
class Domain:
    def get_points(self, alpha : np.ndarray, beta : np.ndarray) -> np.ndarray: raise NotImplementedError()

class ComplexDomain(Domain): pass

class DomainWrapper(Domain):
    def __init__(self, base_domain : Domain):
        self.base_domain = base_domain

    def get_points(self, alpha : np.ndarray, beta : np.ndarray) -> np.ndarray:
        return self.base_domain.get_points(alpha, beta)

class OpenDomain(DomainWrapper):
    def __init__(self, 
                 base_domain : Domain,
                 *,
                 include_limits_alpha : Union[bool,Tuple[bool,bool]] = False,
                 include_limits_beta : Union[bool,Tuple[bool,bool]] = False,
                 epsilon : float = 1e-5):
        DomainWrapper.__init__(self,base_domain)
        self.get_points, self.__get_points = self.__get_points, self.get_points

        self.include_limits_alpha = include_limits_alpha
        self.include_limits_beta = include_limits_beta
        self.epsilon = epsilon

    def __get_points(self, alpha : np.ndarray, beta : np.ndarray) -> np.ndarray:
        if self.include_limits_alpha:
            lower_alpha = 0 if hasattr(self.include_limits_alpha,"__getitem__") and self.include_limits_alpha[0] else self.epsilon
            upper_alpha = 1 if hasattr(self.include_limits_alpha,"__getitem__") and self.include_limits_alpha[1] else 1-self.epsilon
        else:
            lower_alpha, upper_alpha = self.epsilon, 1-self.epsilon

        if self.include_limits_beta:
            lower_beta = 0 if hasattr(self.include_limits_beta,"__getitem__") and self.include_limits_beta[0] else self.epsilon
            upper_beta = 1 if hasattr(self.include_limits_beta,"__getitem__") and self.include_limits_beta[1] else 1-self.epsilon
        else:
            lower_beta, upper_beta = self.epsilon, 1-self.epsilon

        alpha = np.interp(alpha,(0,1),(lower_alpha,upper_alpha))
        beta = np.interp(beta,(0,1),(lower_beta,upper_beta))

        return self.__get_points(alpha, beta)
    

