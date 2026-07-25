from typing import Callable
import numpy as np
from numba import jit
from filters.filters_factory import FiltersFactory
from filters.abstract_filters import AbstractFilters


@FiltersFactory.register("univariate_local_explosions_filter")
class UnivariateLocalExplosionsFilter(AbstractFilters):
    def __init__(self, y: np.ndarray, fn_s: Callable, fn_g: Callable, fn_ll: Callable):
        super().__init__()
        self.y = y
        self.fn_s = fn_s
        self.fn_g = fn_g
        self.fn_ll = fn_ll
        
        self.mu = None
        self.b = None
        self.s =None
        self.survival = None
        self.eps = None
        self.ell = None

    def run_filter(self, theta_mu, theta_b, theta_g, theta_d):
        mu, b, survival, eps, ell = univariate_local_explosions_filter(self.y, theta_mu, theta_b, theta_g, theta_d,
                                                                       self.fn_s, self.fn_g, self.fn_ll)

        self.mu = mu
        self.b = b
        self.survival = survival
        self.eps = eps
        self.ell = ell

    def get_log_likelihood(self):
        return self.ell


@jit(nopython=True, cache=True, fastmath=True)
def univariate_local_explosions_filter(y, theta_mu, theta_b, theta_g, theta_d, fn_s, fn_g, fn_ll):
    # set-up
    delta, beta, gamma = theta_mu
    omega, alpha = theta_b

    T = len(y)
    mu = np.zeros(T + 1, dtype=np.float64)
    b = np.zeros(T, dtype=np.float64)
    s = np.zeros(T, dtype=np.float64)
    survival = np.zeros(T, dtype=np.float64)
    eps = np.zeros(T, dtype=np.float64)
    ell = np.zeros(T, dtype=np.float64)

    # initialization
    mu[0] = np.mean(y[:10])

    # filtering step
    for t in range(1, T):
        # bubble activation at t
        g = fn_g(y[t], mu[t], b[t-1], theta_g)
        survival[t] = 1.0 if g < 0.0 else 0.0
        b[t] = (omega + alpha * b[t-1]) * survival[t]

        # likelihood at t
        eps[t] = y[t] - mu[t] - b[t]
        ell[t] = fn_ll(eps[t], theta_d)

        # GAS update
        s[t] = fn_s(eps[t], theta_d)
        mu[t+1] = delta + beta * mu[t] + gamma * s[t]

    return mu[:T], b, survival, eps, ell






