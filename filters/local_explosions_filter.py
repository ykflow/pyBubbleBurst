import numpy as np
from numba import jit
from filters.filters_factory import FiltersFactory
from filters.abstract_filters import AbstractFilters


@FiltersFactory.register("univariate_local_explosions_filter")
class UnivariateLocalExplosionsFilter(AbstractFilters):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.b = None
        self.s =None
        self.survival = None
        self.eps = None
        self.ell = None

    def run_filter(self, y, delta, beta, gamma, omega, alpha, kappa, c, sigma2):
        mu, b, survival, eps, ell = univariate_local_explosions_filter(
            y, delta, beta, gamma, omega, alpha, kappa, c, sigma2)
        self.mu = mu
        self.b = b
        self.survival = survival
        self.eps = eps
        self.ell = ell

    def get_log_likelihood(self):
        return self.ell


@jit(nopython=True, cache=True, fastmath=True)
def univariate_local_explosions_filter(y, delta, omega, alpha, beta, gamma, sigma, theta_s, theta_g, theta_d, fn_s, fn_g, fn_ll):
    # set-up
    T = len(y)
    mu = np.zeros(T + 1, dtype=np.float64)
    b = np.zeros(T, dtype=np.float64)
    s = np.zeros(T, dtype=np.float64)
    survival = np.zeros(T, dtype=np.float64)
    eps = np.zeros(T, dtype=np.float64)
    ell = np.zeros(T, dtype=np.float64)

    # initialization
    mu[0] = np.mean(y[:10])
    sigma2 = sigma * sigma

    # filtering step
    for t in range(1, T):
        # bubble activation at t
        g = fn_g(y[t], mu[t], b[t-1], theta_g)
        survival[t] = 1.0 if g < 0.0 else 0.0
        b[t] = (omega + alpha * b[t-1]) * survival[t]

        # likelihood at t
        eps[t] = y[t] - mu[t] - b[t]
        ell[t] = fn_ll(eps[t], sigma2, theta_d)

        # GAS update
        s[t] = fn_s(eps[t], sigma2, theta_s)
        mu[t+1] = delta + beta * mu[t] + gamma * s[t]

    return mu[:T], b, survival, eps, ell






