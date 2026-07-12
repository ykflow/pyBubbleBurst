import numpy as np
from numba import jit
from filters.filters_factory import FiltersFactory
from filters.abstract_filters import AbstractFilters


@FiltersFactory.register("local_explosions_e4")
class LocalExplosionsFilter(AbstractFilters):
    def __init__(self):
        super().__init__()
        self.mu = None
        self.b = None
        self.survival = None
        self.margin = None
        self.threshold = None
        self.eps = None
        self.ell = None

    def run_filter(self, y, delta, beta, gamma, omega, alpha, kappa, c, sigma2):
        mu, b, survival, margin, threshold_, eps, ell = local_explosions_filter(
            y, delta, beta, gamma, omega, alpha, kappa, c, sigma2)
        self.mu = mu
        self.b = b
        self.survival = survival
        self.margin = margin
        self.threshold = threshold_
        self.eps = eps
        self.ell = ell

    def get_log_likelihood(self):
        return self.ell


# @jit(nopython=True, cache=True)
def local_explosions_filter(y, delta, beta, gamma, omega, alpha, kappa, c, sigma2):
    # set-up
    T = len(y)
    mu = np.zeros(T, dtype=np.float64)
    b = np.zeros(T, dtype=np.float64)
    survival = np.zeros(T, dtype=np.float64)
    margin = np.zeros(T, dtype=np.float64)
    threshold_ = np.zeros(T, dtype=np.float64)
    eps = np.zeros(T, dtype=np.float64)
    ell = np.zeros(T, dtype=np.float64)

    # initializations
    mu[0] = np.mean(y[:10])
    # b[0] = 0.
    log_2pi_sigma2 = np.log(2. * np.pi * sigma2)

    # filter step
    for t in range(1,T):
        mu[t] = delta + beta * mu[t-1] + gamma * (y[t-1] - mu[t-1] - b[t-1])
        threshold_[t] = kappa * (mu[t] - c) # E4 threshold
        margin[t] = b[t-1] - threshold_[t]
        survival[t] = 1.0 if margin[t] < 0.0 else 0.0
        b[t] = (omega + alpha * b[t-1]) * survival[t-1]
        eps[t] = y[t] - mu[t] - b[t]
        eps2 = eps[t] * eps[t]
        # print(sigma2)
        ell[t] = -0.5 * (log_2pi_sigma2 + eps2/sigma2)

    return mu, b, survival, margin, threshold_, eps, ell






