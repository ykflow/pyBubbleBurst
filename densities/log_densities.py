import numpy as np
from numba import jit
from utilities.numba_specials import log_gamma, tri_gamma


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_loglik(eps, sigma2, theta_d=None):
    return -0.5 * (np.log(2.0 * np.pi * sigma2)
                   + eps * eps / sigma2
                   )


@jit(nopython=True, cache=True, fastmath=True)
def student_loglik(eps, sigma2, theta_d):
    nu = theta_d[0]
    eps2 = eps * eps
    return (log_gamma((nu + 1.0) / 2.0)
            - log_gamma(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * sigma2)
            - ((nu + 1.0) / 2.0)
            * np.log(1.0 + eps2 / (nu * sigma2))
            )


@jit(nopython=True, cache=True, fastmath=True)
def egb2_loglik(eps, sigma2, theta_d):
    z1 = theta_d[0]
    z2 = theta_d[1]

    sigma = np.sqrt(sigma2)
    h = np.sqrt(tri_gamma(z1) + tri_gamma(z2))
    x = h * eps / sigma
    log_beta = log_gamma(z1) + log_gamma(z2) - log_gamma(z1 + z2)

    return (np.log(h) - np.log(sigma) - log_beta
            - z1 * x  - (z1 + z2) * np.log(1.0 + np.exp(-x))
            )