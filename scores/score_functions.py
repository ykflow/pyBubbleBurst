import numpy as np
from numba import jit
from utilities.numba_specials import tri_gamma


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_score(eps, sigma2, theta_s=None):
    return eps/sigma2


@jit(nopython=True, cache=True, fastmath=True)
def student_score(eps, sigma2, theta_s):
    v = theta_s[0]
    eps2 = eps * eps
    tmp1 = (v + 1) / v
    tmp2 = sigma2 + eps2 / v
    return tmp1 * (eps / tmp2)


@jit(nopython=True, cache=True, fastmath=True)
def egb2_score(eps, sigma2, theta_s):
    z1, z2 = theta_s
    sigma = sigma2 ** .5
    sqrt_eps = eps / sigma
    h = (tri_gamma(z1) + tri_gamma(z2)) ** .5

    tmp1 = h * (z1 + z2)
    tmp2 = np.exp(h * sqrt_eps)
    tmp3 = tmp2 / (1 + tmp2)

    return tmp1 * tmp3 / sigma




