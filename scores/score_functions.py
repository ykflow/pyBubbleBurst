import numpy as np
from numba import jit
from utilities.numba_specials import tri_gamma


@jit(nopython=True, cache=True, fastmath=True)
def gaussian_score(eps, theta_d):
    sigma = theta_d[0]
    sigma2 = sigma * sigma
    return eps/sigma2


@jit(nopython=True, cache=True, fastmath=True)
def student_score(eps, theta_d):
    sigma = theta_d[0]
    v = theta_d[1]
    sigma2 = sigma * sigma

    eps2 = eps * eps
    tmp1 = (v + 1) / v
    tmp2 = sigma2 + eps2 / v
    return tmp1 * (eps / tmp2)


@jit(nopython=True, cache=True, fastmath=True)
def egb2_score(eps, theta_d):
    sigma = theta_d[0]
    xi = theta_d[1]
    zeta = theta_d[1]
    sigma2 = sigma * sigma

    sigma = sigma2 ** .5
    e = eps / sigma
    h = (tri_gamma(xi) + tri_gamma(zeta)) ** .5

    tmp1 = h * (xi + zeta)
    tmp2 = np.exp(h * e)
    tmp3 = tmp2 / (1 + tmp2)

    return tmp1 * tmp3 / sigma




