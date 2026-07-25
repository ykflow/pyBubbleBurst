import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def tri_gamma(x):
    val = 0.0
    while x < 8.0:
        val += 1.0 / (x * x)
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv

    val += (inv + 0.5 * inv2
            + inv2 * inv / 6.0
            - inv2 * inv2 * inv / 30.0
            + inv2 * inv2 * inv2 * inv / 42.0)

    return val


@jit(nopython=True, cache=True, fastmath=True)
def log_gamma(x):
    # Lanczos approximation coefficients (g=7, n=9)
    c0 = 0.99999999999980993
    c1 = 676.5203681218851
    c2 = -1259.1392167224028
    c3 = 771.32342877765313
    c4 = -176.61502916214059
    c5 = 12.507343278686905
    c6 = -0.13857109526572012
    c7 = 9.9843695780195716e-6
    c8 = 1.5056327351493116e-7

    if x <= 0.0:
        return np.nan

    # Reflection formula for x < 0.5
    if x < 0.5:
        val =  np.log(np.pi) - np.log(np.sin(np.pi * x)) - log_gamma(1.0 - x)
        return val

    z = x - 1.0

    a = c0
    a += c1 / (z + 1.0)
    a += c2 / (z + 2.0)
    a += c3 / (z + 3.0)
    a += c4 / (z + 4.0)
    a += c5 / (z + 5.0)
    a += c6 / (z + 6.0)
    a += c7 / (z + 7.0)
    a += c8 / (z + 8.0)

    t = z + 7.5

    val = 0.5 * np.log(2.0 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)
    return val