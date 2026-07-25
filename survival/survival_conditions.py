import numba as jit


@jit(nopython=True, cache=True, fastmath=True)
def g1(x, mu, b, theta_g):
    c = theta_g[0]
    return x - c


@jit(nopython=True, cache=True, fastmath=True)
def g2(x, mu, b, theta_g):
    c = theta_g[0]
    return mu - c


@jit(nopython=True, cache=True, fastmath=True)
def g3(x, mu, b, theta_g):
    c = theta_g[0]
    return b - c * x