import numpy as np
from numba import jit
from simulation.simulation_factory import SimulationFactory
from simulation.abstract_simulation import AbstractSimulator


@SimulationFactory.register("e4")
class E4BubbleModelSimulator(AbstractSimulator):
    def __init__(self):
        super().__init__()
        self.y = None
        self.mu = None
        self.b = None
        self.survival = None
        self.margin = None
        self.threshold = None
        self.eps = None
        self.ell = None

    def generate(self, T, burn_in, delta, beta, gamma, omega, alpha, kappa, c, sigma2):
        eps_ = np.random.normal(size=T+burn_in) * np.sqrt(sigma2)
        sim_args = (delta, beta, gamma, omega, alpha, kappa, c, sigma2, eps_, burn_in)
        y, mu, b, survival, margin, threshold_, eps, ell = simulate_e4_model(*sim_args)

        self.y = y
        self.mu = mu
        self.b = b
        self.survival = survival
        self.margin = margin
        self.threshold = threshold_
        self.eps = eps
        self.ell = ell


@jit(nopython=True, cache=True, error_model='numpy')
def simulate_e4_model(delta, beta, gamma, omega, alpha, kappa, c, sigma2, eps, burn_in):
    # set-up
    T = len(eps)
    y = np.zeros(T, dtype=np.float64)
    mu = np.zeros(T, dtype=np.float64)
    b = np.zeros(T, dtype=np.float64)
    survival = np.zeros(T, dtype=np.float64)
    margin = np.zeros(T, dtype=np.float64)
    threshold_ = np.zeros(T, dtype=np.float64)
    ell = np.zeros(T, dtype=np.float64)

    # initializations
    mu[0] = eps[0]
    y[0] = np.exp(eps[1])
    log_2pi_sigma2 = np.log(2. * np.pi * sigma2)

    # filter step
    for t in range(1,T):
        mu[t] = delta + beta * mu[t-1] + gamma * eps[t-1]
        threshold_[t] = kappa * (mu[t-1] - c) #typo in paper
        margin[t] = b[t-1] - threshold_[t]
        survival[t] = 1.0 if margin[t] < 0.0 else 0.0
        b[t] = (omega + alpha * b[t-1]) * survival[t-1]
        y[t] = mu[t] + b[t] + eps[t]
        eps2 = eps[t] * eps[t]
        # print(sigma2)
        ell[t] = -0.5 * (log_2pi_sigma2 + eps2/sigma2)

    return (y[burn_in:], mu[burn_in:], b[burn_in:], survival[burn_in:], margin[burn_in:], threshold_[burn_in:],
            eps[burn_in:], ell[burn_in:])






