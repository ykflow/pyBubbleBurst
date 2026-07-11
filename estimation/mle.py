import numpy as np
from scipy.optimize import minimize, approx_fprime
from filters.abstract_filters import AbstractFilters
from parameters.abstract_parameters import AbstractParameterTransformer
from time import time


class MaximumLikelihoodEstimation:
    def __init__(self, y: np.ndarray, filter_instance: AbstractFilters, transformer: AbstractParameterTransformer,
                 max_iter_mle: int = 500, burn_in: int = 1):
        self.y = y
        self.T = len(y)
        self.filter = filter_instance
        self.transformer = transformer
        self.max_iter_mle = max_iter_mle
        self.burn_in = burn_in
        self.run_time = None
        self.mle_optim_results = None
        self.estimated_params = None

        # Extract unconstrained initializations automatically from the transformer
        self.mle_inits = self.transformer.get_mle_inits()

    def _loglike(self, kwargs_ll: dict) -> np.ndarray:
        try:
            filter_arguments = {"y": self.y, **kwargs_ll}
            self.filter.run_filter(**filter_arguments)
            ell_vector = self.filter.get_log_likelihood()
            return ell_vector[self.burn_in:].sum()
        except Exception:
            return np.array([1e100])

    def _objf(self, transformed_params: np.ndarray) -> np.ndarray:
        untransformed_params = self.transformer.untransform(transformed_params)
        kwargs_ll = self.transformer.to_kwargs(untransformed_params)
        sum_ll = self._loglike(kwargs_ll=kwargs_ll)
        objective_value = -sum_ll / self.T

        print(f"Objective value: {objective_value:.6f}")
        return objective_value

    def estimate(self):
        func = self._objf
        grad = lambda params: approx_fprime(params, func, 6.5e-6)

        start = time()
        self.mle_optim_results = minimize(fun=func, x0=self.mle_inits, jac=grad, method="SLSQP",
                                           options={"maxiter": self.max_iter_mle})
        end = time()
        self.run_time = end - start

        self.estimated_params = self.transformer.to_kwargs(self.transformer.untransform(self.mle_optim_results.x))
