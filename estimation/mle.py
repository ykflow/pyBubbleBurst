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

    def _loglike(self, kwargs_ll: dict) -> float:
        try:
            filter_arguments = {"y": self.y, **kwargs_ll}
            self.filter.run_filter(**filter_arguments)
            ell_vector = self.filter.get_log_likelihood()
            return ell_vector[self.burn_in:].sum()
        except Exception:
            return 1e100

    def _objf(self, transformed_params: np.ndarray) -> float:
        untransformed_params = self.transformer.untransform(transformed_params)
        kwargs_ll = self.transformer.to_kwargs(untransformed_params)
        sum_ll = self._loglike(kwargs_ll=kwargs_ll)
        objective_value = -sum_ll #/ self.T

        # print(f"Objective value: {objective_value:.6f}")
        return objective_value

    def estimate(self):
        lmbda_inits, theta_inits = self.transformer.split_vector(self.mle_inits)
        inner_loop_state = {"best_theta_trans": theta_inits}

        def inner_objective(theta_vars_transformed: np.ndarray, current_lmbda_transformed: np.ndarray) -> float:
            # SLSQP over continuous parameters theta
            full_trans = self.transformer.merge_params(current_lmbda_transformed, theta_vars_transformed)
            return self._objf(full_trans)

        # Finite-difference gradient over the theta parameter dimensions only
        inner_gradient = lambda theta_vars, lmbda_vars: approx_fprime(
            theta_vars, lambda tv: inner_objective(tv, lmbda_vars), 6.5e-6)

        def outer_objective(lmbda_vars_transformed: np.ndarray) -> float:
            # Nelder-Mead over discontinuous parameters lmbda
            res_inner = minimize(
                fun=inner_objective,
                x0=inner_loop_state["best_theta_trans"],
                args=(lmbda_vars_transformed),
                # jac=inner_gradient,
                method="Nelder-Mead",
                options={"maxiter": 500}  # Capped inner steps for performance
            )

            # Update hot-start memory tracking for theta
            if res_inner.success:
                inner_loop_state["best_theta_trans"] = res_inner.x

            return res_inner.fun

        start = time()
        self.mle_optim_results = minimize(fun=outer_objective, x0=lmbda_inits,
                                          method="Nelder-Mead",
                                          options={"maxiter": self.max_iter_mle})
        end = time()
        self.run_time = end - start

        optim_lmbda_trans = self.mle_optim_results.x
        optim_theta_trans = inner_loop_state["best_theta_trans"]
        optim_params_trans = self.transformer.merge_params(optim_lmbda_trans, optim_theta_trans)
        self.estimated_params = self.transformer.to_kwargs(self.transformer.untransform(optim_params_trans))
        self.mle_optim_results.x = optim_params_trans
