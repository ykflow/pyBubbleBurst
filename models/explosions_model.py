import numpy as np
import pandas as pd
from filters.abstract_filters import AbstractFilters
from filters.filters_factory import FiltersFactory
from parameters.local_explosions_parameters import LocalExplosionsParameters
from estimation.mle import MaximumLikelihoodEstimation
from utilities.utils import detrend_linear


class DynamicLocalExplosionsModel:
    filter_instance: AbstractFilters

    def __init__(self, y: np.ndarray, bubble_type: str = "E4", detrend:bool=True):
        self.y = np.asarray(y, dtype=np.float64).flatten()

        raw_y = np.asarray(y, dtype=np.float64).flatten()
        if detrend:
            self.y = detrend_linear(raw_y)
        else:
            self.y = raw_y

        self.bubble_type = bubble_type.upper()
        factory_key = f"local_explosions_{bubble_type.lower()}"
        self.filter_instance = FiltersFactory.create(factory_key)

        if self.bubble_type == "E4":
            self.transformer = LocalExplosionsParameters()
        else:
            raise ValueError(f"Unsupported bubble model specification type: '{bubble_type}'")

        self.mle_summary = None
        self.estimated_params = None
        self.states = None

    def fit(self, mle_inits: np.ndarray = None, burn_in: int = 10, max_iter_mle: int = 500):
        """
        Executes the unconstrained Maximum Likelihood Estimation routine to discover
        the optimal parameters for this specific time-series.
        """

        mle_engine = MaximumLikelihoodEstimation(y=self.y,  filter_instance=self.filter_instance,
                                                 transformer=self.transformer, max_iter_mle=max_iter_mle,
                                                 burn_in=burn_in)

        # Override initial starting points if the user provides custom values
        if mle_inits is not None:
            mle_engine.mle_inits = self.transformer.transform(np.asarray(mle_inits, dtype=np.float64))

        mle_engine.estimate()
        self.mle_summary = mle_engine.mle_optim_results
        self.estimated_params = mle_engine.estimated_params

        # Automatically run the filter loop one last time using the discovered optimal states
        self.run_filter()
        return self

    def run_filter(self, custom_params: dict = None) -> pd.DataFrame:
        """
        Runs the filtering loop explicitly. Can be used after fit() using optimal parameters,
        or driven directly by passing a custom dictionary of configuration values.
        """
        if custom_params is not None:
            params_to_use = custom_params
        elif self.estimated_params is not None:
            params_to_use = self.estimated_params
        else:
            unconstrained_inits = self.transformer.get_mle_inits()
            params_to_use = self.transformer.to_kwargs(self.transformer.untransform(unconstrained_inits))

        payload = {"y": self.y, **params_to_use}
        self.filter_instance.run_filter(**payload)
        self.states = pd.DataFrame({
            "y": self.y,
            "mu": self.filter_instance.mu,
            "b": self.filter_instance.b,
            "survival": self.filter_instance.survival,
            "margin": self.filter_instance.margin,
            "threshold": self.filter_instance.threshold,
            "eps": self.filter_instance.eps,
            "ell": self.filter_instance.get_log_likelihood()})
        return self.states
