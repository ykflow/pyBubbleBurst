from abc import ABC, abstractmethod
import numpy as np


class AbstractFilters(ABC):
    """Abstract Base Class for numerical time-series filters."""

    def __init__(self):
        self.mu = None
        self.b = None
        self.survival = None
        self.margin = None
        self.threshold = None
        self.eps = None
        self.ell = None

    @abstractmethod
    def run_filter(self, *args, **kwargs) -> None:
        """Execute the filtering loop over the input time series.

        Accepts any flexible arguments to accommodate diverse filter shapes.
        """
        ...

    def get_log_likelihood(self) -> np.ndarray:
        """Return the calculated log-likelihood array from the last run."""
        if self.ell is None:
            raise ValueError("Filter has not been run yet. Call run_filter() first.")
        return self.ell
