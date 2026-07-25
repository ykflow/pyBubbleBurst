from abc import ABC, abstractmethod


class AbstractSimulator(ABC):
    """Abstract Base Class for simulation of time-series models."""

    def __init__(self):
        self.y = None
        self.mu = None
        self.b = None
        self.survival = None
        self.margin = None
        self.threshold = None
        self.eps = None
        self.ell = None

    @abstractmethod
    def generate(self, *args, **kwargs) -> None:
        """Generates time-series based on bubble filters.

        Accepts any flexible arguments to accommodate diverse filter shapes.
        """
        ...
