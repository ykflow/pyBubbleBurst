from abc import ABC, abstractmethod
import numpy as np


class AbstractParameterTransformer(ABC):
    """Abstract Base Class for managing model parameter transformations,
    unconstrained mappings, and MLE initializations.
    """

    @abstractmethod
    def transform(self, untransformed_vector: np.ndarray) -> np.ndarray:
        """Map constrained model domain parameters into unconstrained R space."""
        pass

    @abstractmethod
    def untransform(self, transformed_vector: np.ndarray) -> np.ndarray:
        """Map unconstrained R optimizer spaces back to valid model domains."""
        pass

    @abstractmethod
    def to_kwargs(self, untransformed_vector: np.ndarray) -> dict:
        """Convert a flat parameter vector into an argument dictionary for run_filter."""
        pass

    @abstractmethod
    def get_mle_inits(self) -> np.ndarray:
        """Return the default unconstrained initialization vector for the optimizer."""
        pass

    @abstractmethod
    def split_vector(self, full_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Splits a full flat vector into discontinuous (lmbda) and continuous (theta) sub-vectors."""
        pass

    @abstractmethod
    def merge_params(self, lmbda_vector: np.ndarray, theta_vector: np.ndarray) -> np.ndarray:
        """Reconstructs the full flat vector from isolated discontinuous (lmbda) and continuous (theta) parts."""
        pass

    # Universal math helper mappings for unconstrained optimization
    def from_pos_to_r(self, val: np.ndarray) -> np.ndarray:
        """Domain (0, inf) -> R"""
        return np.log(val)

    def from_r_to_pos(self, val: np.ndarray) -> np.ndarray:
        """Domain R -> (0, inf)"""
        return np.exp(val)

    def from_1_plus_to_r(self, val: np.ndarray) -> np.ndarray:
        """Domain (1, inf) -> R"""
        return np.log(val - 1.0)

    def from_r_to_1_plus(self, val: np.ndarray) -> np.ndarray:
        """Domain R -> (1, inf)"""
        return np.exp(val) + 1.0

    def from_0_1_to_r(self, val: np.ndarray) -> np.ndarray:
        """Domain (0, 1) -> R (Logit link function)"""
        return np.log(val / (1.0 - val))

    def from_r_to_0_1(self, val: np.ndarray) -> np.ndarray:
        """Domain R -> (0, 1) (Sigmoid link function)"""
        return np.exp(val) / (1.0 + np.exp(val))
